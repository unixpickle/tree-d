package treed

import (
	"log"
	"math"

	"golang.org/x/exp/constraints"
)

type TAOResult[F constraints.Float, C Coord[F, C], T any] struct {
	Tree    *Tree[F, C, T]
	OldLoss float64
	NewLoss float64
}

type TAO[F constraints.Float, C Coord[F, C], T any] struct {
	// Loss is the loss function to minimize.
	Loss TAOLoss[T]

	// LR is the step size for optimization.
	LR F

	// WeightDecay is the L2 penalty for optimization.
	WeightDecay F

	// Momentum is the Nesterov momentum coefficient.
	Momentum F

	// Iters is the number of optimization iterations to perform.
	Iters int

	// Concurrency, if specified, controls the maximum number of Goroutines to
	// use for optimization.
	Concurrency int

	// Verbose, if true, enables printing during training.
	Verbose bool

	// The following flags are for adaptive resampling, which is disabled by
	// default. Set MinSamples > 0 to enable adaptive resampling.

	// MinSamples is the minimum number of samples to use at each level.
	// If there are fewer samples than this, more are sampled using the bounds
	// and oracle.
	//
	// If this is 0, then adaptive sampling is disabled.
	MinSamples int

	// Sampler is used to sample new points in a polytope.
	// It is only used for adaptive sampling.
	Sampler PolytopeSampler[F, C]

	// Bounds is the initial bounds of the decision space.
	// It is only used for adaptive resampling.
	Bounds Polytope[F, C]

	// Oracle is used to compute new labels for active sampling.
	Oracle func(C) T
}

func (t *TAO[F, C, T]) Optimize(
	tree *Tree[F, C, T],
	coords []C,
	labels []T,
) TAOResult[F, C, T] {
	coords = append([]C{}, coords...)
	labels = append([]T{}, labels...)
	bounds := append(Polytope[F, C]{}, t.Bounds...)
	q := newForkQueue[TAOResult[F, C, T]](t.Concurrency)
	return q.Run(func() TAOResult[F, C, T] {
		return t.optimize(q, tree, bounds, coords, labels)
	})
}

func (t *TAO[F, C, T]) optimize(
	queue *forkQueue[TAOResult[F, C, T]],
	tree *Tree[F, C, T],
	bounds Polytope[F, C],
	coords []C,
	labels []T,
) TAOResult[F, C, T] {
	if len(coords) == 0 {
		return TAOResult[F, C, T]{
			Tree: tree,
		}
	}

	coords, labels = adaptiveResample(
		bounds,
		coords,
		labels,
		t.Oracle,
		t.Sampler,
		t.MinSamples,
		1,
	)

	if tree.IsLeaf() {
		return t.optimizeLeaf(tree, coords, labels)
	}

	oldLoss := t.EvaluateLoss(tree, coords, labels)

	// Note that this has side-effects. In particular, coords and labels are
	// re-ordered to split the decision boundary.
	splitIdx := Partition(tree.Axis, tree.Threshold, coords, labels)
	leftResult, rightResult := queue.Fork(
		func() TAOResult[F, C, T] {
			return t.optimize(
				queue,
				tree.LessThan,
				bounds.Constrain(tree.Axis, tree.Threshold),
				coords[:splitIdx],
				labels[:splitIdx],
			)
		},
		func() TAOResult[F, C, T] {
			return t.optimize(
				queue,
				tree.GreaterEqual,
				bounds.Constrain(tree.Axis.Scale(-1), -tree.Threshold),
				coords[splitIdx:],
				labels[splitIdx:],
			)
		},
	)

	clsTargets := make([]bool, 0, len(coords))
	clsWeights := make([]F, 0, len(coords))
	clsCoords := make([]C, 0, len(coords))
	for i, c := range coords {
		label := labels[i]
		leftLoss := t.Loss.Loss(label, leftResult.Tree.Predict(c))
		rightLoss := t.Loss.Loss(label, rightResult.Tree.Predict(c))
		weight := (F)(math.Abs(leftLoss - rightLoss))
		if weight != 0 {
			clsTargets = append(clsTargets, leftLoss > rightLoss)
			clsWeights = append(clsWeights, weight)
			clsCoords = append(clsCoords, c)
		}
	}

	// If there's no benefit to be gained from updating the decision function,
	// we keep it the same and skip SVM training.
	if len(clsWeights) == 0 {
		if leftResult.Tree != tree.LessThan || rightResult.Tree != tree.GreaterEqual {
			newTree := &Tree[F, C, T]{
				Axis:         tree.Axis,
				Threshold:    tree.Threshold,
				LessThan:     leftResult.Tree,
				GreaterEqual: rightResult.Tree,
			}
			newLoss := t.EvaluateLoss(newTree, coords, labels)
			return TAOResult[F, C, T]{
				Tree:    tree,
				OldLoss: oldLoss,
				NewLoss: newLoss,
			}
		} else {
			return TAOResult[F, C, T]{
				Tree:    tree,
				OldLoss: oldLoss,
				NewLoss: oldLoss,
			}
		}
	}

	newWeight, newBias := t.linearSVM(tree.Axis, -tree.Threshold, clsCoords, clsTargets, clsWeights)
	newTree := &Tree[F, C, T]{
		Axis:         newWeight,
		Threshold:    -newBias,
		LessThan:     leftResult.Tree,
		GreaterEqual: rightResult.Tree,
	}
	alternativeNewTree := &Tree[F, C, T]{
		Axis:         tree.Axis,
		Threshold:    tree.Threshold,
		LessThan:     leftResult.Tree,
		GreaterEqual: rightResult.Tree,
	}
	newLoss := t.EvaluateLoss(newTree, coords, labels)
	alternativeNewLoss := t.EvaluateLoss(alternativeNewTree, coords, labels)
	if t.Verbose {
		log.Printf("old_loss=%f new_loss=%f alternative=%f", oldLoss, newLoss, alternativeNewLoss)
	}
	if newLoss >= alternativeNewLoss {
		if leftResult.Tree != tree.LessThan || rightResult.Tree != tree.GreaterEqual {
			newTree = alternativeNewTree
			newLoss = alternativeNewLoss
		} else {
			newTree = tree
			newLoss = oldLoss
		}
	}

	return TAOResult[F, C, T]{
		Tree:    newTree,
		OldLoss: oldLoss,
		NewLoss: newLoss,
	}
}

func (t *TAO[F, C, T]) optimizeLeaf(
	tree *Tree[F, C, T],
	coords []C,
	labels []T,
) TAOResult[F, C, T] {
	oldLoss := t.EvaluateLoss(tree, coords, labels)
	newLeaf := &Tree[F, C, T]{
		Leaf: t.Loss.Predict(NewListSlice(labels)),
	}
	newLoss := t.EvaluateLoss(newLeaf, coords, labels)
	if newLoss >= oldLoss {
		newLeaf = tree
		newLoss = oldLoss
	}
	return TAOResult[F, C, T]{
		Tree:    tree,
		OldLoss: oldLoss,
		NewLoss: newLoss,
	}
}

func (t *TAO[F, C, T]) EvaluateLoss(tree *Tree[F, C, T], coords []C, labels []T) float64 {
	var total float64
	for i, c := range coords {
		label := labels[i]
		prediction := tree.Predict(c)
		total += t.Loss.Loss(label, prediction)
	}
	return total
}

func (t *TAO[F, C, T]) linearSVM(w C, b F, coords []C, targets []bool, weights []F) (C, F) {
	scale := LineSearchScale[F](w, b, coords, targets, HingeLoss[F]{})
	w = w.Scale(scale)
	b *= scale

	result := LinearClassification[F, C](
		w,
		b,
		coords,
		targets,
		weights,
		HingeLoss[F]{},
		&SGDOptimizer[F, C]{
			LR:          t.LR,
			WeightDecay: t.WeightDecay,
			Momentum:    t.Momentum,
			AnnealIters: t.Iters,
		},
		t.Iters,
	)

	if t.Verbose {
		log.Printf(
			"SVM training: loss=%f->%f acc=%f->%f",
			result.InitLoss,
			result.FinalLoss,
			result.InitAcc,
			result.FinalAcc,
		)
	}

	return result.Weight, result.Bias
}

// Partition re-orders coords and labels such that all coords falling on the
// less-than side of an axis are before all coords falling on the greater-equal
// side. Returns the number of coords on the less-than side.
func Partition[F constraints.Float, C Coord[F, C], T any](
	axis C,
	threshold F,
	coords []C,
	labels []T,
) int {
	numPositive := 0
	for i := 0; i+numPositive < len(coords); i++ {
		x := coords[i]
		if axis.Dot(x) >= threshold {
			numPositive++
			endIdx := len(coords) - numPositive
			coords[i], coords[endIdx] = coords[endIdx], coords[i]
			labels[i], labels[endIdx] = labels[endIdx], labels[i]
			i--
		}
	}
	return len(coords) - numPositive
}
