package treed

import (
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
	LR float64

	// WeightDecay is the L2 penalty for optimization.
	WeightDecay float64

	// Iters is the number of optimization iterations to perform.
	Iters int
}

func (t *TAO[F, C, T]) Optimize(
	tree *Tree[F, C, T],
	coords []C,
	labels []T,
) TAOResult[F, C, T] {
	coords = append([]C{}, coords...)
	labels = append([]T{}, labels...)
	return t.optimize(tree, coords, labels)
}

func (t *TAO[F, C, T]) optimize(
	tree *Tree[F, C, T],
	coords []C,
	labels []T,
) TAOResult[F, C, T] {
	if len(coords) == 0 {
		return TAOResult[F, C, T]{
			Tree: tree,
		}
	}

	if tree.IsLeaf() {
		return t.optimizeLeaf(tree, coords, labels)
	}

	oldLoss := t.evaluateLoss(tree, coords, labels)

	// Note that this has side-effects. In particular, coords and labels are
	// re-ordered to split the decision boundary.
	splitIdx := t.splitDecision(tree.Axis, tree.Threshold, coords, labels)
	leftResult := t.optimize(tree.LessThan, coords[:splitIdx], labels[:splitIdx])
	rightResult := t.optimize(tree.GreaterEqual, coords[splitIdx:], labels[splitIdx:])

	targets := make([]bool, len(coords))
	weights := make([]float64, len(coords))
	for i, c := range coords {
		label := labels[i]
		leftLoss := t.Loss.Loss(label, leftResult.Tree.Apply(c))
		rightLoss := t.Loss.Loss(label, rightResult.Tree.Apply(c))
		targets[i] = leftLoss < rightLoss
		weights[i] = math.Abs(leftLoss - rightLoss)
	}
	newWeight, newBias := t.linearSVM(tree.Axis, -tree.Threshold, coords, targets, weights)
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
	newLoss := t.evaluateLoss(newTree, coords, labels)
	alternativeNewLoss := t.evaluateLoss(alternativeNewTree, coords, labels)
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
	oldLoss := t.evaluateLoss(tree, coords, labels)
	newLeaf := &Tree[F, C, T]{
		Leaf: t.Loss.Predict(NewListSlice(labels)),
	}
	newLoss := t.evaluateLoss(newLeaf, coords, labels)
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

func (t *TAO[F, C, T]) evaluateLoss(tree *Tree[F, C, T], coords []C, labels []T) float64 {
	var total float64
	for i, c := range coords {
		label := labels[i]
		prediction := tree.Apply(c)
		total += t.Loss.Loss(label, prediction)
	}
	return total
}

func (t *TAO[F, C, T]) splitDecision(axis C, threshold F, coords []C, labels []T) int {
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

func (t *TAO[F, C, T]) linearSVM(init C, bias F, coords []C, targets []bool, weights []float64) (C, F) {
	// TODO: implement this.
	return init, bias
}
