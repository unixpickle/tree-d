package treed

import (
	"log"
	"math"
	"runtime"
	"sync/atomic"

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
}

func (t *TAO[F, C, T]) Optimize(
	tree *Tree[F, C, T],
	coords []C,
	labels []T,
) TAOResult[F, C, T] {
	coords = append([]C{}, coords...)
	labels = append([]T{}, labels...)
	q := newForkQueue[TAOResult[F, C, T]](t.Concurrency)
	return q.Run(func() TAOResult[F, C, T] {
		return t.optimize(q, tree, coords, labels)
	})
}

func (t *TAO[F, C, T]) optimize(
	queue *forkQueue[TAOResult[F, C, T]],
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

	oldLoss := t.EvaluateLoss(tree, coords, labels)

	// Note that this has side-effects. In particular, coords and labels are
	// re-ordered to split the decision boundary.
	splitIdx := splitDecision(tree.Axis, tree.Threshold, coords, labels)
	leftResult, rightResult := queue.Fork(
		func() TAOResult[F, C, T] {
			return t.optimize(queue, tree.LessThan, coords[:splitIdx], labels[:splitIdx])
		},
		func() TAOResult[F, C, T] {
			return t.optimize(queue, tree.GreaterEqual, coords[splitIdx:], labels[splitIdx:])
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

func splitDecision[F constraints.Float, C Coord[F, C], T any](
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

type forkQueueTask[T any] struct {
	Started int32
	F       func() T
	Result  chan T
}

// A forkQueue lets you run recursive tasks with a fixed maximum number of
// Goroutines. To use the queue, create with newForkQueue(), then run the root
// task with Run(). Within a task, you may call Fork() to run two sub-tasks,
// potentially allowing separate goroutines to run the two sub-tasks.
type forkQueue[T any] struct {
	queue chan *forkQueueTask[T]
}

func newForkQueue[T any](numWorkers int) *forkQueue[T] {
	if numWorkers == 0 {
		numWorkers = runtime.GOMAXPROCS(0)
	}
	res := &forkQueue[T]{
		queue: make(chan *forkQueueTask[T], numWorkers*1000),
	}
	for i := 0; i < numWorkers; i++ {
		go res.worker()
	}
	return res
}

func (f *forkQueue[T]) Run(fn func() T) T {
	defer close(f.queue)
	task := &forkQueueTask[T]{F: fn, Result: make(chan T, 1)}
	f.queue <- task
	return <-task.Result
}

func (f *forkQueue[T]) Fork(fn1, fn2 func() T) (T, T) {
	task := &forkQueueTask[T]{F: fn2, Result: make(chan T, 1)}
	select {
	case f.queue <- task:
	default:
		// Prevent unbounded memory growth by running the task locally.
		task.Started = 1
		task.Result <- task.F()
	}
	result1 := fn1()
	var result2 T
	if atomic.SwapInt32(&task.Started, 1) == 0 {
		// Task hasn't been started yet, so we kick it off
		// on this worker.
		result2 = fn2()
	} else {
		result2 = <-task.Result
	}
	return result1, result2
}

func (f *forkQueue[T]) worker() {
	for task := range f.queue {
		if atomic.SwapInt32(&task.Started, 1) != 0 {
			// Task already started on a different thread.
			continue
		}
		task.Result <- task.F()
	}
}
