package treed

import (
	"runtime"

	"github.com/unixpickle/essentials"
	"golang.org/x/exp/constraints"
	"golang.org/x/exp/slices"
)

// GreedyTree builds a decision tree by greedily selecting a split at each
// branch which minimizes a loss.
//
// The specified axes are used as features (via a Dot product) for decisions at
// each branch of the tree. These need not be axis-aligned or normalized.
//
// Stops building a branch of the tree once a maximum depth is reached, or once
// the loss does not decrease for any split.
//
// The concurrency argument specifies the maximum number of Goroutines to use
// for search. No more than len(axes) Goroutines can be utilized at once.
// If concurrency is 0, GOMAXPROCS is used.
func GreedyTree[F constraints.Float, C Coord[F, C], T any](
	axes []C,
	coords []C,
	labels []T,
	loss SplitLoss[F, T],
	concurrency int,
	maxDepth int,
) *Tree[F, C, T] {
	if concurrency == 0 {
		concurrency = runtime.GOMAXPROCS(0)
	}
	return newGreedySearchState(
		axes,
		coords,
		labels,
		loss,
		concurrency,
	).Build(maxDepth)
}

type greedySearchState[F constraints.Float, C Coord[F, C], T any] struct {
	Axes        []C
	Sorted      [][]*greedySearchNode[F, C, T]
	Loss        SplitLoss[F, T]
	Value       T
	Concurrency int
}

func newGreedySearchState[F constraints.Float, C Coord[F, C], T any](
	axes []C,
	coords []C,
	labels []T,
	loss SplitLoss[F, T],
	concurrency int,
) *greedySearchState[F, C, T] {
	res := &greedySearchState[F, C, T]{
		Axes:        axes,
		Sorted:      make([][]*greedySearchNode[F, C, T], len(axes)),
		Loss:        loss,
		Concurrency: essentials.MinInt(concurrency, len(axes)),
	}

	nodes := make([]*greedySearchNode[F, C, T], len(coords))
	// Pack all values in contiguous array to avoid many tiny allocations.
	values := make([]F, len(coords)*len(axes))
	for i, c := range coords {
		for j, axis := range axes {
			values[i*len(axes)+j] = c.Dot(axis)
		}
		nodes[i] = &greedySearchNode[F, C, T]{
			Coord:  c,
			Values: values[i*len(axes) : (i+1)*len(axes)],
			Label:  labels[i],
		}
	}

	for i := range axes {
		sorted := append([]*greedySearchNode[F, C, T]{}, nodes...)
		slices.SortFunc(sorted, func(x, y *greedySearchNode[F, C, T]) bool {
			return x.Values[i] < y.Values[i]
		})
		res.Sorted[i] = sorted
	}

	return res
}

func (g *greedySearchState[F, C, T]) Build(maxDepth int) *Tree[F, C, T] {
	if maxDepth == 0 {
		return g.BuildLeaf()
	}

	queries := make(chan int, len(g.Axes))
	for i := range g.Axes {
		queries <- i
	}
	close(queries)

	results := make(chan splitResult, len(g.Axes))
	for i := 0; i < g.Concurrency; i++ {
		go func() {
			for axis := range queries {
				splitInfo := g.Loss.MinimumSplit(g.Labels(axis), g.Thresholds(axis))
				results <- splitResult{
					SplitInfo: splitInfo,
					Axis:      axis,
				}
			}
		}()
	}

	var bestResult splitResult
	for i := 0; i < len(g.Axes); i++ {
		result := <-results
		if i == 0 || result.Loss < bestResult.Loss {
			bestResult = result
		}
	}

	if bestResult.Index == 0 || bestResult.Index == len(g.Sorted[0]) {
		return nil
	}
	split := g.Split(bestResult.Axis, bestResult.Index)
	left := split[0].Build(maxDepth - 1)
	right := split[1].Build(maxDepth - 1)

	bestThresholds := g.Thresholds(bestResult.Axis)
	v1 := bestThresholds.Get(bestResult.Index - 1)
	v2 := bestThresholds.Get(bestResult.Index)
	return &Tree[F, C, T]{
		Axis:         g.Axes[bestResult.Axis],
		Threshold:    (v1 + v2) / 2,
		LessThan:     left,
		GreaterEqual: right,
	}
}

func (g *greedySearchState[F, C, T]) BuildLeaf() *Tree[F, C, T] {
	return &Tree[F, C, T]{
		Leaf: g.Loss.Predict(g.Labels(0)),
	}
}

func (g *greedySearchState[F, C, T]) Labels(axis int) funcList[T] {
	sorted := g.Sorted[axis]
	return funcList[T]{
		Len: len(sorted),
		Get: func(i int) T {
			return sorted[i].Label
		},
	}
}

func (g *greedySearchState[F, C, T]) Thresholds(axis int) funcList[F] {
	sorted := g.Sorted[axis]
	return funcList[F]{
		Len: len(sorted),
		Get: func(i int) F {
			return sorted[i].Values[axis]
		},
	}
}

func (g *greedySearchState[F, C, T]) Split(axis int, index int) [2]*greedySearchState[F, C, T] {
	for i, node := range g.Sorted[axis] {
		node.IsRight = i < index
	}

	var states [2]*greedySearchState[F, C, T]
	for i, count := range [2]int{index, len(g.Sorted[0]) - index} {
		isRight := i == 1
		state := &greedySearchState[F, C, T]{
			Axes:   g.Axes,
			Sorted: make([][]*greedySearchNode[F, C, T], len(g.Axes)),
		}
		for j := range g.Axes {
			subList := make([]*greedySearchNode[F, C, T], 0, count)
			for _, node := range g.Sorted[j] {
				if node.IsRight == isRight {
					subList = append(subList, node)
				}
			}
			state.Sorted[j] = subList
		}
		states[i] = state
	}
	return states
}

type greedySearchNode[F constraints.Float, C Coord[F, C], T any] struct {
	Coord  C
	Values []F
	Label  T

	// Temporary flag used for splitting all sorted arrays.
	IsRight bool
}

type splitResult struct {
	SplitInfo
	Axis int
}
