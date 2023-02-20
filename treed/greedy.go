package treed

import (
	"github.com/unixpickle/essentials"
	"golang.org/x/exp/constraints"
	"golang.org/x/exp/slices"
)

type greedySearchState[F constraints.Float, C Coord[F, C], T any] struct {
	Axes        []C
	Sorted      [][]*greedySearchNode[F, C, T]
	Loss        Loss[F, T]
	Value       T
	Concurrency int
}

func newGreedySearchState[F constraints.Float, C Coord[F, C], T any](
	axes []C,
	coords []C,
	labels []T,
	loss Loss[F, T],
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

func (g *greedySearchState[F, C, T]) MaybeSplit() *[2]*greedySearchState[F, C, T] {
	queries := make(chan int, len(g.Axes))
	for i := range g.Axes {
		queries <- i
	}
	close(queries)

	results := make(chan splitResult, len(g.Axes))
	for i := 0; i < g.Concurrency; i++ {
		go func() {
			for axis := range queries {
				sorted := g.Sorted[axis]
				labels := List[T]{
					Len: len(sorted),
					Get: func(i int) T {
						return sorted[i].Label
					},
				}
				thresholds := List[F]{
					Len: len(sorted),
					Get: func(i int) F {
						return sorted[i].Values[axis]
					},
				}
				splitInfo := g.Loss.MinimumSplit(labels, thresholds)
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
	res := g.Split(bestResult.Axis, bestResult.Index)
	return &res
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
