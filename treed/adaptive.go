package treed

import (
	"math"

	"github.com/unixpickle/essentials"
	"golang.org/x/exp/constraints"
)

const DefaultAdaptiveGreedyTreeSampleIters = 20

// AdaptiveGreedyTree is like GreedyTree(), except that it re-generates the
// dataset at each branch according to an oracle.
//
// The building process starts with a set of initial coordinates which should
// cover the space of labels.
// At each iteration of building the tree, if there are fewer than minSamples
// coordinates which fall under this branch of the tree, more samples will be
// generated.
func AdaptiveGreedyTree[F constraints.Float, C Coord[F, C], T any](
	axes []C,
	initCoords []C,
	oracle func(c C) T,
	loss SplitLoss[F, T],
	sampler *HitAndRunSampler[F, C],
	minSamples int,
	concurrency int,
	maxDepth int,
) *Tree[F, C, T] {
	panic("not yet implemented")
}

func greedySearchSingle[F constraints.Float, C Coord[F, C], T any](
	axes []C,
	coords []C,
	targets []T,
	loss SplitLoss[F, T],
	concurrency int,
) splitResult {
	best := splitResult{SplitInfo: SplitInfo{Loss: math.Inf(1)}}
	essentials.ReduceConcurrentMap(concurrency, len(axes), func() (func(i int), func()) {
		localBest := splitResult{SplitInfo: SplitInfo{Loss: math.Inf(1)}}

		localCoords := append([]C{}, coords...)
		localTargets := append([]T{}, targets...)
		localDots := make([]F, len(coords))
		process := func(i int) {
			axis := axes[i]
			for j, x := range localCoords {
				localDots[j] = x.Dot(axis)
			}
			essentials.VoodooSort(localDots, func(i, j int) bool {
				return localDots[i] < localDots[j]
			}, localTargets, localCoords)
			res := loss.MinimumSplit(NewListSlice(localTargets), NewListSlice(localDots))
			if res.Loss < localBest.Loss {
				localBest = splitResult{SplitInfo: res, Axis: i}
			}
		}
		reduce := func() {
			if localBest.Loss < best.Loss {
				best = localBest
			}
		}
		return process, reduce
	})
	return best
}
