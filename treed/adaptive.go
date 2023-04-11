package treed

import (
	"math"
	"math/rand"

	"github.com/unixpickle/essentials"
	"golang.org/x/exp/constraints"
)

const DefaultAdaptiveGreedyTreeSampleIters = 20

type AxisSchedule[F constraints.Float, C Coord[F, C]] interface {
	Init() []C
	Next(stage int, axis C) []C
}

type ConstantAxisSchedule[F constraints.Float, C Coord[F, C]] []C

func (c ConstantAxisSchedule[F, C]) Init() []C {
	return c
}

func (c ConstantAxisSchedule[F, C]) Next(stage int, axis C) []C {
	return nil
}

// AdaptiveGreedyTree is like GreedyTree(), except that it re-generates the
// dataset at each branch according to an oracle.
//
// The building process starts with a set of initial coordinates which should
// cover the space of labels.
// At each iteration of building the tree, if there are fewer than minSamples
// coordinates which fall under this branch of the tree, more samples will be
// generated.
func AdaptiveGreedyTree[F constraints.Float, C Coord[F, C], T any](
	axisSchedule AxisSchedule[F, C],
	bounds Polytope[F, C],
	coords []C,
	targets []T,
	oracle func(c C) T,
	loss SplitLoss[F, T],
	sampler PolytopeSampler[F, C],
	minSamples int,
	concurrency int,
	maxDepth int,
) *Tree[F, C, T] {
	return adaptiveGreedyTree(
		axisSchedule,
		append(Polytope[F, C]{}, bounds...),
		append([]C{}, coords...),
		append([]T{}, targets...),
		oracle,
		loss,
		sampler,
		minSamples,
		concurrency,
		maxDepth,
	)
}

func adaptiveGreedyTree[F constraints.Float, C Coord[F, C], T any](
	axisSchedule AxisSchedule[F, C],
	bounds Polytope[F, C],
	coords []C,
	targets []T,
	oracle func(c C) T,
	loss SplitLoss[F, T],
	sampler PolytopeSampler[F, C],
	minSamples int,
	concurrency int,
	maxDepth int,
) *Tree[F, C, T] {
	if len(coords) == 0 {
		panic("need at least one point to generate more")
	}
	if len(coords) < minSamples {
		newCoords := make([]C, minSamples)
		copy(newCoords, coords)
		newTargets := make([]T, minSamples)
		copy(newTargets, targets)
		n := len(newCoords) - len(coords)
		essentials.StatefulConcurrentMap(concurrency, n, func() func(i int) {
			gen := rand.New(rand.NewSource(rand.Int63()))
			return func(i int) {
				p := coords[gen.Intn(len(coords))]
				c := sampler.Sample(gen, bounds, p)
				newCoords[i+len(coords)] = c
				newTargets[i+len(coords)] = oracle(c)
			}
		})
		coords, targets = newCoords, newTargets
	}

	axes := axisSchedule.Init()
	var bestSplit splitResult
	var bestThreshold F
	var bestAxis C
	for i := 0; len(axes) > 0; i++ {
		split, threshold := greedySearchSingle(
			axes,
			coords,
			targets,
			loss,
			concurrency,
		)
		if split.Index == 0 || split.Index == len(coords) {
			break
		} else if i == 0 || split.Loss < bestSplit.Loss {
			bestSplit = split
			bestAxis = axes[split.Axis]
			bestThreshold = threshold
		} else {
			break
		}
		axes = axisSchedule.Next(i, bestAxis)
	}
	if bestSplit.Index == 0 || bestSplit.Index == len(coords) {
		return &Tree[F, C, T]{
			Leaf: loss.Predict(NewListSlice(targets)),
		}
	}

	p1 := append(
		append(Polytope[F, C]{}, bounds...),
		Inequality[F, C]{Axis: bestAxis, Max: bestThreshold},
	)
	p2 := append(bounds, Inequality[F, C]{Axis: bestAxis.Scale(-1), Max: -bestThreshold})
	n := Partition(bestAxis, bestThreshold, coords, targets)
	t1 := adaptiveGreedyTree(
		axisSchedule,
		p1,
		coords[:n],
		targets[:n],
		oracle,
		loss,
		sampler,
		minSamples,
		concurrency,
		maxDepth,
	)
	t2 := adaptiveGreedyTree(
		axisSchedule,
		p2,
		coords[n:],
		targets[n:],
		oracle,
		loss,
		sampler,
		minSamples,
		concurrency,
		maxDepth,
	)
	return &Tree[F, C, T]{
		Axis:         bestAxis,
		Threshold:    bestThreshold,
		LessThan:     t1,
		GreaterEqual: t2,
	}
}

func greedySearchSingle[F constraints.Float, C Coord[F, C], T any](
	axes []C,
	coords []C,
	targets []T,
	loss SplitLoss[F, T],
	concurrency int,
) (splitResult, F) {
	best := splitResult{SplitInfo: SplitInfo{Loss: math.Inf(1)}}
	var bestThreshold F
	essentials.ReduceConcurrentMap(concurrency, len(axes), func() (func(i int), func()) {
		localBest := splitResult{SplitInfo: SplitInfo{Loss: math.Inf(1)}}
		var localThreshold F

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
				v1 := localDots[res.Index-1]
				v2 := localDots[res.Index]
				if v1 == v2 {
					panic("split should not exist at equal values")
				}
				localThreshold = (v1 + v2) / 2
			}
		}
		reduce := func() {
			if localBest.Loss < best.Loss {
				best = localBest
				bestThreshold = localThreshold
			}
		}
		return process, reduce
	})
	return best, bestThreshold
}
