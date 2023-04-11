package treed

import "golang.org/x/exp/constraints"

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
