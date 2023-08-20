package treed

import (
	"math"
	"math/rand"
	"sync"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/model3d"
)

// MeshSurfaceTree builds a decision tree by greedily selecting a split on the
// surface of a mesh at each branch which minimizes a loss.
//
// This uses the same sampling strategy as AdaptiveGreedyTree to find more data
// within a branch of a tree.
//
// Stops building a branch of the tree once a maximum depth is reached, or once
// the loss does not decrease for any split.
//
// The concurrency argument specifies the maximum number of Goroutines to use
// for search. If concurrency is 0, GOMAXPROCS is used.
func MeshSurfaceTree[T any](
	mesh *model3d.Mesh,
	bounds Polytope[float64, model3d.Coord3D],
	coords []model3d.Coord3D,
	labels []T,
	oracle func(c model3d.Coord3D) T,
	loss SplitLoss[float64, T],
	sampler PolytopeSampler[float64, model3d.Coord3D],
	minSamples int,
	concurrency int,
	maxDepth int,
	maxSearchSplits int,
) *Tree[float64, model3d.Coord3D, T] {
	return meshSurfaceTree(
		mesh,
		bounds,
		append([]model3d.Coord3D{}, coords...),
		append([]T{}, labels...),
		oracle,
		loss,
		sampler,
		minSamples,
		concurrency,
		maxDepth,
		maxSearchSplits,
	)
}

func meshSurfaceTree[T any](
	mesh *model3d.Mesh,
	bounds Polytope[float64, model3d.Coord3D],
	coords []model3d.Coord3D,
	labels []T,
	oracle func(c model3d.Coord3D) T,
	loss SplitLoss[float64, T],
	sampler PolytopeSampler[float64, model3d.Coord3D],
	minSamples int,
	concurrency int,
	maxDepth int,
	maxSearchSplits int,
) *Tree[float64, model3d.Coord3D, T] {
	if maxDepth == 0 || mesh.NumTriangles() == 0 {
		return &Tree[float64, model3d.Coord3D, T]{
			Leaf: loss.Predict(NewListSlice(labels)),
		}
	}

	axes := make([]model3d.Coord3D, 0)
	thresholds := make([]float64, 0)
	planeSampler := newMeshPlaneSampler(mesh)
	for !planeSampler.Empty() && len(axes) < maxSearchSplits {
		normal, bias := planeSampler.Sample()
		axes = append(axes, normal)
		thresholds = append(thresholds, bias)
	}

	coords, labels = adaptiveResample(
		bounds,
		coords,
		labels,
		oracle,
		sampler,
		minSamples,
		concurrency,
	)

	var resultLock sync.Mutex
	var bestLoss float64
	var bestSplit int = -1

	essentials.StatefulConcurrentMap(concurrency, len(axes), func() func(int) {
		lessThan := make([]T, 0, len(coords))
		greaterEqual := make([]T, 0, len(coords))
		return func(i int) {
			axis := axes[i]
			threshold := thresholds[i]

			lessThan = lessThan[:0]
			greaterEqual = greaterEqual[:0]
			for j, c := range coords {
				label := labels[j]
				if c.Dot(axis) >= threshold {
					greaterEqual = append(greaterEqual, label)
				} else {
					lessThan = append(lessThan, label)
				}
			}

			// Don't entertain empty splits.
			if len(lessThan) == 0 || len(greaterEqual) == 0 {
				return
			}

			l := loss.SplitLoss(NewListSlice[T](lessThan), NewListSlice[T](greaterEqual))
			resultLock.Lock()
			defer resultLock.Unlock()
			if l < bestLoss || bestSplit == -1 {
				bestLoss = l
				bestSplit = i
			}
		}
	})

	if bestSplit == -1 {
		return &Tree[float64, model3d.Coord3D, T]{
			Leaf: loss.Predict(NewListSlice(labels)),
		}
	}

	axis := axes[bestSplit]
	threshold := thresholds[bestSplit]
	mesh1, mesh2 := SplitMesh(mesh, axis, threshold)
	p1 := bounds.Constrain(axis, threshold)
	p2 := bounds.Constrain(axis.Scale(-1), -threshold)

	n := Partition(axis, threshold, coords, labels)
	t1 := meshSurfaceTree(
		mesh1,
		p1,
		coords[:n],
		labels[:n],
		oracle,
		loss,
		sampler,
		minSamples,
		concurrency,
		maxDepth-1,
		maxSearchSplits,
	)
	t2 := meshSurfaceTree(
		mesh2,
		p2,
		coords[n:],
		labels[n:],
		oracle,
		loss,
		sampler,
		minSamples,
		concurrency,
		maxDepth-1,
		maxSearchSplits,
	)
	return &Tree[float64, model3d.Coord3D, T]{
		Axis:         axis,
		Threshold:    threshold,
		LessThan:     t1,
		GreaterEqual: t2,
	}
}

type meshPlaneSampler struct {
	triangles   []*model3d.Triangle
	weights     []float64
	totalWeight float64
}

func newMeshPlaneSampler(mesh *model3d.Mesh) *meshPlaneSampler {
	res := &meshPlaneSampler{}
	mesh.Iterate(func(t *model3d.Triangle) {
		area := t.Area()
		res.triangles = append(res.triangles, t)
		res.weights = append(res.weights, area)
		res.totalWeight = area
	})
	return res
}

func (m *meshPlaneSampler) Empty() bool {
	return len(m.triangles) == 0
}

func (m *meshPlaneSampler) Sample() (normal model3d.Coord3D, bias float64) {
	f := rand.Float64() * m.totalWeight
	for i, t := range m.triangles {
		f -= m.weights[i]
		if f < 0 || i == len(m.triangles)-1 {
			normal = t.Normal()
			bias = normal.Dot(t[0])
			break
		}
	}
	// Remove all coplanar triangles from the sampler.
	for i := 0; i < len(m.triangles); i++ {
		t := m.triangles[i]
		normal1 := t.Normal()
		bias1 := normal1.Dot(t[0])
		if math.Abs(normal1.Dot(normal)) > 1-1e-7 && math.Abs(bias-bias1) < 1e-7 {
			// Replace this triangle with the last one and truncate.
			m.totalWeight -= m.weights[i]
			m.weights[i] = m.weights[len(m.weights)-1]
			m.triangles[i] = m.triangles[len(m.triangles)-1]
			m.weights = m.weights[:len(m.weights)-1]
			m.triangles = m.triangles[:len(m.triangles)-1]
			i--
		}
	}
	return
}
