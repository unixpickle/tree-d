package treed

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/model3d"
	"github.com/unixpickle/model3d/render3d"
)

// SampleDecisionBoundaryCast samples points near the surface defined by a
// boolean field using ray marching to find surface points.
//
// This may have an infinite runtime if the space is empty, since rays will
// never find points on the surface.
// To avoid this issue, specify maxTries to limit the number of ray queries.
func SampleDecisionBoundaryCast(
	b *BoundedSolidTree,
	numPoints int,
	maxQueries int,
) []model3d.Coord3D {
	remainingQueries := int64(maxQueries)

	resChan := make(chan model3d.Coord3D, numPoints)
	var wg sync.WaitGroup

	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			collider := NewCollider(b)
			min, max := collider.Min(), collider.Max()
			gen := rand.New(rand.NewSource(rand.Int63()))
			done := false
			for !done {
				if maxQueries != 0 {
					if atomic.AddInt64(&remainingQueries, -1) < 0 {
						return
					}
				}
				ray := &model3d.Ray{
					Origin: model3d.XYZ(
						gen.Float64(),
						gen.Float64(),
						gen.Float64(),
					).Mul(max.Sub(min)).Add(min),
					Direction: model3d.XYZ(
						gen.NormFloat64(),
						gen.NormFloat64(),
						gen.NormFloat64(),
					).Normalize(),
				}
				collider.RayCollisions(ray, func(rc model3d.RayCollision) {
					point := ray.Origin.Add(ray.Direction.Scale(rc.Scale))
					select {
					case resChan <- point:
					default:
						done = true
					}
				})
			}
		}()
	}
	wg.Wait()
	close(resChan)

	res := make([]model3d.Coord3D, 0, numPoints)
	for x := range resChan {
		res = append(res, x)
	}

	return res
}

// SampleDecisionBoundaryMesh samples points near the surface defined by a
// boolean field.
//
// This may miss thin parts of the decision surface, unlike
// SampleDecisionBoundaryCast().
func SampleDecisionBoundaryMesh(
	b *BoundedSolidTree,
	numPoints int,
	gridSize int,
) []model3d.Coord3D {
	rotation := model3d.Rotation(model3d.NewCoord3DRandUnit(), rand.Float64()*math.Pi*2)
	rotatedSolid := model3d.TransformSolid(rotation, TreeSolid(b))

	maxSize := rotatedSolid.Max().Sub(rotatedSolid.Min()).MaxCoord()
	mesh := model3d.MarchingCubesSearch(rotatedSolid, maxSize/float64(gridSize), 8)
	mesh = mesh.Transform(rotation.Inverse())

	res := make([]model3d.Coord3D, numPoints)
	essentials.StatefulConcurrentMap(0, numPoints, func() func(int) {
		sampler := MeshPointSampler(mesh)
		return func(i int) {
			res[i] = sampler()
		}
	})

	return res
}

func MeshPointSampler(mesh *model3d.Mesh) func() model3d.Coord3D {
	gen := rand.New(rand.NewSource(rand.Int63()))
	light := render3d.NewMeshAreaLight(mesh, render3d.NewColor(1))
	return func() model3d.Coord3D {
		point, _, _ := light.SampleLight(gen)
		return point
	}
}
