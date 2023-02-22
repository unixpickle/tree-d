package treed

import (
	"math"
	"math/rand"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/model3d"
	"github.com/unixpickle/model3d/render3d"
)

func SampleDecisionBoundary(
	t *Tree[float64, model3d.Coord3D, bool],
	numPoints, gridSize int,
	min, max model3d.Coord3D,
) []model3d.Coord3D {
	rotation := model3d.Rotation(model3d.NewCoord3DRandUnit(), rand.Float64()*math.Pi*2)
	solid := model3d.CheckedFuncSolid(
		min,
		max,
		func(c model3d.Coord3D) bool {
			return t.Predict(c)
		},
	)
	rotatedSolid := model3d.TransformSolid(rotation, solid)

	maxSize := rotatedSolid.Max().Sub(rotatedSolid.Min()).MaxCoord()
	mesh := model3d.MarchingCubesSearch(rotatedSolid, maxSize/float64(gridSize), 8)
	mesh = mesh.Transform(rotation.Inverse())

	res := make([]model3d.Coord3D, numPoints)
	essentials.StatefulConcurrentMap(0, numPoints, func() func(int) {
		sampler := pointSampler(mesh)
		return func(i int) {
			res[i] = sampler()
		}
	})

	return res
}

func pointSampler(mesh *model3d.Mesh) func() model3d.Coord3D {
	gen := rand.New(rand.NewSource(rand.Int63()))
	light := render3d.NewMeshAreaLight(mesh, render3d.NewColor(1))
	return func() model3d.Coord3D {
		point, _, _ := light.SampleLight(gen)
		return point
	}
}
