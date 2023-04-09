package treed

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/model3d/model3d"
)

func TestHitAndRunSampler(t *testing.T) {
	sampler := &HitAndRunSampler[float64, model3d.Coord3D]{
		Iterations: 20,
	}

	// Create a mostly spherical polytope.
	var polytope Polytope[float64, model3d.Coord3D]
	model3d.NewMeshIcosphere(model3d.Origin, 1, 2).IterateVertices(func(c model3d.Coord3D) {
		polytope = append(polytope, Inequality[float64, model3d.Coord3D]{
			Axis: c,
			Max:  1.0,
		})
	})

	approxPoints := make([]model3d.Coord3D, 50000)
	r := rand.New(rand.NewSource(1337))
	start := model3d.XYZ(0.1, 0.1, 0.2) // arbitrary point inside sphere
	for i := range approxPoints {
		approxPoints[i] = sampler.Sample(r, polytope, start)
	}

	actualPoints := make([]model3d.Coord3D, 0, 50000)
	for len(actualPoints) < cap(actualPoints) {
		p := model3d.XYZ(r.Float64()*2-1, r.Float64()*2-1, r.Float64()*2-1)
		if p.Norm() <= 1 {
			actualPoints = append(actualPoints, p)
		}
	}

	approxNorm := meanNorm(approxPoints)
	actualNorm := meanNorm(actualPoints)
	if math.Abs(approxNorm-actualNorm) > 0.05 {
		t.Errorf("norm should be %f but got %f", actualNorm, approxNorm)
	}

	if m := meanPoint(approxPoints); m.Norm() > 0.01 {
		t.Errorf("mean should be 0 but got %v", m)
	}
}

func meanNorm(ps []model3d.Coord3D) float64 {
	var total float64
	for _, p := range ps {
		total += p.Norm()
	}
	return total / float64(len(ps))
}

func meanPoint(ps []model3d.Coord3D) model3d.Coord3D {
	var res model3d.Coord3D
	for _, p := range ps {
		res = res.Add(p)
	}
	return res.Scale(1 / float64(len(ps)))
}
