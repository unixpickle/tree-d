package treed

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/unixpickle/model3d/model2d"
	"github.com/unixpickle/model3d/model3d"
	"golang.org/x/exp/constraints"
)

const DefaultHitAndRunEpsilon = 1e-5

// A HitAndRunSampler samples coordinates within polytopes using hit and run
// sampling.
type HitAndRunSampler[F constraints.Float, C Coord[F, C]] struct {
	// Iterations is the number of monte carlo steps to take.
	// More iterations takes longer to run, but gives more uniform samples.
	Iterations int

	// RandDirection generates random directions of type C.
	// This needn't be passed if C is model3d.Coord2D or model3d.Coord3D.
	RandDirection func(r *rand.Rand) C

	// Epsilon is a small number used to prevent points from exiting the
	// bounds. If zero, DefaultHitAndRunEpsilon is used.
	Epsilon float64
}

// Sample produces a random point inside the polytope, starting the sampling
// process at some initial point that must be within the polytope.
func (h *HitAndRunSampler[F, C]) Sample(r *rand.Rand, p Polytope[F, C], init C) C {
	eps := h.Epsilon
	if eps == 0 {
		eps = DefaultHitAndRunEpsilon
	}
	dirSampler := h.randDir()
	cur := init
	for i := 0; i < h.Iterations; i++ {
		d := dirSampler(r)
		negT, posT := p.Cast(cur, d)

		if math.IsInf(float64(negT), 0) || math.IsInf(float64(posT), 0) {
			panic("polytope is not closed or we ended up outside of it")
		}

		// Epsilon is added to avoid hitting the edge of the polytope
		// and going outside of it due to rounding errors.
		t := F(r.Float64()*(1-2*eps)+eps)*(posT-negT) + negT

		cur = cur.Add(d.Scale(t))
	}
	return cur
}

func (h *HitAndRunSampler[F, C]) randDir() func(r *rand.Rand) C {
	var zero C
	var x interface{} = zero
	switch zero := x.(type) {
	case model2d.Coord:
		var x interface{} = sampleCoord2D
		return x.(func(r *rand.Rand) C)
	case model3d.Coord3D:
		var x interface{} = sampleCoord3D
		return x.(func(r *rand.Rand) C)
	default:
		panic(fmt.Sprintf("must specify RandDirection function for unknown type: %T", zero))
	}
}

func sampleCoord2D(r *rand.Rand) model2d.Coord {
	for {
		c := model2d.XY(r.NormFloat64(), r.NormFloat64())
		n := c.Norm()
		if n > 1e-5 {
			return c.Scale(1 / n)
		}
	}
}

func sampleCoord3D(r *rand.Rand) model3d.Coord3D {
	for {
		c := model3d.XYZ(r.NormFloat64(), r.NormFloat64(), r.NormFloat64())
		n := c.Norm()
		if n > 1e-5 {
			return c.Scale(1 / n)
		}
	}
}

// An Inequality makes up one constraint in a Polytope.
// It is the space of points c where c.Dot(Axis) < Max.
type Inequality[F constraints.Float, C Coord[F, C]] struct {
	Axis C
	Max  F
}

// A Polytope is a union of linear half-spaces.
type Polytope[F constraints.Float, C Coord[F, C]] []Inequality[F, C]

// Cast shoots a ray in the positive and negative direction, and returns the
// lowest magnitude scales for collisions with the bounds of the polytope in
// both directions, assuming that the origin is within the polytope.
//
// If there is no collision in a direction, it will have infinite magnitude.
func (p Polytope[F, C]) Cast(origin, direction C) (negT, posT F) {
	negT = F(math.Inf(-1))
	posT = F(math.Inf(1))

	for _, ineq := range p {
		dot := ineq.Axis.Dot(direction)
		if dot != 0 {
			offset := ineq.Max - ineq.Axis.Dot(origin)
			t := offset / dot
			if t < 0 && t > negT {
				negT = t
			} else if t > 0 && t < posT {
				posT = t
			}
		}
	}

	return
}
