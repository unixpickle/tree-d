package treed

import (
	"github.com/unixpickle/model3d/model3d"
	"golang.org/x/exp/constraints"
)

type BoundedTree[F constraints.Float, C Coord[F, C], T any] struct {
	Min  C
	Max  C
	Tree *Tree[F, C, T]
}

type BoundedSolidTree = BoundedTree[float64, model3d.Coord3D, bool]
