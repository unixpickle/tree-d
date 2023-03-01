package treed

import (
	"github.com/unixpickle/model3d/model3d"
	"golang.org/x/exp/constraints"
)

type BoundedSolidTree = BoundedTree[float64, model3d.Coord3D, bool]

type BoundedTree[F constraints.Float, C Coord[F, C], T any] struct {
	Min  C
	Max  C
	Tree *Tree[F, C, T]
}

func TreeSolid(b *BoundedSolidTree) model3d.Solid {
	return model3d.CheckedFuncSolid(
		b.Min,
		b.Max,
		b.Tree.Predict,
	)
}
