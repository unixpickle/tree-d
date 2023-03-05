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

func TreePolytopes(b *BoundedSolidTree) []model3d.ConvexPolytope {
	boundsPoly := model3d.NewConvexPolytopeRect(b.Min, b.Max)
	return computePolytopes(b.Tree, boundsPoly)
}

func computePolytopes(
	tree *SolidTree,
	polytope model3d.ConvexPolytope,
) []model3d.ConvexPolytope {
	if tree.IsLeaf() {
		if tree.Leaf == true {
			return []model3d.ConvexPolytope{append(model3d.ConvexPolytope{}, polytope...)}
		} else {
			return []model3d.ConvexPolytope{}
		}
	}
	subPoly := append(
		polytope,
		&model3d.LinearConstraint{Normal: tree.Axis, Max: tree.Threshold},
	)
	results := computePolytopes(tree.LessThan, subPoly)
	subPoly[len(subPoly)-1] = &model3d.LinearConstraint{Normal: tree.Axis.Scale(-1), Max: -tree.Threshold}
	return append(results, computePolytopes(tree.GreaterEqual, subPoly)...)
}
