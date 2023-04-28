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

func (b *BoundedTree[F, C, T]) Scale(s F) *BoundedTree[F, C, T] {
	return &BoundedTree[F, C, T]{
		Min:  b.Min.Scale(s),
		Max:  b.Max.Scale(s),
		Tree: b.Tree.Scale(s),
	}
}

func (b *BoundedTree[F, C, T]) Translate(c C) *BoundedTree[F, C, T] {
	return &BoundedTree[F, C, T]{
		Min:  b.Min.Add(c),
		Max:  b.Max.Add(c),
		Tree: b.Tree.Translate(c),
	}
}

// AsTree embeds the bounds into the tree and returns the resulting tree.
//
// The given zero argument will be returned from leaf nodes outside the bounds.
//
// Requires that you pass every axis for the coordinate space.
// For example, in 3D, this would require (1, 0, 0), (0, 1, 0), (0, 0, 1).
func (b *BoundedTree[F, C, T]) AsTree(zero T, axes ...C) *Tree[F, C, T] {
	if len(axes) == 0 {
		return b.Tree
	}
	return &Tree[F, C, T]{
		Axis:      axes[0],
		Threshold: axes[0].Dot(b.Min),
		LessThan:  &Tree[F, C, T]{Leaf: zero},
		GreaterEqual: &Tree[F, C, T]{
			Axis:         axes[0],
			Threshold:    axes[0].Dot(b.Max),
			LessThan:     b.AsTree(zero, axes[1:]...),
			GreaterEqual: &Tree[F, C, T]{Leaf: zero},
		},
	}
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
