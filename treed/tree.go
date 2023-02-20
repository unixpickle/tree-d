package treed

import "golang.org/x/exp/constraints"

type Tree[F constraints.Float, C Coord[F, C], T any] struct {
	Axis         C
	Threshold    F
	LessThan     *Tree[F, C, T]
	GreaterEqual *Tree[F, C, T]

	Leaf T
}

func (t *Tree[F, C, T]) IsLeaf() bool {
	return t.LessThan == nil
}

func (t *Tree[F, C, T]) Apply(c C) T {
	if t.IsLeaf() {
		return t.Leaf
	} else {
		dot := t.Axis.Dot(c)
		if dot < t.Threshold {
			return t.LessThan.Apply(c)
		} else {
			return t.GreaterEqual.Apply(c)
		}
	}
}
