package treed

import "golang.org/x/exp/constraints"

type BoundedTree[F constraints.Float, C Coord[F, C], T any] struct {
	Min  C
	Max  C
	Tree *Tree[F, C, T]
}
