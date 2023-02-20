package treed

import "golang.org/x/exp/constraints"

type Coord[F constraints.Float, Self any] interface {
	Dot(Self) F
}

// A List is a general array type which can have an arbitrary getter.
// This can be useful for avoiding contiguous slice allocations.
type List[T any] struct {
	Len int
	Get func(int) T
}

func NewListSlice[T any](s []T) List[T] {
	return List[T]{
		Len: len(s),
		Get: func(i int) T {
			return s[i]
		},
	}
}
