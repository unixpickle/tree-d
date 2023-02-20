package treed

// A List is a general array type which can have an arbitrary getter.
// This can be useful for avoiding contiguous slice allocations.
type List[T any] struct {
	Len int
	Get func(int) T
}
