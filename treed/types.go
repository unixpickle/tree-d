package treed

// A funcList is a general array type which can have an arbitrary getter.
// This can be useful for avoiding contiguous slice allocations.
type funcList[T any] struct {
	Len int
	Get func(int) T
}
