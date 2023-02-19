package treed

type SplitInfo[T any] struct {
	// Number of elements in less than branch to keep.
	Index int

	// Losses of both branches.
	Losses [2]float64

	// Locally optimal predictions in each branch.
	Preds [2]T
}

type Loss[F comparable, T any] interface {
	MinimumSplit(sorted []T, values []F) *SplitInfo[T]
}
