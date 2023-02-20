package treed

import "math"

type SplitInfo[T any] struct {
	// Number of elements in less than branch to keep.
	Index int

	// Total loss of both branches.
	Loss float64

	// Locally optimal predictions in each branch.
	Preds [2]T
}

type Loss[F comparable, T any] interface {
	// Get the best split of the data according to the loss function.
	//
	// May be a split where all of the values are on one side, in which case no
	// split reduces the loss.
	MinimumSplit(sorted []T, values []F) *SplitInfo[T]
}

type EntropyLoss[F comparable] struct{}

func (_ EntropyLoss[F]) MinimumSplit(sorted []bool, values []F) *SplitInfo[bool] {
	var leftSum int
	var rightSum int
	for _, x := range sorted {
		if x {
			rightSum += 1
		}
	}

	lastIndex := 0
	var bestSplit *SplitInfo[bool]
	iterateSplitPoints(values, func(i int) {
		for lastIndex < i {
			if sorted[lastIndex] {
				leftSum++
				rightSum--
			}
			lastIndex++
		}
		leftCount := i
		rightCount := len(values) - i
		split := SplitInfo[bool]{
			Index: i,
			Loss:  entropy(leftCount, leftSum) + entropy(rightCount, rightSum),
			Preds: [2]bool{
				leftSum*2 >= leftCount,
				rightSum*2 >= rightCount,
			},
		}
		if bestSplit == nil {
			bestSplit = new(SplitInfo[bool])
			*bestSplit = split
		} else if split.Loss < bestSplit.Loss {
			*bestSplit = split
		}
	})

	return bestSplit
}

func entropy(numPoints, numTrue int) float64 {
	numFalse := numPoints - numTrue
	fracTrue := float64(numTrue) / float64(numPoints)
	fracFalse := float64(numFalse) / float64(numPoints)
	return -(float64(numTrue)*logOrZero(fracTrue) +
		float64(numFalse)*logOrZero(fracFalse))
}

func logOrZero(x float64) float64 {
	if x == 0 {
		return 0
	}
	return math.Log(x)
}

func iterateSplitPoints[F comparable](values []F, f func(int)) {
	var prevValue F
	for i, x := range values {
		if i == 0 || x != prevValue {
			f(i)
		}
		prevValue = x
	}
	f(len(values))
}
