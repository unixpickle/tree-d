package treed

import "math"

// SplitInfo is returned by Loss.MinimumSplit() to indicate where to split a
// list of labels for sorted samples to minimize a loss.
type SplitInfo struct {
	// Number of elements in less than branch to keep.
	Index int

	// Total loss of both branches.
	Loss float64
}

// A SplitLoss implements a decision criterion used to select the best split of
// a tree. The loss must understand the label type T, and uses a threshold type
// F only to determine when two data points have exactly the same split
// threshold and therefore must always be grouped together.
type SplitLoss[F comparable, T any] interface {
	// Predict returns the value to minimize the loss of a leaf.
	Predict(List[T]) T

	// Get the best split of the data according to the loss function.
	//
	// The first argument must be a sorted list of labels, according to the
	// thresholds passed to the second argument.
	//
	// The result may be a split where all of the values are on one side or the
	// other, in which case no split reduces the loss.
	MinimumSplit(sorted List[T], thresholds List[F]) SplitInfo
}

// EntropySplitLoss is a SplitLoss which computes the total entropy across both
// branches.
type EntropySplitLoss[F comparable] struct{}

func (_ EntropySplitLoss[F]) Predict(items List[bool]) bool {
	return countTrue(items)*2 > items.Len
}

func (_ EntropySplitLoss[F]) MinimumSplit(sorted List[bool], thresholds List[F]) SplitInfo {
	if sorted.Len != thresholds.Len {
		panic("values and thresholds must have same length")
	}

	leftSum := 0
	rightSum := countTrue(sorted)

	lastIndex := 0
	var bestSplit SplitInfo
	iterateSplitPoints(thresholds, func(i int) {
		for lastIndex < i {
			if sorted.Get(lastIndex) {
				leftSum++
				rightSum--
			}
			lastIndex++
		}
		leftCount := i
		rightCount := sorted.Len - i
		split := SplitInfo{
			Index: i,
			Loss:  entropy(leftCount, leftSum) + entropy(rightCount, rightSum),
		}
		if i == 0 || split.Loss < bestSplit.Loss {
			bestSplit = split
		}
	})

	return bestSplit
}

func countTrue(list List[bool]) int {
	var count int
	for i := 0; i < list.Len; i++ {
		if list.Get(i) {
			count++
		}
	}
	return count
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

func iterateSplitPoints[F comparable](thresholds List[F], f func(int)) {
	var prevValue F
	for i := 0; i < thresholds.Len; i++ {
		x := thresholds.Get(i)
		if i == 0 || x != prevValue {
			f(i)
		}
		prevValue = x
	}
	f(thresholds.Len)
}
