package treed

import (
	"math"

	"golang.org/x/exp/constraints"
)

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
type EntropySplitLoss[F comparable] struct {
	// MinCount can be used to prevent splits which result in leaves with only
	// a small number of representative samples. In particular, splits with
	// less than MinCount samples on the left or right will not be returned
	// from MinimumSplit().
	MinCount int
}

func (e EntropySplitLoss[F]) Predict(items List[bool]) bool {
	return countTrue(items)*2 > items.Len
}

func (e EntropySplitLoss[F]) MinimumSplit(sorted List[bool], thresholds List[F]) SplitInfo {
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
		if (split.Loss < bestSplit.Loss && leftCount >= e.MinCount && rightCount >= e.MinCount) ||
			i == 0 {
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
	if numPoints == 0 {
		return 0
	}
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

// VarianceSplitLoss is a SplitLoss which computes the per-coordinate variance
// for each side of the split.
type VarianceSplitLoss[F constraints.Float, C Coord[F, C]] struct {
	// MinCount can be used to prevent splits which result in leaves with only
	// a small number of representative samples. In particular, splits with
	// less than MinCount samples on the left or right will not be returned
	// from MinimumSplit().
	MinCount int
}

func (v VarianceSplitLoss[F, C]) Predict(items List[C]) C {
	var sum C
	for i := 0; i < items.Len; i++ {
		sum = sum.Add(items.Get(i))
	}
	return sum.Scale(1.0 / F(items.Len))
}

func (v VarianceSplitLoss[F, C]) MinimumSplit(sorted List[C], thresholds List[F]) SplitInfo {
	if sorted.Len != thresholds.Len {
		panic("values and thresholds must have same length")
	}

	var leftSum, total rollingVariance[F, C]
	total.AddAll(sorted)

	lastIndex := 0
	var bestSplit SplitInfo
	iterateSplitPoints(thresholds, func(i int) {
		for lastIndex < i {
			leftSum.Add(sorted.Get(lastIndex))
			lastIndex++
		}
		var rightSum rollingVariance[F, C]
		rightSum.Diff(&total, &leftSum)
		leftCount := i
		rightCount := sorted.Len - i
		split := SplitInfo{
			Index: i,
			Loss:  float64(leftSum.TotalVariance()) + float64(rightSum.TotalVariance()),
		}
		if (split.Loss < bestSplit.Loss && leftCount >= v.MinCount && rightCount >= v.MinCount) ||
			i == 0 {
			bestSplit = split
		}
	})

	return bestSplit
}

type rollingVariance[F constraints.Float, C Coord[F, C]] struct {
	Sum   C
	SqSum C
	Count int
}

func (r *rollingVariance[F, C]) TotalVariance() F {
	mean := r.Sum.Scale(1 / F(r.Count))
	sqMean := r.SqSum.Scale(1 / F(r.Count))
	return F(r.Count) * sqMean.Sub(mean.Mul(mean)).Sum()
}

func (r *rollingVariance[F, C]) AddAll(cs List[C]) {
	for i := 0; i < cs.Len; i++ {
		c := cs.Get(i)
		r.Add(c)
	}
}

func (r *rollingVariance[F, C]) Add(c C) {
	r.Sum = r.Sum.Add(c)
	r.SqSum = r.SqSum.Add(c.Mul(c))
	r.Count++
}

func (r *rollingVariance[F, C]) Diff(a, b *rollingVariance[F, C]) {
	r.Sum = a.Sum.Sub(b.Sum)
	r.SqSum = a.SqSum.Sub(b.SqSum)
	r.Count = a.Count - b.Count
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
