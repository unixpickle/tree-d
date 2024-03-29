package treed

import "golang.org/x/exp/constraints"

// A TAOLoss is a loss function which can be computed between a label and a
// leaf prediction, both of the same type.
type TAOLoss[T any] interface {
	// Predict produces the optimal leaf node output for a list of labels.
	Predict(List[T]) T

	// Loss returns a scalar, non-negative loss for the pair.
	Loss(label, prediction T) float64
}

// TotalTAOLoss predicts outputs for each input, and returns the sum of the
// losses across all examples.
func TotalTAOLoss[F constraints.Float, C Coord[F, C], T any](
	t *Tree[F, C, T],
	loss TAOLoss[T],
	inputs []C,
	targets []T,
) float64 {
	var total float64
	for i, target := range targets {
		total += loss.Loss(target, t.Predict(inputs[i]))
	}
	return total
}

// EqualityTAOLoss is always 1 when the label does not equal the target, and 0
// otherwise. The mode of the inputs determines the leaf predictions.
//
// This can be used for trees with discrete predictions, such as trees with
// boolean predictions.
type EqualityTAOLoss[T comparable] struct{}

func (_ EqualityTAOLoss[T]) Predict(items List[T]) T {
	// Ensure deterministic order for fixed `items` by keeping ordered list of
	// values instead of relying on map ordering. This only matters when
	// breaking ties between two modes.
	var values []T

	counts := map[T]int{}
	for i := 0; i < items.Len; i++ {
		x := items.Get(i)
		if old, ok := counts[x]; ok {
			counts[x] = old + 1
		} else {
			counts[x] = 1
			values = append(values, x)
		}
		counts[items.Get(i)]++
	}
	maxCount := 0
	for _, c := range counts {
		if c > maxCount {
			maxCount = c
		}
	}
	for _, x := range values {
		if counts[x] == maxCount {
			return x
		}
	}

	// In the case where the list has length zero.
	var zero T
	return zero
}

func (_ EqualityTAOLoss[T]) Loss(label, prediction T) float64 {
	if label == prediction {
		return 0
	} else {
		return 1
	}
}

// SquaredErrorTAOLoss computes the squared error between the prediction and
// the target, summed across dimensions.
type SquaredErrorTAOLoss[F constraints.Float, C Coord[F, C]] struct{}

func (_ SquaredErrorTAOLoss[F, C]) Predict(items List[C]) C {
	var sum C
	for i := 0; i < items.Len; i++ {
		sum = sum.Add(items.Get(i))
	}
	return sum.Scale(1 / F(items.Len))
}

func (_ SquaredErrorTAOLoss[F, C]) Loss(label, prediction C) float64 {
	diff := label.Sub(prediction)
	return float64(diff.Dot(diff))
}
