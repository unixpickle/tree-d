package treed

import "golang.org/x/exp/constraints"

// A VecSumEnsemble sums the vector outputs from multiple trees.
type VecSumEnsemble[F constraints.Float, C Coord[F, C], T Coord[F, T]] []*Tree[F, C, T]

func (t VecSumEnsemble[F, C, T]) Predict(x C) T {
	res := t[0].Predict(x)
	for _, t1 := range t[1:] {
		res = res.Add(t1.Predict(x))
	}
	return res
}

// A VecSumNormEnsemble is like VecSumEnsemble, but normalizes the outputs to
// have unit norm.
type VecSumNormEnsemble[F constraints.Float, C Coord[F, C], T Coord[F, T]] []*Tree[F, C, T]

func (t VecSumNormEnsemble[F, C, T]) Predict(x C) T {
	res := VecSumEnsemble[F, C, T](t).Predict(x)
	norm := res.Norm()
	if norm != 0 {
		res = res.Scale(1 / norm)
	}
	return res
}
