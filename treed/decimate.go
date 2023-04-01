package treed

import (
	"golang.org/x/exp/constraints"
)

func BestReplacement[F constraints.Float, C Coord[F, C], T any](
	t *Tree[F, C, T],
	loss TAOLoss[T],
	inputs []C,
	targets []T,
) (replacement *Replacement[F, C, T], totalLoss float64) {
	if t.IsLeaf() {
		return nil, TotalTAOLoss(t, loss, inputs, targets)
	}

	mid := Partition(t.Axis, t.Threshold, inputs, targets)
	leftRes, leftLoss := BestReplacement(t.LessThan, loss, inputs[:mid], targets[:mid])
	leftOther := TotalTAOLoss(t.LessThan, loss, inputs[mid:], targets[mid:])
	rightRes, rightLoss := BestReplacement(t.GreaterEqual, loss, inputs[mid:], targets[mid:])
	rightOther := TotalTAOLoss(t.GreaterEqual, loss, inputs[:mid], targets[:mid])

	q := leftLoss + rightLoss
	leftNewLoss := leftLoss + leftOther
	rightNewLoss := rightLoss + rightOther
	var res *Replacement[F, C, T]
	if leftNewLoss < rightNewLoss {
		res = &Replacement[F, C, T]{
			OldLoss: q,
			NewLoss: leftNewLoss,
			Replace: t,
			With:    t.LessThan,
		}
	} else {
		res = &Replacement[F, C, T]{
			OldLoss: q,
			NewLoss: rightNewLoss,
			Replace: t,
			With:    t.GreaterEqual,
		}
	}
	for _, r := range []*Replacement[F, C, T]{leftRes, rightRes} {
		if r != nil && r.Delta() > res.Delta() {
			res = r
		}
	}
	return res, q
}

type Replacement[F constraints.Float, C Coord[F, C], T any] struct {
	OldLoss float64
	NewLoss float64

	Replace *Tree[F, C, T]
	With    *Tree[F, C, T]
}

func (r *Replacement[F, C, T]) Delta() float64 {
	return r.OldLoss - r.NewLoss
}
