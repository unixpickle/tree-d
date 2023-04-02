package treed

import (
	"golang.org/x/exp/constraints"
)

// BestReplacement returns a branch in the tree to replace with one of its
// children, greedily choosing the replacement that minimizes the increase in
// loss function.
//
// This can be applied repeatedly in combination with t.Replace() to decimate a
// tree while attempting to minimize the quality hit.
//
// If maxGos is 0, this will use GOMAXPROCS concurrent Goroutines at once.
// Otherwise, maxGos determines the maximum concurrency.
func BestReplacement[F constraints.Float, C Coord[F, C], T any](
	t *Tree[F, C, T],
	loss TAOLoss[T],
	inputs []C,
	targets []T,
	maxGos int,
) (replacement *Replacement[F, C, T], totalLoss float64) {
	return bestReplacement(
		t,
		loss,
		inputs,
		targets,
		newForkQueue[replacementBranch[F, C, T]](maxGos),
	)
}

func bestReplacement[F constraints.Float, C Coord[F, C], T any](
	t *Tree[F, C, T],
	loss TAOLoss[T],
	inputs []C,
	targets []T,
	queue *forkQueue[replacementBranch[F, C, T]],
) (replacement *Replacement[F, C, T], totalLoss float64) {
	if t.IsLeaf() {
		return nil, TotalTAOLoss(t, loss, inputs, targets)
	}

	mid := Partition(t.Axis, t.Threshold, inputs, targets)
	left, right := queue.Fork(
		func() replacementBranch[F, C, T] {
			res, lossVal := bestReplacement(t.LessThan, loss, inputs[:mid], targets[:mid], queue)
			other := TotalTAOLoss(t.LessThan, loss, inputs[mid:], targets[mid:])
			return replacementBranch[F, C, T]{res, lossVal, other}
		},
		func() replacementBranch[F, C, T] {
			res, lossVal := bestReplacement(t.GreaterEqual, loss, inputs[mid:], targets[mid:], queue)
			other := TotalTAOLoss(t.GreaterEqual, loss, inputs[:mid], targets[:mid])
			return replacementBranch[F, C, T]{res, lossVal, other}
		},
	)

	q := left.loss + right.loss
	leftNewLoss := left.loss + left.otherLoss
	rightNewLoss := right.loss + right.otherLoss
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
	for _, r := range []*Replacement[F, C, T]{left.res, right.res} {
		if r != nil && r.Delta() > res.Delta() {
			res = r
		}
	}
	return res, q
}

type replacementBranch[F constraints.Float, C Coord[F, C], T any] struct {
	res       *Replacement[F, C, T]
	loss      float64
	otherLoss float64
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
