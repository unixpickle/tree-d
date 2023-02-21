package treed

import (
	"fmt"
	"strings"

	"golang.org/x/exp/constraints"
)

type Tree[F constraints.Float, C Coord[F, C], T any] struct {
	Axis         C
	Threshold    F
	LessThan     *Tree[F, C, T]
	GreaterEqual *Tree[F, C, T]

	Leaf T
}

func (t *Tree[F, C, T]) IsLeaf() bool {
	return t.LessThan == nil
}

func (t *Tree[F, C, T]) Predict(c C) T {
	if t.IsLeaf() {
		return t.Leaf
	} else {
		dot := t.Axis.Dot(c)
		if dot < t.Threshold {
			return t.LessThan.Predict(c)
		} else {
			return t.GreaterEqual.Predict(c)
		}
	}
}

func (t *Tree[F, C, T]) String() string {
	if t.IsLeaf() {
		return fmt.Sprintf("return %v", t.Leaf)
	} else {
		return fmt.Sprintf(
			"if point * %v < %v {\n%s\n} else {\n%s\n}",
			t.Axis,
			t.Threshold,
			indentText(t.LessThan.String()),
			indentText(t.GreaterEqual.String()),
		)
	}
}

// SimplifyTree prunes the tree when it does not increase the loss.
func (t *Tree[F, C, T]) Simplify(
	coords []C,
	targets []T,
	loss TAOLoss[T],
) *Tree[F, C, T] {
	coords = append([]C{}, coords...)
	targets = append([]T{}, targets...)
	return t.simplify(coords, targets, loss)
}

func (t *Tree[F, C, T]) simplify(
	coords []C,
	targets []T,
	loss TAOLoss[T],
) *Tree[F, C, T] {
	if t.IsLeaf() {
		return t
	}

	idx := splitDecision(t.Axis, t.Threshold, coords, targets)
	if idx == 0 {
		return t.GreaterEqual
	} else if idx == len(coords) {
		return t.LessThan
	}
	left := t.LessThan.simplify(coords[:idx], targets[:idx], loss)
	right := t.GreaterEqual.simplify(coords[idx:], targets[idx:], loss)

	var leftBetter int
	var rightBetter int
	for i, c := range coords {
		target := targets[i]
		leftLoss := F(loss.Loss(target, left.Predict(c)))
		rightLoss := F(loss.Loss(target, right.Predict(c)))
		if leftLoss < rightLoss {
			leftBetter++
		} else if rightLoss < leftLoss {
			rightBetter++
		}
	}
	if leftBetter == 0 {
		return right
	} else if rightBetter == 0 {
		return left
	}
	return &Tree[F, C, T]{
		Axis:         t.Axis,
		Threshold:    t.Threshold,
		LessThan:     left,
		GreaterEqual: right,
	}
}

func indentText(text string) string {
	lines := strings.Split(text, "\n")
	for i, x := range lines {
		lines[i] = "  " + x
	}
	return strings.Join(lines, "\n")
}
