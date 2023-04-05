package treed

import (
	"fmt"
	"strings"

	"github.com/unixpickle/model3d/model3d"
	"golang.org/x/exp/constraints"
)

type SolidTree = Tree[float64, model3d.Coord3D, bool]
type CoordTree = Tree[float64, model3d.Coord3D, model3d.Coord3D]

type Tree[F constraints.Float, C Coord[F, C], T any] struct {
	Axis         C
	Threshold    F
	LessThan     *Tree[F, C, T]
	GreaterEqual *Tree[F, C, T]

	Leaf T
}

// MapLeaves applies a function f to all the leaves of the tree.
func MapLeaves[F constraints.Float, C Coord[F, C], T1, T2 any](t *Tree[F, C, T1], f func(T1) T2) *Tree[F, C, T2] {
	if t.IsLeaf() {
		return &Tree[F, C, T2]{
			Leaf: f(t.Leaf),
		}
	} else {
		return &Tree[F, C, T2]{
			Axis:         t.Axis,
			Threshold:    t.Threshold,
			LessThan:     MapLeaves(t.LessThan, f),
			GreaterEqual: MapLeaves(t.GreaterEqual, f),
		}
	}
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

func (t *Tree[F, C, T]) NumLeaves() int {
	if t.IsLeaf() {
		return 1
	}
	return t.LessThan.NumLeaves() + t.GreaterEqual.NumLeaves()
}

func (t *Tree[F, C, T]) Scale(s F) *Tree[F, C, T] {
	if t.IsLeaf() {
		return t
	}
	return &Tree[F, C, T]{
		Axis:         t.Axis,
		Threshold:    t.Threshold * s,
		LessThan:     t.LessThan.Scale(s),
		GreaterEqual: t.GreaterEqual.Scale(s),
	}
}

func (t *Tree[F, C, T]) Translate(c C) *Tree[F, C, T] {
	if t.IsLeaf() {
		return t
	}
	return &Tree[F, C, T]{
		Axis:         t.Axis,
		Threshold:    t.Threshold + t.Axis.Dot(c),
		LessThan:     t.LessThan.Translate(c),
		GreaterEqual: t.GreaterEqual.Translate(c),
	}
}

// Replace swaps a node in the tree for a different node, and returns true in
// the second argument if the old node was successfully found.
func (t *Tree[F, C, T]) Replace(old, new *Tree[F, C, T]) (*Tree[F, C, T], bool) {
	if t == old {
		return new, true
	} else if t.IsLeaf() {
		return t, false
	} else {
		newLeft, ok1 := t.LessThan.Replace(old, new)
		newRight, ok2 := t.GreaterEqual.Replace(old, new)
		if ok1 || ok2 {
			return &Tree[F, C, T]{
				Axis:         t.Axis,
				Threshold:    t.Threshold,
				LessThan:     newLeft,
				GreaterEqual: newRight,
			}, true
		}
		return t, false
	}
}

// Simplify prunes the tree when it does not increase the loss.
func (t *Tree[F, C, T]) Simplify(
	coords []C,
	labels []T,
	loss TAOLoss[T],
) *Tree[F, C, T] {
	coords = append([]C{}, coords...)
	labels = append([]T{}, labels...)
	return t.simplify(coords, labels, loss)
}

func (t *Tree[F, C, T]) simplify(
	coords []C,
	labels []T,
	loss TAOLoss[T],
) *Tree[F, C, T] {
	if t.IsLeaf() {
		return t
	}

	idx := Partition(t.Axis, t.Threshold, coords, labels)
	if idx == 0 {
		return t.GreaterEqual
	} else if idx == len(coords) {
		return t.LessThan
	}
	left := t.LessThan.simplify(coords[:idx], labels[:idx], loss)
	right := t.GreaterEqual.simplify(coords[idx:], labels[idx:], loss)

	var leftBetter int
	var rightBetter int
	for i, c := range coords {
		target := labels[i]
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
