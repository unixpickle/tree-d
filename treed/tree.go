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

func indentText(text string) string {
	lines := strings.Split(text, "\n")
	for i, x := range lines {
		lines[i] = "  " + x
	}
	return strings.Join(lines, "\n")
}
