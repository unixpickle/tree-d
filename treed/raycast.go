package treed

import "math"

// RayChangePoints iterates the points where the ray causes a change in the
// tree decision path.
//
// For each ray collision, the point of the change is passed to f, as well as
// the normal axis which caused the change.
//
// If f returns false, iteration is terminated early.
func (t *Tree[F, C, T]) RayChangePoints(origin, direction C, f func(C, C) bool) {
	for {
		point, normal, changeT := t.nextBranchChange(origin, direction)
		if math.IsInf(float64(changeT), 0) {
			return
		}
		if !f(point, normal) {
			return
		}
		origin = point
	}
}

func (t *Tree[F, C, T]) nextBranchChange(origin, direction C) (point, normal C, changeT F) {
	if t.IsLeaf() {
		var zero C
		return zero, zero, F(math.Inf(1))
	}
	curDot := t.Axis.Dot(origin)
	dirDot := t.Axis.Dot(direction)

	child := t.LessThan
	if curDot >= t.Threshold {
		child = t.GreaterEqual
	}

	thisT := (t.Threshold - curDot) / dirDot
	if thisT <= 0 {
		return child.nextBranchChange(origin, direction)
	} else {
		childPoint, childNormal, childT := child.nextBranchChange(origin, direction)
		if thisT > childT {
			return childPoint, childNormal, childT
		} else {
			return t.pointOfChange(origin, direction, thisT), t.Axis, thisT
		}
	}
}

func (t *Tree[F, C, T]) pointOfChange(origin, direction C, approxT F) C {
	orig := t.Axis.Dot(origin) < t.Threshold
	for {
		x := origin.Add(direction.Scale(approxT))
		if t.Axis.Dot(x) < t.Threshold != orig {
			return x
		}
		approxT *= 1.0000001
	}
}
