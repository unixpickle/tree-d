package treed

import (
	"math"
	"sync"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/model3d"
)

// A Collider implements model3d.Collider for a wrapped boolean tree.
type Collider struct {
	tree *BoundedSolidTree

	lock          sync.RWMutex
	truePolytopes []model3d.Collider
}

func (c *Collider) Min() model3d.Coord3D {
	return c.tree.Min
}

func (c *Collider) Max() model3d.Coord3D {
	return c.tree.Max
}

func (c *Collider) SphereCollision(center model3d.Coord3D, r float64) bool {
	c.lock.RLock()
	polytopes := c.truePolytopes
	c.lock.RUnlock()
	if polytopes == nil {
		c.lock.Lock()
		if c.truePolytopes == nil {
			boundsPoly := model3d.NewConvexPolytopeRect(c.Min(), c.Max())
			polytopes := c.computePolytopes(c.tree.Tree, boundsPoly)
			c.truePolytopes = make([]model3d.Collider, len(polytopes))
			essentials.ConcurrentMap(0, len(polytopes), func(i int) {
				c.truePolytopes[i] = model3d.MeshToCollider(polytopes[i].Mesh())
			})
		}
		polytopes = c.truePolytopes
		c.lock.Unlock()
	}
	for _, polytope := range polytopes {
		if polytope.SphereCollision(center, r) {
			return true
		}
	}
	return false
}

func (c *Collider) computePolytopes(
	tree *SolidTree,
	polytope model3d.ConvexPolytope,
) []model3d.ConvexPolytope {
	if tree.IsLeaf() {
		if tree.Leaf == true {
			return []model3d.ConvexPolytope{append(model3d.ConvexPolytope{}, polytope...)}
		} else {
			return []model3d.ConvexPolytope{}
		}
	}
	subPoly := append(
		polytope,
		&model3d.LinearConstraint{Normal: tree.Axis, Max: tree.Threshold},
	)
	results := c.computePolytopes(tree.LessThan, subPoly)
	subPoly[len(subPoly)-1] = &model3d.LinearConstraint{Normal: tree.Axis.Scale(-1), Max: -tree.Threshold}
	return append(results, c.computePolytopes(tree.GreaterEqual, subPoly)...)
}

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
