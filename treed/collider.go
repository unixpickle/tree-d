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

func NewCollider(b *BoundedSolidTree) *Collider {
	return &Collider{tree: b}
}

func (c *Collider) Min() model3d.Coord3D {
	return c.tree.Min
}

func (c *Collider) Max() model3d.Coord3D {
	return c.tree.Max
}

func (c *Collider) RayCollisions(r *model3d.Ray, f func(model3d.RayCollision)) (count int) {
	bounds := model3d.BoundsRect(c)
	var i int
	var rcs [2]model3d.RayCollision
	bounds.RayCollisions(r, func(rc model3d.RayCollision) {
		rcs[i] = rc
		i++
	})

	if i == 0 {
		return 0
	}

	var entry, exit model3d.RayCollision

	curPoint := r.Origin
	curT := 0.0
	if i == 1 {
		exit = rcs[0]
	} else {
		entry, exit = rcs[0], rcs[1]
		curPoint = r.Origin.Add(r.Direction.Scale(entry.Scale))
		curT = entry.Scale
	}
	prevValue := c.tree.Tree.Predict(curPoint)

	if i == 2 {
		if prevValue {
			count++
			if f != nil {
				f(entry)
			}
		}
	}

	terminated := false
	c.tree.Tree.RayChangePoints(curPoint, r.Direction, func(t float64, p, n model3d.Coord3D) bool {
		curT += t
		if curT >= exit.Scale {
			// Early termination due to exiting the bounds.
			terminated = true
			if prevValue {
				count++
				if f != nil {
					f(exit)
				}
			}
			return false
		}

		newValue := c.tree.Tree.Predict(p)
		if newValue != prevValue {
			prevValue = newValue
			count++
			if f != nil {
				if !newValue {
					n = n.Scale(-1)
				}
				f(model3d.RayCollision{
					Scale:  curT,
					Normal: n,
				})
			}
		}
		return true
	})

	if !terminated && prevValue {
		count++
		if f != nil {
			f(exit)
		}
	}

	return count
}

func (c *Collider) FirstRayCollision(r *model3d.Ray) (collision model3d.RayCollision, collides bool) {
	c.RayCollisions(r, func(rc model3d.RayCollision) {
		if !collides {
			collides = true
			collision = rc
		}
	})
	return
}

func (c *Collider) SphereCollision(center model3d.Coord3D, r float64) bool {
	c.lock.RLock()
	polytopes := c.truePolytopes
	c.lock.RUnlock()
	if polytopes == nil {
		c.lock.Lock()
		if c.truePolytopes == nil {
			polytopes := TreePolytopes(c.tree)
			c.truePolytopes = make([]model3d.Collider, len(polytopes))
			essentials.ConcurrentMap(0, len(polytopes), func(i int) {
				m := polytopes[i].Mesh()
				c.truePolytopes[i] = model3d.MeshToCollider(m)
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
// For each ray collision, the point of the change and associated scale for the
// direction is passed to f, as well as the normal axis of the surface.
//
// If f returns false, iteration is terminated early.
func (t *Tree[F, C, T]) RayChangePoints(origin, direction C, f func(F, C, C) bool) {
	for {
		point, normal, changeT := t.nextBranchChange(origin, direction)
		if math.IsInf(float64(changeT), 0) {
			return
		}
		if !f(changeT, point, normal.Scale(1/normal.Norm())) {
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
	dirDot := t.Axis.Dot(direction)

	absDirDot := dirDot
	if absDirDot < 0 {
		absDirDot = -absDirDot
	}
	if absDirDot < t.Axis.Norm()*direction.Norm()*1e-8 {
		var zero C
		return zero, zero, F(math.Inf(1))
	}

	curDot := t.Axis.Dot(origin)
	child := t.LessThan
	if curDot >= t.Threshold {
		child = t.GreaterEqual
	}

	normal = t.Axis
	if child == t.LessThan {
		normal = normal.Scale(-1)
	}

	thisT := (t.Threshold - curDot) / dirDot

	// This edge case might seem extremely unusual, but it actually occurs
	// naturally for trees with tight bounding boxes.
	if t.Threshold == curDot {
		maxT := F(1e8)
		maxDot := t.Axis.Dot(origin.Add(direction.Scale(maxT)))
		if (curDot >= t.Threshold) != (maxDot >= t.Threshold) {
			changeT := t.changeT(origin, direction, thisT, maxT)
			return origin.Add(direction.Scale(changeT)), normal, changeT
		}
	}

	if thisT <= 0 {
		return child.nextBranchChange(origin, direction)
	} else {
		childPoint, childNormal, childT := child.nextBranchChange(origin, direction)
		if thisT > childT {
			return childPoint, childNormal, childT
		} else {
			changeT := t.changeT(origin, direction, thisT, thisT*2)
			return origin.Add(direction.Scale(changeT)), normal, changeT
		}
	}
}

func (t *Tree[F, C, T]) changeT(origin, direction C, minT, maxT F) F {
	orig := t.Axis.Dot(origin) < t.Threshold
	x := origin.Add(direction.Scale(minT))
	if t.Axis.Dot(x) < t.Threshold != orig {
		return minT
	}
	if t.Axis.Dot(origin.Add(direction.Scale(maxT))) < t.Threshold == orig {
		panic("impossible situation encountered: collision was expected")
	}
	for i := 0; i < 32; i++ {
		midT := (minT + maxT) / 2
		if t.Axis.Dot(origin.Add(direction.Scale(midT))) < t.Threshold != orig {
			maxT = midT
		} else {
			minT = midT
		}
	}
	return maxT
}
