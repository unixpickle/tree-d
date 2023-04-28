package treed

import (
	"math"
	"sync"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/model3d"
)

// A NormalMap is a function that maps 3D spatial coordinates to normal
// vectors.
type NormalMap interface {
	Predict(c model3d.Coord3D) model3d.Coord3D
}

type normalMapCollider struct {
	model3d.Collider
	mapping NormalMap
}

// MapNormals wraps a collider c and replaces collision normals using results
// from a NormalMap.
func MapNormals(c model3d.Collider, mapping NormalMap) model3d.Collider {
	return &normalMapCollider{
		Collider: c,
		mapping:  mapping,
	}
}

func (n *normalMapCollider) RayCollisions(r *model3d.Ray, f func(model3d.RayCollision)) (count int) {
	return n.Collider.RayCollisions(r, func(rc model3d.RayCollision) {
		if f != nil {
			rc.Normal = n.mapping.Predict(r.Origin.Add(r.Direction.Scale(rc.Scale)))
			f(rc)
		}
	})
}

func (n *normalMapCollider) FirstRayCollision(r *model3d.Ray) (collision model3d.RayCollision, collides bool) {
	collision, collides = n.Collider.FirstRayCollision(r)
	if collides {
		collision.Normal = n.mapping.Predict(r.Origin.Add(r.Direction.Scale(collision.Scale)))
	}
	return
}

// A Collider implements model3d.Collider for a wrapped boolean tree.
type Collider struct {
	min, max model3d.Coord3D
	tree     *SolidTree

	lock          sync.RWMutex
	truePolytopes []model3d.Collider
}

func NewCollider(b *BoundedSolidTree) *Collider {
	return &Collider{
		min:  b.Min,
		max:  b.Max,
		tree: b.AsTree(false, model3d.X(1), model3d.Y(1), model3d.Z(1)),
	}
}

func (c *Collider) Min() model3d.Coord3D {
	return c.min
}

func (c *Collider) Max() model3d.Coord3D {
	return c.max
}

func (c *Collider) RayCollisions(r *model3d.Ray, f func(model3d.RayCollision)) (count int) {
	return c.rayCollisions(r, false, f)
}

func (c *Collider) FirstRayCollision(r *model3d.Ray) (collision model3d.RayCollision, collides bool) {
	c.rayCollisions(r, true, func(rc model3d.RayCollision) {
		if !collides {
			collides = true
			collision = rc
		}
	})
	return
}

func (c *Collider) rayCollisions(r *model3d.Ray, firstOnly bool, f func(model3d.RayCollision)) (count int) {
	curPoint := r.Origin
	curT := 0.0
	prevValue := c.tree.Predict(curPoint)

	c.tree.RayChangePoints(curPoint, r.Direction, func(t float64, p, n model3d.Coord3D) bool {
		curT += t
		newValue := c.tree.Predict(p)
		if newValue == prevValue {
			return true
		}

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
		if firstOnly {
			return false
		}
		return true
	})

	return count
}

func (c *Collider) SphereCollision(center model3d.Coord3D, r float64) bool {
	c.lock.RLock()
	polytopes := c.truePolytopes
	c.lock.RUnlock()
	if polytopes == nil {
		c.lock.Lock()
		if c.truePolytopes == nil {
			polytopes := computePolytopes(c.tree, model3d.ConvexPolytope{})
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
			maxT := thisT * 2
			if maxT < 1e-4 {
				maxT = 1e-4
			}
			changeT := t.changeT(origin, direction, thisT, maxT)
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
