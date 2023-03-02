package treed

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/model3d/model3d"
)

func TestColliderSphereCollision(t *testing.T) {
	bounded := testTree()
	mesh := model3d.MarchingCubesSearch(TreeSolid(bounded), 0.01, 8)
	meshCollider := model3d.MeshToCollider(mesh)
	treeCollider := NewCollider(bounded)

	for i := 0; i < 5000; i++ {
		c := model3d.NewCoord3DRandNorm()
		r := rand.Float64() * 2

		mainResult := meshCollider.SphereCollision(c, r)
		lowerResult := meshCollider.SphereCollision(c, r-0.05)
		upperResult := meshCollider.SphereCollision(c, r+0.05)
		if mainResult == lowerResult && mainResult == upperResult {
			actualResult := treeCollider.SphereCollision(c, r)
			if actualResult != mainResult {
				t.Errorf("point %v radius %f should have collision=%v but got %v",
					c, r, mainResult, actualResult)
			}
		} else {
			i--
		}
	}
}

func testTree() *BoundedSolidTree {
	rand.Seed(0)

	xs := make([]model3d.Coord3D, 10000)
	ys := make([]bool, len(xs))
	for i := range xs {
		x := model3d.NewCoord3DRandBounds(model3d.XYZ(-1, -1, -1), model3d.XYZ(1, 1, 1))
		y := x.Dist(model3d.XYZ(0.5, -0.5, -0.5)) < 0.3 || x.Dist(model3d.XYZ(-1.0, 0.0, 0.5)) < 0.5
		xs[i] = x
		ys[i] = y
	}
	tree := GreedyTree[float64, model3d.Coord3D, bool](
		[]model3d.Coord3D{model3d.X(1), model3d.Y(1), model3d.Z(1)},
		xs[:5000],
		ys[:5000],
		EntropySplitLoss[float64]{},
		0,
		10,
	)

	// Remove effects of outliers. Without this, we may end up with
	// very thin polytopes that break tests.
	tree = tree.simplify(xs[5000:], ys[5000:], EqualityTAOLoss[bool]{})

	return &BoundedSolidTree{
		Min:  model3d.XYZ(-1, -1, -1),
		Max:  model3d.XYZ(1, 1, 1),
		Tree: tree,
	}
}
