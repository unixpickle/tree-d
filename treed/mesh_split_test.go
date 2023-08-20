package treed

import (
	"math"
	"testing"

	"github.com/unixpickle/model3d/model3d"
)

func TestSplitTriangleBasic(t *testing.T) {
	triangle := &model3d.Triangle{
		model3d.XYZ(-1, -1, 0.0),
		model3d.XYZ(1, -1, 0.1),
		model3d.XYZ(1, 1, -0.1),
	}
	axis := model3d.XYZ(0.01, 0.9, 0.01).Normalize()
	threshold := 0.1

	lt, gt := splitTriangle(triangle, axis, threshold)
	if len(lt) != 2 || len(gt) != 1 {
		t.Fatalf("expected two less-than and one greater-than but got %d %d", len(lt), len(gt))
	}

	totalArea := 0.0
	for _, tri := range append(append([]*model3d.Triangle{}, lt...), gt...) {
		if tri.Normal().Dot(triangle.Normal()) < 1-1e-5 {
			t.Fatalf("triangle normal should be %v but got %v", triangle.Normal(), tri.Normal())
		}
		totalArea += tri.Area()
	}
	if math.Abs(totalArea-triangle.Area()) > 1e-5 {
		t.Fatalf("total area should be %f but got %f", triangle.Area(), totalArea)
	}

	// Make sure permutations yield the same result.
	for i := 0; i < 2; i++ {
		triangle[0], triangle[1], triangle[2] = triangle[1], triangle[2], triangle[0]
		lt1, gt1 := splitTriangle(triangle, axis, threshold)
		for j, pair := range [2][2][]*model3d.Triangle{{lt, lt1}, {gt, gt1}} {
			if len(pair[0]) != len(pair[1]) {
				t.Fatalf("invalid pair: %d %d", i, j)
			}
			area1 := 0.0
			area2 := 0.0
			for _, tri := range pair[0] {
				area1 += tri.Area()
			}
			for k, tri := range pair[1] {
				area2 += tri.Area()
				if tri.Normal().Dot(triangle.Normal()) < 1-1e-5 {
					t.Fatalf("invalid normal: %d %d %d", i, j, k)
				}
			}
			if math.Abs(area1-area2) > 1e-5 {
				t.Fatalf("mismatched area: %d %d %f %f", i, j, area1, area2)
			}
		}
	}
}
