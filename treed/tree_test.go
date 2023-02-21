package treed

import (
	"testing"

	"github.com/unixpickle/model3d/model3d"
)

func TestTreeCoord3D(t *testing.T) {
	tree := &Tree[float64, model3d.Coord3D, string]{
		Axis:      model3d.Z(1),
		Threshold: 0.5,
		LessThan: &Tree[float64, model3d.Coord3D, string]{
			Leaf: "left",
		},
		GreaterEqual: &Tree[float64, model3d.Coord3D, string]{
			Leaf: "right",
		},
	}
	mustEqual(t, "left", tree.Predict(model3d.Z(0)))
	mustEqual(t, "left", tree.Predict(model3d.Z(0.499999)))
	mustEqual(t, "right", tree.Predict(model3d.Z(0.5)))
	mustEqual(t, "right", tree.Predict(model3d.Z(0.50001)))
}

func mustEqual(t *testing.T, x, y string) {
	if x != y {
		t.Fatalf("expected %s but got %s", x, y)
	}
}
