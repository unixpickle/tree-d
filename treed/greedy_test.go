package treed

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/model3d/model3d"
)

func TestGreedyOneLayer(t *testing.T) {
	points := []model3d.Coord3D{
		model3d.XYZ(0.60779, 0.273, 0.84464),
		model3d.XYZ(0.3206, 0.26422, 0.52094),
		model3d.XYZ(0.15469, 0.39314, 0.94174),
		model3d.XYZ(0.32412, 0.19605, 0.65235),
		model3d.XYZ(0.85606, 0.73316, 0.15401),
		model3d.XYZ(0.50925, 0.35245, 0.89935),
		model3d.XYZ(0.85105, 0.33801, 0.74244),
		model3d.XYZ(0.82971, 0.24029, 0.96707),
		model3d.XYZ(0.090814, 0.095153, 0.68592),
		model3d.XYZ(0.85454, 0.98837, 0.50694),
		model3d.XYZ(0.11416, 0.92107, 0.59644),
		model3d.XYZ(0.34615, 0.82264, 0.36529),
		model3d.XYZ(0.15533, 0.051568, 0.48636),
		model3d.XYZ(0.39776, 0.17687, 0.32365),
		model3d.XYZ(0.54357, 0.22281, 0.37119),
		model3d.XYZ(0.28501, 0.43394, 0.42455),
		model3d.XYZ(0.1176, 0.55334, 0.76542),
		model3d.XYZ(0.2719, 0.26032, 0.043687),
		model3d.XYZ(0.20398, 0.41625, 0.35441),
		model3d.XYZ(0.47036, 0.81723, 0.2178),
		model3d.XYZ(0.72063, 0.4537, 0.75298),
		model3d.XYZ(0.83556, 0.3956, 0.34812),
		model3d.XYZ(0.13877, 0.42288, 0.42567),
		model3d.XYZ(0.7462, 0.46695, 0.53442),
		model3d.XYZ(0.74893, 0.90623, 0.72674),
		model3d.XYZ(0.0087975, 0.11854, 0.44421),
		model3d.XYZ(0.94152, 0.64466, 0.14137),
		model3d.XYZ(0.88142, 0.646, 0.0063851),
		model3d.XYZ(0.21093, 0.37, 0.75318),
		model3d.XYZ(0.23007, 0.87637, 0.21424),
	}

	for i := 0; i < 30; i++ {
		// Permuting the input points should not matter.
		for j := 0; j+1 < len(points); j++ {
			idx := j + rand.Intn(len(points)-j)
			points[j], points[idx] = points[idx], points[j]
		}

		// Simple decision function that should always be
		// modeled perfectly by depth 2 tree.
		labels := make([]bool, len(points))
		for i, x := range points {
			if x.X < 0.5 {
				labels[i] = x.Y > 0.4
			} else {
				labels[i] = true
			}
		}

		// Search along just x and y should suffice, but we add a distractor.
		axes := []model3d.Coord3D{
			model3d.X(1),
			model3d.Y(1),
			model3d.Z(1),
		}
		tree := GreedyTree[float64, model3d.Coord3D, bool](
			axes,
			points,
			labels,
			EntropySplitLoss[float64]{},
			0,
			2,
		)

		for i, x := range points {
			label := labels[i]
			pred := tree.Apply(x)
			if label != pred {
				t.Errorf("point %v got %v but should be %v", x, pred, label)
			}
		}
	}
}
