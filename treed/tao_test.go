package treed

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/model3d/model3d"
)

func TestTAO(t *testing.T) {
	rand.Seed(1337)
	points := make([]model3d.Coord3D, 5000)
	for i := range points {
		points[i] = model3d.NewCoord3DRandUniform()
	}
	labels := make([]bool, len(points))
	for i, x := range points {
		if x.Dist(model3d.XYZ(0.3, 0.7, 0.5)) < 0.5 {
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
		4,
	)

	acc := boolAccuracy(tree, points, labels)

	tao := TAO[float64, model3d.Coord3D, bool]{
		Loss:        EqualityTAOLoss[bool]{},
		LR:          1e-1,
		WeightDecay: 1e-4,
		Momentum:    0.9,
		Iters:       1000,
	}
	result := tao.Optimize(tree, points, labels)
	if result.OldLoss <= result.NewLoss {
		t.Errorf("expected loss %f > %f", result.OldLoss, result.NewLoss)
	}

	newAcc := boolAccuracy(result.Tree, points, labels)
	if acc >= newAcc {
		t.Errorf("expected accuracy %f > %f", newAcc, acc)
	}
}

func boolAccuracy(tree *Tree[float64, model3d.Coord3D, bool], coords []model3d.Coord3D, labels []bool) float64 {
	count := 0
	for i, c := range coords {
		label := labels[i]
		if tree.Apply(c) == label {
			count++
		}
	}
	return float64(count) / float64(len(coords))
}
