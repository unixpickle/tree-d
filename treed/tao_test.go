package treed

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/model3d/model3d"
)

func TestSplitDecision(t *testing.T) {
	inputCoords := make([]model3d.Coord3D, 30)
	inputTargets := make([]bool, 30)
	inputMapping := map[model3d.Coord3D]bool{}
	for i := range inputCoords {
		inputCoords[i] = model3d.NewCoord3DRandNorm()
		inputTargets[i] = rand.Intn(2) == 0
		inputMapping[inputCoords[i]] = inputTargets[i]
	}

	coords := append([]model3d.Coord3D{}, inputCoords...)
	targets := append([]bool{}, inputTargets...)
	axis := model3d.XY(1.0, -1.0)
	threshold := 0.1
	idx := splitDecision(axis, threshold, coords, targets)

	for i, c := range coords {
		decision := axis.Dot(c) >= threshold
		if i < idx {
			if decision != false {
				t.Fatal("unexpected false decision")
			}
		} else {
			if decision != true {
				t.Fatal("unexpected true decision")
			}
		}
		actualT := targets[i]
		expectedT, ok := inputMapping[c]
		if !ok {
			t.Fatal("unexpected point duplication")
		}
		delete(inputMapping, c)
		if actualT != expectedT {
			t.Fatal("incorrect target")
		}
	}
}

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
		LR:          1e-2,
		WeightDecay: 1e-3,
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

func BenchmarkTAO(b *testing.B) {
	rand.Seed(1337)
	points := make([]model3d.Coord3D, 10000)
	for i := range points {
		points[i] = model3d.NewCoord3DRandUniform()
	}
	labels := make([]bool, len(points))
	for i, x := range points {
		if x.Dist(model3d.XYZ(0.3, 0.7, 0.5)) < 0.5 {
			labels[i] = true
		}
	}

	axes := model3d.NewMeshIcosphere(model3d.Origin, 1, 1).VertexSlice()
	tree := GreedyTree[float64, model3d.Coord3D, bool](
		axes,
		points,
		labels,
		EntropySplitLoss[float64]{},
		0,
		6,
	)

	tao := TAO[float64, model3d.Coord3D, bool]{
		Loss:        EqualityTAOLoss[bool]{},
		LR:          1e-2,
		WeightDecay: 1e-3,
		Momentum:    0.9,
		Iters:       1000,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tao.Optimize(tree, points, labels)
	}
}

func boolAccuracy(tree *SolidTree, coords []model3d.Coord3D, labels []bool) float64 {
	count := 0
	for i, c := range coords {
		label := labels[i]
		if tree.Predict(c) == label {
			count++
		}
	}
	return float64(count) / float64(len(coords))
}
