package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/model3d"
	"github.com/unixpickle/tree-d/treed"
)

func main() {
	var maxLeaves int
	var numSamples int
	flag.IntVar(&maxLeaves, "max-leaves", 512, "maximum number of leaves")
	flag.IntVar(&numSamples, "num-samples", 2000000, "number of point samples to use")
	flag.Parse()

	args := flag.Args()
	if len(args) != 3 {
		fmt.Fprintln(os.Stderr, "Usage: simplify_tree [flags] <input.stl> <input.bin> <output.bin>")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
		os.Exit(1)
	}
	meshPath, inputPath, outputPath := args[0], args[1], args[2]

	log.Println("Loading tree...")
	f, err := os.Open(inputPath)
	essentials.Must(err)
	tree, err := treed.ReadBoundedSolidTree(f)
	f.Close()
	essentials.Must(err)

	log.Println("Loading mesh...")
	f, err = os.Open(meshPath)
	essentials.Must(err)
	tris, err := model3d.ReadSTL(f)
	f.Close()
	essentials.Must(err)
	mesh := model3d.NewMeshTriangles(tris)
	meshSolid := model3d.NewColliderSolid(model3d.MeshToCollider(mesh))

	log.Println("Sampling points...")
	points := make([]model3d.Coord3D, numSamples)
	values := make([]bool, numSamples)
	essentials.StatefulConcurrentMap(0, numSamples, func() func(i int) {
		gen := rand.New(rand.NewSource(rand.Int63()))
		min := tree.Min
		max := tree.Max
		size := max.Sub(min)
		return func(i int) {
			point := model3d.XYZ(gen.Float64(), gen.Float64(), gen.Float64()).Mul(size).Add(min)
			points[i] = point
			values[i] = meshSolid.Contains(point)
		}
	})

	log.Println("Pruning...")
	initLeaves := tree.Tree.NumLeaves()
	initQuality := TreeQuality(tree.Tree, points, values)
	for tree.Tree.NumLeaves() > maxLeaves {
		replacement, _ := BestReplacement(tree.Tree, points, values)
		if replacement.Replace == tree.Tree {
			tree.Tree = replacement.With
		} else {
			if !Replace(tree.Tree, replacement.Replace, replacement.With) {
				panic("no branch found; this should be impossible")
			}
		}
	}
	finalQuality := TreeQuality(tree.Tree, points, values)
	log.Printf(
		"Quality went from %f => %f (%d => %d leaves)",
		initQuality,
		finalQuality,
		initLeaves,
		tree.Tree.NumLeaves(),
	)

	log.Println("Saving tree...")
	f, err = os.Create(outputPath)
	essentials.Must(err)
	err = treed.WriteBoundedSolidTree(f, tree)
	f.Close()
	essentials.Must(err)
}

func Replace(t, old, new *treed.SolidTree) bool {
	if t.IsLeaf() {
		return false
	} else if t.LessThan == old {
		t.LessThan = new
		return true
	} else if t.GreaterEqual == old {
		t.GreaterEqual = new
		return true
	} else {
		return Replace(t.LessThan, old, new) || Replace(t.GreaterEqual, old, new)
	}
}

func BestReplacement(
	t *treed.SolidTree,
	inputs []model3d.Coord3D,
	targets []bool,
) (*Replacement, float64) {
	if t.IsLeaf() {
		q := TreeQuality(t, inputs, targets)
		return nil, q
	}

	mid := treed.Partition(t.Axis, t.Threshold, inputs, targets)
	leftRes, leftQuality := BestReplacement(t.LessThan, inputs[:mid], targets[:mid])
	leftOther := TreeQuality(t.LessThan, inputs[mid:], targets[mid:])
	rightRes, rightQuality := BestReplacement(t.GreaterEqual, inputs[mid:], targets[mid:])
	rightOther := TreeQuality(t.GreaterEqual, inputs[:mid], targets[:mid])

	q := leftQuality + rightQuality
	leftNewQuality := leftQuality + leftOther
	rightNewQuality := rightQuality + rightOther
	var res *Replacement
	if leftNewQuality > rightNewQuality {
		res = &Replacement{
			OldQuality: q,
			NewQuality: leftNewQuality,
			Replace:    t,
			With:       t.LessThan,
		}
	} else {
		res = &Replacement{
			OldQuality: q,
			NewQuality: rightNewQuality,
			Replace:    t,
			With:       t.GreaterEqual,
		}
	}
	for _, r := range []*Replacement{leftRes, rightRes} {
		if r != nil && r.Delta() > res.Delta() {
			res = r
		}
	}
	return res, q
}

type Replacement struct {
	OldQuality float64
	NewQuality float64

	Replace *treed.SolidTree
	With    *treed.SolidTree
}

func (r *Replacement) Delta() float64 {
	return r.NewQuality - r.OldQuality
}

func TreeQuality(
	t *treed.SolidTree,
	inputs []model3d.Coord3D,
	targets []bool,
) float64 {
	var quality float64
	for i, target := range targets {
		if t.Predict(inputs[i]) == target {
			quality++
		}
	}
	return quality
}
