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

	log.Println("Creating initial cache...")
	cache := map[*treed.SolidTree]*Replacement{}
	PopulateCache(tree.Tree, points, values, cache)
	initQuality := TreeQuality(tree.Tree, points, values)

	log.Println("Pruning...")
	for tree.Tree.NumLeaves() > maxLeaves {
		replacement := BestInCache(cache)
		if replacement == nil {
			fmt.Println(tree.Tree, cache)
		}
		Replace(tree.Tree, replacement.Replace, replacement.With)
		cache = PruneCache(tree.Tree, cache)
		PopulateCacheUnder(replacement.With, tree.Tree, points, values, cache)
	}
	finalQuality := TreeQuality(tree.Tree, points, values)
	log.Printf("Quality went from %f => %f (at %d leaves)",
		initQuality, finalQuality, tree.Tree.NumLeaves())

	log.Println("Saving tree...")
	f, err = os.Create(outputPath)
	essentials.Must(err)
	err = treed.WriteBoundedSolidTree(f, tree)
	f.Close()
	essentials.Must(err)
}

func Replace(t, old, new *treed.SolidTree) {
	if t.IsLeaf() {
		return
	}
	if t.LessThan == old {
		t.LessThan = new
	} else if t.GreaterEqual == old {
		t.GreaterEqual = new
	} else {
		Replace(t.LessThan, old, new)
		Replace(t.GreaterEqual, old, new)
	}
}

func BestInCache(cache map[*treed.SolidTree]*Replacement) *Replacement {
	var res *Replacement
	for _, v := range cache {
		if res == nil || res.NewQuality-res.OldQuality < v.NewQuality-v.OldQuality {
			res = v
		}
	}
	return res
}

func PruneCache(t *treed.SolidTree, cache map[*treed.SolidTree]*Replacement) map[*treed.SolidTree]*Replacement {
	seen := map[*treed.SolidTree]struct{}{}
	var treeNodes func(t *treed.SolidTree)
	treeNodes = func(t *treed.SolidTree) {
		if !t.IsLeaf() {
			seen[t] = struct{}{}
			treeNodes(t.LessThan)
			treeNodes(t.GreaterEqual)
		}
	}
	treeNodes(t)

	res := map[*treed.SolidTree]*Replacement{}
	for k, v := range cache {
		if _, ok := seen[k]; ok {
			res[k] = v
		}
	}
	return res
}

func PopulateCacheUnder(
	parent *treed.SolidTree,
	t *treed.SolidTree,
	inputs []model3d.Coord3D,
	targets []bool,
	cache map[*treed.SolidTree]*Replacement,
) {
	if t == parent {
		PopulateCache(t, inputs, targets, cache)
	} else if !t.IsLeaf() {
		mid := treed.Partition(t.Axis, t.Threshold, inputs, targets)
		PopulateCacheUnder(parent, t.LessThan, inputs[:mid], targets[:mid], cache)
		PopulateCacheUnder(parent, t.GreaterEqual, inputs[mid:], targets[mid:], cache)
	}
}

func PopulateCache(
	t *treed.SolidTree,
	inputs []model3d.Coord3D,
	targets []bool,
	cache map[*treed.SolidTree]*Replacement,
) float64 {
	if t.IsLeaf() {
		q := TreeQuality(t, inputs, targets)
		return q
	}

	mid := treed.Partition(t.Axis, t.Threshold, inputs, targets)
	leftQuality := PopulateCache(t.LessThan, inputs[:mid], targets[:mid], cache)
	leftOther := TreeQuality(t.LessThan, inputs[mid:], targets[mid:])
	rightQuality := PopulateCache(t.GreaterEqual, inputs[mid:], targets[mid:], cache)
	rightOther := TreeQuality(t.GreaterEqual, inputs[:mid], targets[:mid])

	q := leftQuality + rightQuality
	leftNewQuality := leftQuality + leftOther
	rightNewQuality := rightQuality + rightOther
	onlyRight := &Replacement{
		OldQuality: q,
		NewQuality: leftNewQuality,
		Replace:    t,
		With:       t.LessThan,
	}
	onlyLeft := &Replacement{
		OldQuality: q,
		NewQuality: rightNewQuality,
		Replace:    t,
		With:       t.GreaterEqual,
	}
	cache[t] = BestReplacement(onlyRight, onlyLeft)
	return q
}

type Replacement struct {
	OldQuality float64
	NewQuality float64

	Replace *treed.SolidTree
	With    *treed.SolidTree
}

func BestReplacement(prunes ...*Replacement) *Replacement {
	var res *Replacement
	for _, p := range prunes {
		if res == nil {
			res = p
		} else if p.NewQuality > res.NewQuality {
			res = p
		}
	}
	return res
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
