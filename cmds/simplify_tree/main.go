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
		min := mesh.Min()
		max := mesh.Max()
		size := max.Sub(min)
		return func(i int) {
			point := model3d.XYZ(gen.Float64(), gen.Float64(), gen.Float64()).Mul(size).Add(min)
			points[i] = point
			values[i] = meshSolid.Contains(point)
		}
	})

	// TODO: simplify tree here.

	log.Println("Saving tree...")
	f, err = os.Create(outputPath)
	essentials.Must(err)
	err = treed.WriteBoundedSolidTree(f, tree)
	f.Close()
	essentials.Must(err)
}
