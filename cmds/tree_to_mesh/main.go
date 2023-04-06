package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/model3d"
	"github.com/unixpickle/tree-d/treed"
)

func main() {
	var gridSize int
	flag.IntVar(&gridSize, "grid-size", 64, "marching cubes grid size")
	flag.Parse()

	args := flag.Args()
	if len(args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: tree_to_mesh [flags] <input.bin> <output.stl>")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
		os.Exit(1)
	}
	inputPath, outputPath := args[0], args[1]

	log.Println("Loading tree...")
	tree, err := treed.Load(inputPath, treed.ReadBoundedSolidTree)
	essentials.Must(err)

	log.Println("Creating mesh...")
	solid := model3d.CheckedFuncSolid(
		tree.Min,
		tree.Max,
		func(c model3d.Coord3D) bool {
			return tree.Tree.Predict(c)
		},
	)
	maxSize := tree.Max.Sub(tree.Min).MaxCoord()
	mesh := model3d.MarchingCubesSearch(solid, maxSize/float64(gridSize), 8)
	essentials.Must(mesh.SaveGroupedSTL(outputPath))
}
