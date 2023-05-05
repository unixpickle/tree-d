package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/model3d"
	"github.com/unixpickle/model3d/render3d"
	"github.com/unixpickle/tree-d/treed"
)

func main() {
	var gridSize int
	var imageSize int
	var fps float64
	var frames int
	var normalMapPath string
	flag.IntVar(&gridSize, "grid-size", 3, "grid size (used for rows and columns)")
	flag.IntVar(&imageSize, "image-size", 300, "size of each image in the grid")
	flag.Float64Var(&fps, "fps", 10.0, "FPS for GIF outputs")
	flag.IntVar(&frames, "frames", 20, "total number of frames for GIF outputs")
	flag.StringVar(&normalMapPath, "normal-map", "", "path to optional normal map tree")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage: render_tree [flags] <input.bin> <output.png>")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
	}
	flag.Parse()

	args := flag.Args()
	if len(args) != 2 {
		flag.Usage()
		os.Exit(1)
	}
	inputPath, outputPath := args[0], args[1]

	log.Println("Loading tree...")
	tree, err := treed.Load(inputPath, treed.ReadBoundedSolidTree)
	essentials.Must(err)

	log.Println("Creating renderable object...")
	var collider model3d.Collider = treed.NewCollider(tree)
	if normalMapPath != "" {
		log.Println(" - Loading normal map...")
		normalMapTrees, err := treed.LoadMultiple(normalMapPath, treed.ReadCoordTree)
		essentials.Must(err)
		normalMap := treed.VecSumNormEnsemble[float64, model3d.Coord3D, model3d.Coord3D](normalMapTrees)
		collider = treed.MapNormals(collider, normalMap)
	}
	object := render3d.Objectify(collider, nil)

	log.Println("Rendering...")
	ext := filepath.Ext(outputPath)
	if strings.ToLower(ext) == ".gif" {
		essentials.Must(
			render3d.SaveRotatingGIF(
				outputPath,
				object,
				model3d.Z(1),
				model3d.YZ(-1, 0.1).Normalize(),
				imageSize,
				frames,
				fps,
				nil,
			),
		)
	} else {
		essentials.Must(
			render3d.SaveRandomGrid(outputPath, object, gridSize, gridSize, imageSize, nil),
		)
	}
}
