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
	var datasetSize int
	var depth int
	var minLeafSize int
	var axisResolution int
	flag.IntVar(&datasetSize, "dataset-size", 1000000, "dataset size for surface")
	flag.IntVar(&depth, "max-depth", 8, "maximum tree depth")
	flag.IntVar(&minLeafSize, "min-leaf-size", 5, "minimum samples per leaf for greedy trees")
	flag.IntVar(&axisResolution, "axis-resolution", 2,
		"number of icosphere subdivisions to do when creating split axes")
	flag.Parse()

	args := flag.Args()
	if len(args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: mesh_to_normal_map [flags] <tree.bin> <mesh.stl> <output.bin>")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
		os.Exit(1)
	}

	treePath, meshPath, outputPath := args[0], args[1], args[2]

	log.Println("Loading tree...")
	f, err := os.Open(treePath)
	essentials.Must(err)
	solidTree, err := treed.ReadBoundedSolidTree(f)
	f.Close()
	essentials.Must(err)

	log.Println("Loading mesh...")
	f, err = os.Open(meshPath)
	essentials.Must(err)
	inputTris, err := model3d.ReadSTL(f)
	f.Close()
	essentials.Must(err)
	inputMesh := model3d.NewMeshTriangles(inputTris)
	meshField := model3d.MeshToSDF(inputMesh)

	log.Println("Sampling dataset...")
	labels := treed.SampleDecisionBoundaryCast(solidTree, datasetSize, 0)
	targets := make([]model3d.Coord3D, len(labels))
	for i, x := range labels {
		targets[i], _ = meshField.NormalSDF(x)
	}

	log.Println("Building greedy tree...")
	axes := model3d.NewMeshIcosphere(model3d.Origin, 1.0, axisResolution).VertexSlice()
	tree := treed.GreedyTree[float64, model3d.Coord3D, model3d.Coord3D](
		axes,
		labels,
		targets,
		treed.VarianceSplitLoss[float64, model3d.Coord3D]{MinCount: minLeafSize},
		0,
		depth,
	)

	// TODO: write tree.

	_ = tree
	_ = outputPath
	// TODO: delete above line
}