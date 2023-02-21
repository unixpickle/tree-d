package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/model3d"
	"github.com/unixpickle/tree-d/treed"
)

func main() {
	var lr float64
	var weightDecay float64
	var momentum float64
	var iters int
	var taoIters int
	var depth int
	var datasetSize int
	var verbose bool
	flag.Float64Var(&lr, "lr", 1.0, "learning rate for SVM training")
	flag.Float64Var(&weightDecay, "weight-decay", 1e-4, "weight decay for SVM training")
	flag.Float64Var(&momentum, "momentum", 0.9, "Nesterov momentum for SVM training")
	flag.IntVar(&iters, "iters", 5000, "iterations for SVM training")
	flag.IntVar(&taoIters, "tao-iters", 10, "maximum iterations of TAO")
	flag.IntVar(&depth, "depth", 8, "maximum tree depth")
	flag.IntVar(&datasetSize, "dataset-size", 1000000, "number of points to sample for dataset")
	flag.BoolVar(&verbose, "verbose", false, "print out extra optimization information")
	flag.Parse()

	args := flag.Args()
	if len(args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: mesh_to_tree [flags] <input.stl> <output.json>")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
		os.Exit(1)
	}
	inputPath, outputPath := args[0], args[1]

	log.Println("Creating mesh dataset...")
	f, err := os.Open(inputPath)
	essentials.Must(err)
	inputTris, err := model3d.ReadSTL(f)
	f.Close()
	essentials.Must(err)
	inputMesh := model3d.NewMeshTriangles(inputTris)
	coords, labels := MeshDataset(inputMesh, datasetSize)

	log.Println("Building initial tree...")
	axes := []model3d.Coord3D{
		model3d.X(1),
		model3d.Y(1),
		model3d.Z(1),
	}
	tree := treed.GreedyTree[float64, model3d.Coord3D, bool](
		axes,
		coords,
		labels,
		treed.EntropySplitLoss[float64]{},
		0,
		depth,
	)

	log.Println("Refining tree with TAO...")
	tao := treed.TAO[float64, model3d.Coord3D, bool]{
		Loss:        treed.EqualityTAOLoss[bool]{},
		LR:          lr,
		WeightDecay: weightDecay,
		Momentum:    momentum,
		Iters:       iters,
		Verbose:     verbose,
	}
	for i := 0; i < taoIters; i++ {
		result := tao.Optimize(tree, coords, labels)
		if result.NewLoss >= result.OldLoss {
			log.Printf("no improvement at iteration %d: loss=%f", i, result.OldLoss)
			break
		}
		log.Printf("TAO iteration %d: loss=%f->%f", i, result.OldLoss, result.NewLoss)
		tree = result.Tree
	}

	log.Println("Writing output...")
	f, err = os.Create(outputPath)
	essentials.Must(err)
	defer f.Close()
	essentials.Must(json.NewEncoder(f).Encode(tree))
}

func MeshDataset(mesh *model3d.Mesh, numPoints int) (points []model3d.Coord3D, labels []bool) {
	coll := model3d.MeshToCollider(mesh)
	solid := model3d.NewColliderSolid(coll)

	min, max := mesh.Min(), mesh.Max()
	size := min.Dist(max)
	min = min.AddScalar(-size * 0.1)
	max = max.AddScalar(size * 0.1)

	for i := 0; i < numPoints; i++ {
		point := model3d.NewCoord3DRandBounds(min, max)
		label := solid.Contains(point)
		points = append(points, point)
		labels = append(labels, label)
	}

	return
}
