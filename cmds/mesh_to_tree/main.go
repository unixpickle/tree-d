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
	var axisResolution int
	var verbose bool
	flag.Float64Var(&lr, "lr", 0.1, "learning rate for SVM training")
	flag.Float64Var(&weightDecay, "weight-decay", 1e-4, "weight decay for SVM training")
	flag.Float64Var(&momentum, "momentum", 0.9, "Nesterov momentum for SVM training")
	flag.IntVar(&iters, "iters", 1000, "iterations for SVM training")
	flag.IntVar(&taoIters, "tao-iters", 10, "maximum iterations of TAO")
	flag.IntVar(&depth, "depth", 6, "maximum tree depth")
	flag.IntVar(&datasetSize, "dataset-size", 1000000, "number of points to sample for dataset")
	flag.IntVar(&axisResolution, "axis-resolution", 2,
		"number of icosphere subdivisions to do when creating split axes")
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
	coll := model3d.MeshToCollider(inputMesh)
	solid := model3d.NewColliderSolid(coll)
	coords, labels := SolidDataset(solid, datasetSize)

	log.Println("Building initial tree...")
	axes := model3d.NewMeshIcosphere(model3d.Origin, 1.0, axisResolution).VertexSlice()
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
		coords, labels = SolidDataset(solid, datasetSize)
		result := tao.Optimize(tree, coords, labels)
		if result.NewLoss >= result.OldLoss {
			log.Printf("no improvement at iteration %d: loss=%f", i, result.OldLoss)
			break
		}
		log.Printf("TAO iteration %d: loss=%f->%f", i, result.OldLoss, result.NewLoss)
		tree = result.Tree
	}

	log.Println("Simplifying tree...")
	oldCount := tree.NumLeaves()
	tree = tree.Simplify(coords, labels, tao.Loss)
	newCount := tree.NumLeaves()
	log.Printf(" => went from %d to %d leaves", oldCount, newCount)

	log.Println("Writing output...")
	f, err = os.Create(outputPath)
	essentials.Must(err)
	defer f.Close()
	essentials.Must(json.NewEncoder(f).Encode(&treed.BoundedTree[float64, model3d.Coord3D, bool]{
		Min:  inputMesh.Min(),
		Max:  inputMesh.Max(),
		Tree: tree,
	}))
}

func SolidDataset(solid model3d.Solid, numPoints int) (points []model3d.Coord3D, labels []bool) {
	min, max := solid.Min(), solid.Max()
	size := min.Dist(max)
	min = min.AddScalar(-size * 0.1)
	max = max.AddScalar(size * 0.1)

	points = make([]model3d.Coord3D, numPoints)
	labels = make([]bool, numPoints)

	essentials.ConcurrentMap(0, numPoints, func(i int) {
		point := model3d.NewCoord3DRandBounds(min, max)
		label := solid.Contains(point)
		points[i] = point
		labels[i] = label
	})

	return
}
