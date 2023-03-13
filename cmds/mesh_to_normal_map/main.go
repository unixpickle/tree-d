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
	var taoIters int
	var lr float64
	var weightDecay float64
	var momentum float64
	var iters int
	var minLeafSize int
	var axisResolution int
	var verbose bool
	flag.IntVar(&datasetSize, "dataset-size", 1000000, "dataset size for surface")
	flag.IntVar(&depth, "max-depth", 8, "maximum tree depth")
	flag.IntVar(&taoIters, "tao-iters", 5, "maximum number of TAO iterations")
	flag.Float64Var(&lr, "lr", 0.1, "learning rate for SVM training")
	flag.Float64Var(&weightDecay, "weight-decay", 1e-4, "weight decay for SVM training")
	flag.Float64Var(&momentum, "momentum", 0.9, "Nesterov momentum for SVM training")
	flag.IntVar(&iters, "iters", 1000, "iterations for SVM training")
	flag.IntVar(&minLeafSize, "min-leaf-size", 5, "minimum samples per leaf for greedy trees")
	flag.IntVar(&axisResolution, "axis-resolution", 2,
		"number of icosphere subdivisions to do when creating split axes")
	flag.BoolVar(&verbose, "verbose", false, "print out extra optimization information")
	flag.Parse()

	args := flag.Args()
	if len(args) != 3 {
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
	sampleDataset := func() (labels, targets []model3d.Coord3D) {
		labels = treed.SampleDecisionBoundaryCast(solidTree, datasetSize, 0)
		targets = make([]model3d.Coord3D, len(labels))
		for i, x := range labels {
			targets[i], _ = meshField.NormalSDF(x)
		}
		return
	}
	inputs, targets := sampleDataset()

	log.Println("Building greedy tree...")
	axes := model3d.NewMeshIcosphere(model3d.Origin, 1.0, axisResolution).VertexSlice()
	tree := treed.GreedyTree[float64, model3d.Coord3D, model3d.Coord3D](
		axes,
		inputs,
		targets,
		treed.VarianceSplitLoss[float64, model3d.Coord3D]{MinCount: minLeafSize},
		0,
		depth,
	)

	log.Println("Performing TAO...")
	testCoords, testLabels := sampleDataset()
	tao := treed.TAO[float64, model3d.Coord3D, model3d.Coord3D]{
		Loss:        treed.SquaredErrorTAOLoss[float64, model3d.Coord3D]{},
		LR:          lr,
		WeightDecay: weightDecay,
		Momentum:    momentum,
		Iters:       iters,
		Verbose:     verbose,
	}
	testLoss := tao.EvaluateLoss(tree, testCoords, testLabels)
	for i := 0; i < taoIters; i++ {
		essentials.Must(WriteTree(outputPath, tree))
		result := tao.Optimize(tree, inputs, targets)
		if result.NewLoss >= result.OldLoss {
			log.Printf("no improvement at iteration %d: loss=%f test_loss=%f", i, result.OldLoss,
				testLoss)
			break
		}
		newTestLoss := tao.EvaluateLoss(result.Tree, testCoords, testLabels)

		log.Printf("TAO iteration %d: loss=%f->%f test_loss=%f->%f", i, result.OldLoss,
			result.NewLoss, testLoss, newTestLoss)

		testLoss = newTestLoss
		tree = result.Tree
	}

	log.Println("Simplifying tree...")
	oldCount := tree.NumLeaves()
	tree = tree.Simplify(inputs, targets, tao.Loss)
	newCount := tree.NumLeaves()
	log.Printf(" => went from %d to %d leaves", oldCount, newCount)

	log.Println("Writing output...")
	essentials.Must(WriteTree(outputPath, tree))
}

func WriteTree(outputPath string, tree *treed.CoordTree) error {
	f, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer f.Close()
	return treed.WriteCoordTree(f, tree)
}
