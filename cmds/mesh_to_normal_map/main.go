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
	var datasetEpsilon float64
	var numTrees int
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
	flag.Float64Var(&datasetEpsilon, "dataset-epsilon", 1e-4, "noise to add to input points")
	flag.IntVar(&numTrees, "num-trees", 3, "number of trees in ensemble")
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
	solidTree, err := treed.Load(treePath, treed.ReadBoundedSolidTree)
	essentials.Must(err)

	log.Println("Loading mesh...")
	inputTris, err := treed.Load(meshPath, model3d.ReadSTL)
	essentials.Must(err)
	inputMesh := model3d.NewMeshTriangles(inputTris)
	removed := 0
	inputMesh.Iterate(func(t *model3d.Triangle) {
		if t.Area() == 0 {
			removed++
			inputMesh.Remove(t)
		}
	})
	if removed > 0 {
		log.Printf(" - removed %d invalid triangles", removed)
	}
	meshField := model3d.MeshToSDF(inputMesh)

	log.Println("Sampling dataset...")
	sampleDataset := func() (inputs, targets []model3d.Coord3D) {
		meshScale := meshField.Min().Dist(meshField.Max())
		noiseScale := meshScale * datasetEpsilon
		inputs = treed.SampleDecisionBoundaryCast(solidTree, datasetSize, 0)
		targets = make([]model3d.Coord3D, len(inputs))
		essentials.ConcurrentMap(0, len(inputs), func(i int) {
			inputs[i] = inputs[i].Add(model3d.NewCoord3DRandNorm().Scale(noiseScale))
			targets[i], _ = meshField.NormalSDF(inputs[i])
		})
		return
	}
	inputs, targets := sampleDataset()
	testInputs, testTargets := sampleDataset()

	var trees []*treed.CoordTree
	for i := 0; i < numTrees; i++ {
		log.Printf("Creating tree %d/%d ...", i+1, numTrees)
		tree := BuildTree(
			inputs,
			targets,
			testInputs,
			testTargets,
			axisResolution,
			depth,
			minLeafSize,
			lr,
			weightDecay,
			momentum,
			iters,
			taoIters,
			verbose,
		)
		trees = append(trees, tree)

		getResidual := func(t *treed.CoordTree, inputs, targets []model3d.Coord3D) {
			for i, x := range inputs {
				targets[i] = targets[i].Sub(t.Predict(x))
			}
		}
		getResidual(tree, inputs, targets)
		getResidual(tree, testInputs, testTargets)
	}

	log.Println("Writing output...")
	essentials.Must(treed.SaveMultiple(outputPath, trees, treed.WriteCoordTree))
}

func BuildTree(
	inputs []model3d.Coord3D,
	targets []model3d.Coord3D,
	testInputs []model3d.Coord3D,
	testTargets []model3d.Coord3D,
	axisResolution int,
	depth int,
	minLeafSize int,
	lr float64,
	weightDecay float64,
	momentum float64,
	iters int,
	taoIters int,
	verbose bool,
) *treed.CoordTree {
	log.Println("Building greedy tree...")
	axes := treed.NewConstantAxisScheduleIcosphere(axisResolution).Init()
	tree := treed.GreedyTree[float64, model3d.Coord3D, model3d.Coord3D](
		axes,
		inputs,
		targets,
		treed.VarianceSplitLoss[float64, model3d.Coord3D]{MinCount: minLeafSize},
		0,
		depth,
	)

	log.Println("Performing TAO...")
	tao := treed.TAO[float64, model3d.Coord3D, model3d.Coord3D]{
		Loss:        treed.SquaredErrorTAOLoss[float64, model3d.Coord3D]{},
		LR:          lr,
		WeightDecay: weightDecay,
		Momentum:    momentum,
		Iters:       iters,
		Verbose:     verbose,
	}
	testLoss := tao.EvaluateLoss(tree, testInputs, testTargets)
	for i := 0; i < taoIters; i++ {
		result := tao.Optimize(tree, inputs, targets)
		if result.NewLoss >= result.OldLoss {
			log.Printf("no improvement at iteration %d: loss=%f test_loss=%f", i, result.OldLoss,
				testLoss)
			break
		}
		newTestLoss := tao.EvaluateLoss(result.Tree, testInputs, testTargets)

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

	return tree
}
