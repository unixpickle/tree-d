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
	var lr float64
	var weightDecay float64
	var momentum float64
	var iters int
	var taoIters int
	var depth int
	var minLeafSize int
	var datasetSize int
	var taoDatasetSize int
	var surfaceSamples int
	var surfaceEpsilon float64
	var activeRebuilds int
	var activePoints int
	var activeGridSize int
	var activeEpsilon float64
	var axisResolution int
	var verbose bool
	flag.Float64Var(&lr, "lr", 0.1, "learning rate for SVM training")
	flag.Float64Var(&weightDecay, "weight-decay", 1e-4, "weight decay for SVM training")
	flag.Float64Var(&momentum, "momentum", 0.9, "Nesterov momentum for SVM training")
	flag.IntVar(&iters, "iters", 1000, "iterations for SVM training")
	flag.IntVar(&taoIters, "tao-iters", 10, "maximum iterations of TAO")
	flag.IntVar(&depth, "depth", 6, "maximum tree depth")
	flag.IntVar(&minLeafSize, "min-leaf-size", 5, "minimum samples per leaf for greedy trees")
	flag.IntVar(&datasetSize, "dataset-size", 1000000, "number of points to sample for dataset")
	flag.IntVar(&surfaceSamples, "surface-samples", 0,
		"number of points to sample near the surface for the dataset")
	flag.Float64Var(&surfaceEpsilon, "surface-epsilon", 0.01, "noise scale for sampling near surface")
	flag.IntVar(&taoDatasetSize, "tao-dataset-size", 1000000, "number of points to sample for TAO")
	flag.IntVar(&activeRebuilds, "active-rebuilds", 1,
		"number of times to rebuild with active learning")
	flag.IntVar(&activePoints, "active-points", 50000,
		"number of points to sample for active learning steps")
	flag.IntVar(&activeGridSize, "active-grid-size", 64, "grid size for active learning mesh")
	flag.Float64Var(&activeEpsilon, "active-epsilon", 0.01, "noise scale for active learning")
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
	coords, labels := Dataset(inputMesh, solid, datasetSize, surfaceSamples, surfaceEpsilon)

	log.Println("Building initial tree...")
	axes := model3d.NewMeshIcosphere(model3d.Origin, 1.0, axisResolution).VertexSlice()
	greedyLoss := treed.EntropySplitLoss[float64]{MinCount: minLeafSize}
	tree := treed.GreedyTree[float64, model3d.Coord3D, bool](
		axes,
		coords,
		labels,
		greedyLoss,
		0,
		depth,
	)
	for i := 0; i < activeRebuilds; i++ {
		essentials.Must(WriteTree(outputPath, solid, tree))
		log.Printf("Apply active learning rebuild %d/%d...", i+1, activeRebuilds)
		coords, labels = ActiveLearning(
			tree,
			solid,
			coords,
			labels,
			activePoints,
			activeGridSize,
			activeEpsilon,
			verbose,
		)
		tree = treed.GreedyTree[float64, model3d.Coord3D, bool](
			axes,
			coords,
			labels,
			greedyLoss,
			0,
			depth,
		)
	}

	log.Println("Sampling TAO dataset...")
	if len(coords) < taoDatasetSize {
		newCoords, newLabels := Dataset(
			inputMesh, solid, datasetSize, surfaceSamples, surfaceEpsilon,
		)
		coords = append(coords, newCoords...)
		labels = append(labels, newLabels...)
	}
	testCoords, testLabels := SolidDataset(solid, taoDatasetSize)

	log.Println("Refining tree with TAO...")
	tao := treed.TAO[float64, model3d.Coord3D, bool]{
		Loss:        treed.EqualityTAOLoss[bool]{},
		LR:          lr,
		WeightDecay: weightDecay,
		Momentum:    momentum,
		Iters:       iters,
		Verbose:     verbose,
	}
	testLoss := tao.EvaluateLoss(tree, testCoords, testLabels)
	for i := 0; i < taoIters; i++ {
		essentials.Must(WriteTree(outputPath, solid, tree))
		coords, labels = ActiveLearning(
			tree,
			solid,
			coords,
			labels,
			activePoints,
			activeGridSize,
			activeEpsilon,
			verbose,
		)

		result := tao.Optimize(tree, coords, labels)
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
	tree = tree.Simplify(coords, labels, tao.Loss)
	newCount := tree.NumLeaves()
	log.Printf(" => went from %d to %d leaves", oldCount, newCount)

	log.Println("Writing output...")
	essentials.Must(WriteTree(outputPath, solid, tree))
}

func WriteTree(outputPath string, solid model3d.Solid, tree *treed.SolidTree) error {
	f, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer f.Close()
	return treed.WriteBoundedSolidTree(f, &treed.BoundedSolidTree{
		Min:  solid.Min(),
		Max:  solid.Max(),
		Tree: tree,
	})
}

func Dataset(
	mesh *model3d.Mesh,
	solid model3d.Solid,
	datasetSize int,
	surfaceSamples int,
	eps float64,
) ([]model3d.Coord3D, []bool) {
	coords, labels := SolidDataset(solid, datasetSize-surfaceSamples)
	if surfaceSamples > 0 {
		meshCoords, meshLabels := MeshDataset(mesh, solid, surfaceSamples, eps)
		coords = append(coords, meshCoords...)
		labels = append(labels, meshLabels...)
	}
	return coords, labels
}

func SolidDataset(solid model3d.Solid, numPoints int) (points []model3d.Coord3D, labels []bool) {
	min, max := PaddedBounds(solid)

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

func MeshDataset(
	mesh *model3d.Mesh,
	solid model3d.Solid,
	numPoints int,
	eps float64,
) (points []model3d.Coord3D, labels []bool) {
	points = make([]model3d.Coord3D, numPoints)
	labels = make([]bool, numPoints)
	delta := eps * mesh.Min().Dist(mesh.Max())
	essentials.StatefulConcurrentMap(0, numPoints, func() func(int) {
		sampler := treed.MeshPointSampler(mesh)
		return func(i int) {
			point := sampler().Add(model3d.NewCoord3DRandNorm().Scale(delta))
			label := solid.Contains(point)
			points[i] = point
			labels[i] = label
		}
	})
	return
}

func PaddedBounds(solid model3d.Solid) (min, max model3d.Coord3D) {
	min, max = solid.Min(), solid.Max()
	size := min.Dist(max)
	min = min.AddScalar(-size * 0.1)
	max = max.AddScalar(size * 0.1)
	return
}

func ActiveLearning(
	tree *treed.SolidTree,
	solid model3d.Solid,
	coords []model3d.Coord3D,
	labels []bool,
	activePoints int,
	activeGridSize int,
	activeEpsilon float64,
	verbose bool,
) ([]model3d.Coord3D, []bool) {
	if activePoints == 0 {
		return coords, labels
	}

	if verbose {
		log.Printf("Creating %d active learning samples...", activePoints)
	}
	min, max := PaddedBounds(solid)
	epsilon := min.Dist(max) * activeEpsilon

	boundedTree := &treed.BoundedSolidTree{Tree: tree, Min: min, Max: max}
	activeSamples := treed.SampleDecisionBoundaryCast(boundedTree, activePoints/2, activePoints*128)
	if len(activeSamples) < activePoints/2 {
		extra := treed.SampleDecisionBoundaryMesh(
			boundedTree,
			activePoints/2-len(activeSamples),
			activeGridSize,
		)
		activeSamples = append(activeSamples, extra...)
	}

	// Sample some points slightly outside the decision boundary
	// to allow faster growth/shrinking.
	for i := 0; i < len(activeSamples); i += 2 {
		activeSamples[i] = activeSamples[i].Add(model3d.NewCoord3DRandNorm().Scale(epsilon))
	}

	// Sample around points that are misclassified.
	var badPoints []model3d.Coord3D
	for i, c := range coords {
		if tree.Predict(c) != labels[i] {
			badPoints = append(badPoints, c)
		}
	}
	if len(badPoints) > 0 {
		for i := 0; i < activePoints/2; i++ {
			point := badPoints[rand.Intn(len(badPoints))]
			point = point.Add(model3d.NewCoord3DRandNorm().Scale(epsilon))
			activeSamples = append(activeSamples, point)
		}
	}

	var numCorrect int
	for _, c := range activeSamples {
		label := solid.Contains(c)
		pred := tree.Predict(c)
		coords = append(coords, c)
		labels = append(labels, label)
		if pred == label {
			numCorrect++
		}
	}
	if verbose {
		log.Printf("=> active accuracy is %f", float64(numCorrect)/float64(activePoints))
	}

	return coords, labels
}
