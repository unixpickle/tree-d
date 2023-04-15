package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

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
	var initDatasetSize int
	var minDatasetSize int
	var axisResolution int
	var mutationCount int
	var mutationStddev flagFloats = []float64{0.025}
	var hitAndRunIterations int
	var verbose bool
	flag.Float64Var(&lr, "lr", 0.1, "learning rate for SVM training")
	flag.Float64Var(&weightDecay, "weight-decay", 1e-4, "weight decay for SVM training")
	flag.Float64Var(&momentum, "momentum", 0.9, "Nesterov momentum for SVM training")
	flag.IntVar(&iters, "iters", 1000, "iterations for SVM training")
	flag.IntVar(&taoIters, "tao-iters", 50, "maximum iterations of TAO")
	flag.IntVar(&depth, "depth", 20, "maximum tree depth")
	flag.IntVar(&minLeafSize, "min-leaf-size", 5, "minimum samples per leaf when splitting")
	flag.IntVar(&initDatasetSize, "init-dataset-size", 50000,
		"number of points to sample for dataset")
	flag.IntVar(&minDatasetSize, "min-dataset-size", 1000, "minimum dataset size at leaves")
	flag.IntVar(&axisResolution, "axis-resolution", 2,
		"number of icosphere subdivisions to do when creating split axes")
	flag.IntVar(&mutationCount, "mutation-count", 30, "number of mutation directions")
	flag.Var(&mutationStddev, "mutation-stddev", "scale of mutations; may be comma-separated list")
	flag.IntVar(&hitAndRunIterations, "hit-and-run-iterations", 20,
		"minimum dataset size at leaves")
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
	inputTris, err := treed.Load(inputPath, model3d.ReadSTL)
	essentials.Must(err)
	inputMesh := model3d.NewMeshTriangles(inputTris)
	coll := model3d.MeshToCollider(inputMesh)
	solid := model3d.NewColliderSolid(coll)
	coords, labels := SolidDataset(solid, initDatasetSize)

	log.Println("Building initial tree...")
	mutationCounts := make([]int, len(mutationStddev))
	for i := range mutationStddev {
		mutationCounts[i] = mutationCount
	}
	axisSchedule := &treed.MutationAxisSchedule[float64, model3d.Coord3D]{
		Initial: treed.NewConstantAxisScheduleIcosphere(axisResolution).Init(),
		Counts:  mutationCounts,
		Stddevs: mutationStddev,
	}
	greedyLoss := treed.EntropySplitLoss[float64]{MinCount: minLeafSize}
	sampler := &treed.HitAndRunSampler[float64, model3d.Coord3D]{
		Iterations: hitAndRunIterations,
	}
	bounds := treed.NewPolytopeBounds(solid.Min(), solid.Max())
	tree := treed.AdaptiveGreedyTree[float64, model3d.Coord3D, bool](
		axisSchedule,
		bounds,
		coords,
		labels,
		solid.Contains,
		greedyLoss,
		sampler,
		minDatasetSize,
		0,
		depth,
	)

	testCoords, testLabels := SolidDataset(solid, initDatasetSize)

	log.Println("Refining tree with TAO...")
	tao := treed.TAO[float64, model3d.Coord3D, bool]{
		Loss:        treed.EqualityTAOLoss[bool]{},
		LR:          lr,
		WeightDecay: weightDecay,
		Momentum:    momentum,
		Iters:       iters,
		Verbose:     verbose,

		// Adaptive dataset configuration.
		MinSamples: minDatasetSize,
		Sampler:    sampler,
		Bounds:     bounds,
		Oracle:     solid.Contains,
	}
	testLoss := tao.EvaluateLoss(tree, testCoords, testLabels)
	for i := 0; i < taoIters; i++ {
		essentials.Must(WriteTree(outputPath, solid, tree))

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

	log.Println("Writing output...")
	essentials.Must(WriteTree(outputPath, solid, tree))
}

func WriteTree(outputPath string, solid model3d.Solid, tree *treed.SolidTree) error {
	boundedTree := &treed.BoundedSolidTree{
		Min:  solid.Min(),
		Max:  solid.Max(),
		Tree: tree,
	}
	return treed.Save(outputPath, boundedTree, treed.WriteBoundedSolidTree)
}

func SolidDataset(solid model3d.Solid, numPoints int) (points []model3d.Coord3D, labels []bool) {
	min, max := solid.Min(), solid.Max()

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

type flagFloats []float64

func (i *flagFloats) String() string {
	parts := make([]string, len(*i))
	for j, x := range *i {
		parts[j] = strconv.FormatFloat(x, 'f', 0, 64)
	}
	return strings.Join(parts, ",")
}

func (i *flagFloats) Set(value string) error {
	var res []float64
	for _, part := range strings.Split(value, ",") {
		parsed, err := strconv.ParseFloat(part, 64)
		if err != nil {
			return fmt.Errorf("unexpected part %q: %w", part, err)
		}
		res = append(res, parsed)
	}
	*i = res
	return nil
}
