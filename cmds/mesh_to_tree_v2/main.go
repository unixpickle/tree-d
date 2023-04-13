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
	var iters int
	var depth int
	var minLeafSize int
	var initDatasetSize int
	var minDatasetSize int
	var axisResolution int
	var mutationCount int
	var mutationStddev float64
	var hitAndRunIterations int
	var verbose bool
	flag.IntVar(&iters, "iters", 1000, "iterations for SVM training")
	flag.IntVar(&depth, "depth", 20, "maximum tree depth")
	flag.IntVar(&minLeafSize, "min-leaf-size", 5, "minimum samples per leaf when splitting")
	flag.IntVar(&initDatasetSize, "init-dataset-size", 50000,
		"number of points to sample for dataset")
	flag.IntVar(&minDatasetSize, "min-dataset-size", 1000, "minimum dataset size at leaves")
	flag.IntVar(&axisResolution, "axis-resolution", 2,
		"number of icosphere subdivisions to do when creating split axes")
	flag.IntVar(&mutationCount, "mutation-count", 30, "number of mutation directions")
	flag.Float64Var(&mutationStddev, "mutation-stddev", 0.1, "scale of mutations")
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
	axisSchedule := &treed.MutationAxisSchedule[float64, model3d.Coord3D]{
		Initial: treed.NewConstantAxisScheduleIcosphere(axisResolution).Init(),
		Counts:  []int{mutationCount},
		Stddevs: []float64{mutationStddev},
	}
	greedyLoss := treed.EntropySplitLoss[float64]{MinCount: minLeafSize}
	tree := treed.AdaptiveGreedyTree[float64, model3d.Coord3D, bool](
		axisSchedule,
		treed.NewPolytopeBounds(solid.Min(), solid.Max()),
		coords,
		labels,
		solid.Contains,
		greedyLoss,
		&treed.HitAndRunSampler[float64, model3d.Coord3D]{
			Iterations: hitAndRunIterations,
		},
		minDatasetSize,
		0,
		depth,
	)

	// log.Println("Simplifying tree...")
	// oldCount := tree.NumLeaves()
	// tree = tree.Simplify(coords, labels, treed.EqualityTAOLoss[bool]{})
	// newCount := tree.NumLeaves()
	// log.Printf(" => went from %d to %d leaves", oldCount, newCount)

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
