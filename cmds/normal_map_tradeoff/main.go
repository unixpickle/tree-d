package main

import (
	"bytes"
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
	var datasetSize int
	var datasetEpsilon float64
	flag.IntVar(&datasetSize, "dataset-size", 1000000, "dataset size for surface")
	flag.Float64Var(&datasetEpsilon, "dataset-epsilon", 1e-4, "noise to add to input points")
	flag.Parse()

	args := flag.Args()
	if len(args) != 3 {
		fmt.Fprintln(os.Stderr, "Usage: mesh_to_normal_map [flags] <tree.bin> <mesh.stl> <map.bin>")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
		os.Exit(1)
	}

	treePath, meshPath, normalsPath := args[0], args[1], args[2]

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

	log.Println("Loading normal map...")
	trees, err := treed.LoadMultiple(normalsPath, treed.ReadCoordTree)
	essentials.Must(err)

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

	log.Println("Evaluating trees...")
	preds := make([]model3d.Coord3D, len(targets))
	sizes := make([]int, len(trees))
	mses := make([]float64, len(trees))
	dots := make([]float64, len(trees))
	for i, tree := range trees {
		for j, x := range inputs {
			p := preds[j].Add(tree.Predict(x))
			preds[j] = p
			actual := targets[j]
			mses[i] += actual.SquaredDist(p)
			dots[i] += actual.Dot(p.Normalize())
		}
		w := bytes.NewBuffer(nil)
		treed.WriteCoordTree(w, tree)
		sizes[i] = len(w.Bytes())
		if i > 0 {
			sizes[i] += sizes[i-1]
		}
		mses[i] /= float64(len(inputs))
		dots[i] /= float64(len(inputs))
	}

	log.Printf("Sizes: %v", JSONString(sizes))
	log.Printf("MSE: %v", JSONString(mses))
	log.Printf("Dot: %v", JSONString(dots))
}

func JSONString(x any) string {
	data, err := json.Marshal(x)
	essentials.Must(err)
	return string(data)
}
