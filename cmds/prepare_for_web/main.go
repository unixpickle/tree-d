package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/model3d"
	"github.com/unixpickle/tree-d/treed"
)

func main() {
	var meshPath string
	var modelPath string
	var normalsPath string
	var outputPath string
	var numSamples int
	flag.StringVar(&meshPath, "mesh", "", "path to input mesh")
	flag.StringVar(&modelPath, "model", "", "path to input model")
	flag.StringVar(&normalsPath, "normals", "", "path to normal map")
	flag.StringVar(&outputPath, "output", "", "path to output directory")
	flag.IntVar(&numSamples, "num-samples", 2000000, "number of samples for simplification")
	flag.Parse()
	if modelPath == "" || normalsPath == "" || outputPath == "" {
		essentials.Die("Missing required -mesh, -model, -normals, or -output flags. See -help.")
	}

	log.Println("Loading input tree...")
	f, err := os.Open(modelPath)
	essentials.Must(err)
	model, err := treed.ReadBoundedSolidTree(f)
	f.Close()
	essentials.Must(err)

	log.Println("Loading normal map...")
	f, err = os.Open(modelPath)
	essentials.Must(err)
	normals, err := treed.ReadCoordTree(f)
	f.Close()
	essentials.Must(err)

	log.Println("Loading mesh...")
	f, err = os.Open(meshPath)
	essentials.Must(err)
	tris, err := model3d.ReadSTL(f)
	f.Close()
	essentials.Must(err)
	mesh := model3d.NewMeshTriangles(tris)
	meshSolid := model3d.NewColliderSolid(model3d.MeshToCollider(mesh))

	log.Println("Sampling points...")
	points := make([]model3d.Coord3D, numSamples)
	values := make([]bool, numSamples)
	essentials.StatefulConcurrentMap(0, numSamples, func() func(i int) {
		gen := rand.New(rand.NewSource(rand.Int63()))
		min := model.Min
		max := model.Max
		size := max.Sub(min)
		return func(i int) {
			point := model3d.XYZ(gen.Float64(), gen.Float64(), gen.Float64()).Mul(size).Add(min)
			points[i] = point
			values[i] = meshSolid.Contains(point)
		}
	})

	log.Println("Writing outputs...")
	essentials.Must(os.MkdirAll(outputPath, 0755))

	offset := model.Max.Mid(model.Min).Scale(-1)
	scale := 2 / model.Max.Sub(model.Min).Abs().MaxCoord()
	model = model.Translate(offset).Scale(scale)
	normals = normals.Translate(offset).Scale(scale)

	metadata := &Metadata{
		Normals: WriteNormals(filepath.Join(outputPath, "normals.bin"), normals),
		LODs: []*TreeInfo{
			WriteTree(filepath.Join(outputPath, "full.bin"), model),
		},
	}

	log.Println("Writing LODs...")
	for _, lod := range []int{1024, 512, 256} {
		numLeaves := model.Tree.NumLeaves()
		if numLeaves <= lod {
			continue
		}
		log.Printf(" - working on LOD %d", lod)
		for model.Tree.NumLeaves() > lod {
			rep, _ := treed.BestReplacement[float64, model3d.Coord3D, bool](
				model.Tree,
				treed.EqualityTAOLoss[bool]{},
				points,
				values,
				0,
			)
			model.Tree, _ = model.Tree.Replace(rep.Replace, rep.With)
		}
		lodPath := filepath.Join(outputPath, fmt.Sprintf("lod_%d.bin", lod))
		metadata.LODs = append(metadata.LODs, WriteTree(lodPath, model))
	}

	log.Println("Saving metadata...")
	f, err = os.Create(filepath.Join(outputPath, "metadata.json"))
	essentials.Must(err)
	err = json.NewEncoder(f).Encode(metadata)
	f.Close()
	essentials.Must(err)
}

func WriteTree(path string, tree *treed.BoundedSolidTree) *TreeInfo {
	f, err := os.Create(path)
	essentials.Must(err)
	defer f.Close()
	essentials.Must(treed.WriteBoundedSolidTree(f, tree))
	info, err := f.Stat()
	essentials.Must(err)
	return &TreeInfo{
		NumLeaves: tree.Tree.NumLeaves(),
		FileSize:  info.Size(),
	}
}

func WriteNormals(path string, tree *treed.CoordTree) *TreeInfo {
	f, err := os.Create(path)
	essentials.Must(err)
	defer f.Close()
	essentials.Must(treed.WriteCoordTree(f, tree))
	info, err := f.Stat()
	essentials.Must(err)
	return &TreeInfo{
		NumLeaves: tree.NumLeaves(),
		FileSize:  info.Size(),
	}
}

type Metadata struct {
	Normals *TreeInfo   `json:"num_leaves"`
	LODs    []*TreeInfo `json:"lods"`
}

type TreeInfo struct {
	NumLeaves int   `json:"num_leaves"`
	FileSize  int64 `json:"file_size"`
}