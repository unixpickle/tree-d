package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
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
	model, err := treed.Load(modelPath, treed.ReadBoundedSolidTree)
	essentials.Must(err)

	log.Println("Loading normal map...")
	normals, err := treed.Load(normalsPath, treed.ReadCoordTree)
	essentials.Must(err)

	log.Println("Loading mesh...")
	tris, err := treed.Load(meshPath, model3d.ReadSTL)
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
	normals = normals.Translate(offset).Scale(scale)

	metadata := &Metadata{
		Normals: WriteNormals(filepath.Join(outputPath, "normals.bin"), normals),
		LODs: []*TreeInfo{
			WriteTree(filepath.Join(outputPath, "full.bin"), model.Translate(offset).Scale(scale)),
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
		lodPath := filepath.Join(outputPath, fmt.Sprintf("lod_%d.bin", model.Tree.NumLeaves()))
		metadata.LODs = append(
			metadata.LODs,
			WriteTree(lodPath, model.Translate(offset).Scale(scale)),
		)
	}

	log.Println("Saving metadata...")
	metadataPath := filepath.Join(outputPath, "metadata.json")
	essentials.Must(treed.Save(metadataPath, metadata, func(w io.Writer, metadata *Metadata) error {
		return json.NewEncoder(w).Encode(metadata)
	}))
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
		Filename:  info.Name(),
		Size:      info.Size(),
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
		Filename:  info.Name(),
		Size:      info.Size(),
	}
}

type Metadata struct {
	Normals *TreeInfo   `json:"normals"`
	LODs    []*TreeInfo `json:"lods"`
}

type TreeInfo struct {
	NumLeaves int    `json:"num_leaves"`
	Filename  string `json:"filename"`
	Size      int64  `json:"file_size"`
}
