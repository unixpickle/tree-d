package treed

import (
	"encoding/binary"
	"io"

	"github.com/pkg/errors"
	"github.com/unixpickle/model3d/model3d"
)

// WriteBoundedSolidTree serializes b in a 32-bit precision binary format.
func WriteBoundedSolidTree(w io.Writer, b *BoundedSolidTree) error {
	bounds := []float32{
		float32(b.Min.X),
		float32(b.Min.Y),
		float32(b.Min.Z),
		float32(b.Max.X),
		float32(b.Max.Y),
		float32(b.Max.Z),
	}
	if err := binary.Write(w, binary.LittleEndian, bounds); err != nil {
		return errors.Wrap(err, "write bounded solid tree")
	}
	err := writeSolidTree(w, b.Tree)
	if err != nil {
		return errors.Wrap(err, "write bounded solid tree")
	}
	return nil
}

// ReadBoundedSolidTree reads the output written by WriteBoundedSolidTree.
func ReadBoundedSolidTree(r io.Reader) (*BoundedSolidTree, error) {
	var bounds [6]float32
	if err := binary.Read(r, binary.LittleEndian, &bounds); err != nil {
		return nil, errors.Wrap(err, "read bounded solid tree")
	}
	tree, err := readSolidTree(r)
	if err != nil {
		return nil, errors.Wrap(err, "read bounded solid tree")
	}
	return &BoundedSolidTree{
		Min:  model3d.XYZ(float64(bounds[0]), float64(bounds[1]), float64(bounds[2])),
		Max:  model3d.XYZ(float64(bounds[3]), float64(bounds[4]), float64(bounds[5])),
		Tree: tree,
	}, nil
}

// WriteSolidTree serializes t in a 32-bit precision binary format.
func WriteSolidTree(w io.Writer, t *SolidTree) error {
	err := writeSolidTree(w, t)
	if err != nil {
		err = errors.Wrap(err, "write solid tree")
	}
	return err
}

func writeSolidTree(w io.Writer, t *SolidTree) error {
	if t.IsLeaf() {
		var leaf float32
		if t.Leaf {
			leaf = 1
		}
		return binary.Write(w, binary.LittleEndian, []float32{
			0, 0, 0, leaf,
		})
	} else {
		if t.Axis == model3d.Origin {
			panic("cannot encode zero axis for branch")
		}
		err := binary.Write(w, binary.LittleEndian, []float32{
			float32(t.Axis.X),
			float32(t.Axis.Y),
			float32(t.Axis.Z),
			float32(t.Threshold),
		})
		if err != nil {
			return err
		}
		if err := writeSolidTree(w, t.LessThan); err != nil {
			return err
		}
		return writeSolidTree(w, t.GreaterEqual)
	}
}

// ReadSolidTree reads the output written by WriteSolidTree.
func ReadSolidTree(r io.Reader) (*SolidTree, error) {
	res, err := readSolidTree(r)
	if err != nil {
		return nil, errors.Wrap(err, "read solid tree")
	}
	return res, nil
}

func readSolidTree(r io.Reader) (*SolidTree, error) {
	var values [4]float32
	if err := binary.Read(r, binary.LittleEndian, &values); err != nil {
		return nil, err
	}
	if values[0] == 0 && values[1] == 0 && values[2] == 0 {
		return &SolidTree{
			Leaf: values[3] != 0,
		}, nil
	}
	left, err := readSolidTree(r)
	if err != nil {
		return nil, err
	}
	right, err := readSolidTree(r)
	if err != nil {
		return nil, err
	}
	return &SolidTree{
		Axis:         model3d.XYZ(float64(values[0]), float64(values[1]), float64(values[2])),
		Threshold:    float64(values[3]),
		LessThan:     left,
		GreaterEqual: right,
	}, nil
}
