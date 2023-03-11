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
	return writeCoordBranchTree(w, t, func(w io.Writer, leaf bool) error {
		var x float32
		if leaf {
			x = 1
		}
		return binary.Write(w, binary.LittleEndian, x)
	})
}

// WriteCoordTree serialize t in a 32-bit precision binary format.
func WriteCoordTree(w io.Writer, t *CoordTree) error {
	err := writeCoordBranchTree(w, t, func(w io.Writer, leaf model3d.Coord3D) error {
		return binary.Write(w, binary.LittleEndian, []float32{
			float32(leaf.X),
			float32(leaf.Y),
			float32(leaf.Z),
		})
	})
	if err != nil {
		err = errors.Wrap(err, "write coord tree")
	}
	return err
}

func writeCoordBranchTree[T any](
	w io.Writer,
	t *Tree[float64, model3d.Coord3D, T],
	f func(io.Writer, T) error,
) error {
	if t.IsLeaf() {
		if err := binary.Write(w, binary.LittleEndian, []float32{0, 0, 0}); err != nil {
			return err
		}
		return f(w, t.Leaf)
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
		if err := writeCoordBranchTree(w, t.LessThan, f); err != nil {
			return err
		}
		return writeCoordBranchTree(w, t.GreaterEqual, f)
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
	return readCoordBranchTree(r, func(r io.Reader) (bool, error) {
		var x float32
		if err := binary.Read(r, binary.LittleEndian, &x); err != nil {
			return false, err
		}
		return x != 0, nil
	})
}

func ReadCoordTree(r io.Reader) (*CoordTree, error) {
	res, err := readCoordBranchTree(r, func(r io.Reader) (model3d.Coord3D, error) {
		var x [3]float32
		if err := binary.Read(r, binary.LittleEndian, &x); err != nil {
			return model3d.Coord3D{}, err
		}
		return model3d.XYZ(float64(x[0]), float64(x[1]), float64(x[2])), nil
	})
	if err != nil {
		return nil, errors.Wrap(err, "read coord tree")
	}
	return res, nil
}

func readCoordBranchTree[T any](
	r io.Reader,
	f func(io.Reader) (T, error),
) (*Tree[float64, model3d.Coord3D, T], error) {
	var values [3]float32
	if err := binary.Read(r, binary.LittleEndian, &values); err != nil {
		return nil, err
	}
	if values[0] == 0 && values[1] == 0 && values[2] == 0 {
		leaf, err := f(r)
		if err != nil {
			return nil, err
		}
		return &Tree[float64, model3d.Coord3D, T]{
			Leaf: leaf,
		}, nil
	}
	var threshold float32
	if err := binary.Read(r, binary.LittleEndian, &threshold); err != nil {
		return nil, err
	}

	left, err := readCoordBranchTree(r, f)
	if err != nil {
		return nil, err
	}
	right, err := readCoordBranchTree(r, f)
	if err != nil {
		return nil, err
	}

	return &Tree[float64, model3d.Coord3D, T]{
		Axis:         model3d.XYZ(float64(values[0]), float64(values[1]), float64(values[2])),
		Threshold:    float64(threshold),
		LessThan:     left,
		GreaterEqual: right,
	}, nil
}
