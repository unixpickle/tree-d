package treed

import (
	"bytes"
	"reflect"
	"testing"

	"github.com/unixpickle/model3d/model3d"
)

func TestReadWriteBoundedSolidTree(t *testing.T) {
	// Note: all values in this tree are equivalent in float32 and float64.
	tree := &BoundedSolidTree{
		Min: model3d.XYZ(-0.5, 0.75, 0.0),
		Max: model3d.XYZ(2.0, 3.0, 4.0),
		Tree: &SolidTree{
			Axis:      model3d.XYZ(0.5, 0.25, -0.125),
			Threshold: 1.0,
			LessThan: &SolidTree{
				Axis:         model3d.XYZ(4.0, 5.0, 6.0),
				Threshold:    10.0,
				LessThan:     &SolidTree{Leaf: true},
				GreaterEqual: &SolidTree{Leaf: false},
			},
			GreaterEqual: &SolidTree{Leaf: true},
		},
	}
	var b bytes.Buffer
	if err := WriteBoundedSolidTree(&b, tree); err != nil {
		t.Fatal(err)
	}
	if result, err := ReadBoundedSolidTree(&b); err != nil {
		t.Fatal(err)
	} else {
		if !reflect.DeepEqual(result, tree) {
			t.Fatalf("%v != %v", tree, result)
		}
	}
}
