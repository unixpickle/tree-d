package treed

import (
	"testing"

	"github.com/unixpickle/model3d/model3d"
)

func TestGreedyOneLayer(t *testing.T) {
	points := []model3d.Coord3D{
		model3d.XYZ(0.3372206473036895, 0.08886795945548376, 0.10904913382284742),
		model3d.XYZ(0.049119467397132355, 0.09222594251531613, 0.4943364215240996),
		model3d.XYZ(0.14954889566362062, 0.09591537249828586, 0.4883142715040665),
		model3d.XYZ(0.7396615509684099, 0.6494343896107354, 0.04710926475574162),
		model3d.XYZ(0.18110567759776663, 0.12178645674867394, 0.1781492151666072),
		model3d.XYZ(0.4799865065965574, 0.7751502264326332, 0.13791305176614643),
		model3d.XYZ(0.5870608576745152, 0.5787210558975383, 0.2731905219519455),
		model3d.XYZ(0.6348790374801981, 0.010481100066182969, 0.9764385069150084),
		model3d.XYZ(0.9576663133369374, 0.8105906575205878, 0.7771650677885487),
		model3d.XYZ(0.5718692195395219, 0.7179289012997414, 0.49840886025758924),
	}
	labels := make([]bool, len(points))
	for i, x := range points {
		if x.X < 0.5 {
			labels[i] = x.Y > 0.4
		} else {
			labels[i] = true
		}
	}
	axes := []model3d.Coord3D{
		model3d.X(1),
		model3d.Y(1),
		model3d.Z(1),
	}
	tree := GreedyTree[float64, model3d.Coord3D, bool](
		axes,
		points,
		labels,
		EntropySplitLoss[float64]{},
		0,
		2,
	)

	for i, x := range points {
		label := labels[i]
		pred := tree.Apply(x)
		if label != pred {
			t.Errorf("point %v got %v but should be %v", x, pred, label)
		}
	}
}
