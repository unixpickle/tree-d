# tree-d

It is a known fact that 3D models can be represented as relatively small neural networks (e.g. via [NeRF](https://arxiv.org/abs/2003.08934)). But why limit the fun to just neural networks? Other function approximators are also great at overfitting a small, special-purpose dataset like a single 3D model.

To that end, this project aims to represent 3D models as oblique decision trees. In this case, I use the decision tree as an occupancy function, where the output is *true* for points inside an object and *false* otherwise. I use oblique decision trees, where each branch divides the space across a 3D plane. An oblique decision tree can be learned using greedy search techniques and improved with [Tree Alternating Optimization](https://proceedings.neurips.cc/paper/2018/file/185c29dc24325934ee377cfda20e414c-Paper.pdf), both of which are implemented from scratch in this repository.

Using decision trees to represent a 3D model has a number of potential advantages:

 1. **Compact representations:** Decision trees are great at overfitting a large dataset with a tiny model. This isn't great for real-world ML, but when we want to fit a field to a fully-known function like a 3D model, it might offer a large amount of compression.
 2. **Fast rendering:** Because each decision boundary is defined by a plane, it is possible to efficiently cast a ray through a volume represented by a decision tree. In particular, casting a ray amounts to repeatedly finding the next decision boundary that will be crossed by the ray. The maximum runtime per ray is therefore bounded by the number of leaves in the tree, but is typically much faster than that.
 3. **Fast occupancy evaluation:** Decision trees are incredibly fast to evaluate. This could offer a speedup over mesh or neural representations for various operations, the most obvious of which being occupancy evaluations.

# Usage

## Building a tree

To build a tree from a known 3D triangle mesh, you should first save the mesh as an STL file. Then you can run the following command to create a decision tree representing the occupancy function of the model:

```bash
go run cmds/mesh_to_tree/*.go \
    input.stl \
    occupancy_tree.bin
```

This may take a while to run. To build a smaller tree for testing purposes, you can pass `-depth 14` (default is 20). You can also try a different algorithm for creating the tree using a slightly different command:

```bash
go run cmds/mesh_to_tree_v2/*.go \
    input.stl \
    occupancy_tree.bin
```

This version uses a simpler active learning approach based on polytope sampling. As a result, it may create larger initial trees that can benefit more from simplification. One downside is that it sometimes results in visible undesirable artifacts, such as long, thin slivers that are not meant to be contained in the occupancy function.

## Building a normal map

To build a normal map, you can run:

```bash
go run cmds/mesh_to_normal_map/*.go \
    -dataset-size 2000000 \
    -dataset-epsilon 0.01 \
    occupancy_tree.bin \
    input.stl \
    normal_tree.bin
```

The `-dataset-size` argument controls how large the training set of points is. You can reduce this for faster but less accurate results. The `-dataset-epsilon` argument can be tuned to make the normal map more or less robust to noise. This can be helpful if you later plan to simplify the tree.

## Rendering and exporting

You can render a tree with its normal map into a GIF file like so:

```bash
go run cmds/render_tree/*.go \
    -normal-map normal_tree.bin \
    occupancy_tree.bin \
    output.gif
```

If you omit the `-normal-map <path.bin>` argument, the tree will be rendered with inferred normals.

To export the tree with a number of different levels-of-detail, with accompanying metadata to be used in the web demo, you can run:

```bash
go run cmds/prepare_for_web/*.go \
    -mesh input.stl \
    -model occupancy_tree.bin \
    -normals normal_tree.bin \
    -output export_dir
```
