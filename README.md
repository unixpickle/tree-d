# tree-d

It is a known fact that 3D models can be represented as relatively small neural networks (e.g. via [NeRF](https://arxiv.org/abs/2003.08934)). But why limit the fun to just neural networks? Other function approximators are also great at overfitting a small, special-purpose dataset like a single 3D model.

To that end, this project aims to represent 3D models as oblique decision trees. In this case, an oblique decision tree is a tree of binary decisions, where each branch divides the space by a 3D plane. An oblique decision tree can be learned using greedy search techniques and improved with [Tree Alternating Optimization](https://proceedings.neurips.cc/paper/2018/file/185c29dc24325934ee377cfda20e414c-Paper.pdf), both of which are implemented from scratch in this repository.

Using decision trees to represent a 3D model has a number of potential advantages:

 1. Decision trees are great at overfitting a large dataset with a tiny model. This isn't great for real-world ML, but when we want to fit a field to a fully-known function like a 3D model, it might offer a large amount of compression.
 2. Decision trees are incredibly fast to evaluate. This could offer a speedup over mesh or neural representations for various operations.
 3. Because each decision boundary is defined by a plane, it is possible to efficiently cast a ray through a volume represented by a decision tree. In particular, casting a ray amounts to repeatedly finding the next decision boundary that will be crossed by the ray. The maximum runtime per ray is therefore bounded by the number of leaves in the tree, but will likely be much faster than that.
