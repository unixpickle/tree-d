(function () {

    const Vector = self.treed['Vector'];
    const Ray = self.treed['Ray'];
    const ChangePoint = self.treed['ChangePoint'];

    class Tree {
        constructor(axis, threshold, left, right, leaf) {
            this.axis = axis;
            this.threshold = threshold;
            this.left = left;
            this.right = right;
            this.leaf = leaf;

            this._axisNorm = axis ? axis.norm() : 0;
        }

        static newLeaf(value) {
            return new Tree(null, null, null, null, value);
        }

        predict(x) {
            if (this.isLeaf()) {
                return this.leaf;
            }
            if (this.axis.dot(x) < this.threshold) {
                return this.left.predict(x);
            } else {
                return this.right.predict(x);
            }
        }

        isLeaf() {
            return this.axis === null;
        }

        scale(s) {
            if (this.isLeaf()) {
                return this;
            } else {
                return new Tree(
                    this.axis,
                    this.threshold * s,
                    this.left.scale(s),
                    this.right.scale(s),
                    null,
                )
            }
        }

        translate(t) {
            if (this.isLeaf()) {
                return this;
            } else {
                return new Tree(
                    this.axis,
                    this.threshold + this.axis.dot(t),
                    this.left.translate(t),
                    this.right.translate(t),
                    null,
                )
            }
        }

        castRay(ray) {
            const value = this.predict(ray.origin);
            let prevT = 0;
            while (true) {
                const internalChange = this._nextChange(ray);
                if (internalChange === null) {
                    return null;
                }
                const change = internalChange.changePoint(ray);
                const newValue = this.predict(change.point);
                if (newValue !== value) {
                    return change.addT(prevT);
                }
                prevT += change.t;
                ray = new Ray(change.point, ray.direction);
            }
        }

        _nextChange(ray) {
            if (this.isLeaf()) {
                return null;
            }
            const dirDot = ray.direction.dot(this.axis);

            const curDot = this.axis.dot(ray.origin);
            const child = curDot >= this.threshold ? this.right : this.left;
            const normalScale = child === this.right ? 1 : -1;
            const thisT = (this.threshold - curDot) / dirDot;

            // This edge case might seem extremely unusual, but it actually occurs
            // naturally for trees with tight bounding boxes.
            if (this.threshold == curDot) {
                const maxT = 1e8;
                const maxDot = this.axis.dot(ray.at(maxT));
                if ((curDot >= this.threshold) != (maxDot >= this.threshold)) {
                    return new InternalChangePoint(this, normalScale, thisT, maxT);
                }
            }

            const childRes = child._nextChange(ray);
            if (thisT <= 0 || Math.abs(dirDot) < this._axisNorm * 1e-8) {
                return childRes;
            } else if (childRes !== null && thisT > childRes.t) {
                return childRes;
            } else {
                return new InternalChangePoint(this, normalScale, thisT, Math.max(thisT * 2, 1e-4));
            }
        }
    }

    class InternalChangePoint {
        constructor(branch, normalScale, t, maxT) {
            this.branch = branch;
            this.normalScale = normalScale;
            this.t = t;
            this.maxT = maxT;
        }

        changePoint(ray) {
            const t = this._changeT(ray, this.t, this.maxT);
            const normal = this.branch.axis.normalize().scale(this.normalScale);
            return new ChangePoint(ray.at(t), normal, t);
        }

        _changeT(ray, minT, maxT) {
            const orig = this.branch.axis.dot(ray.origin) < this.branch.threshold;
            const x = ray.at(minT);
            if (this.branch.axis.dot(x) < this.branch.threshold != orig) {
                return minT;
            }
            if (this.branch.axis.dot(ray.at(maxT)) < this.branch.threshold == orig) {
                throw Error("impossible situation encountered: collision was expected");
            }
            for (let i = 0; i < 32; ++i) {
                const midT = (minT + maxT) / 2;
                if (this.branch.axis.dot(ray.at(midT)) < this.branch.threshold != orig) {
                    maxT = midT;
                } else {
                    minT = midT;
                }
            }
            return maxT;
        }
    }

    class FloatReader {
        constructor(bytes) {
            this.arr = new Float32Array(flipToLittleEndian(bytes));
            this.offset = 0;
        }

        done() {
            return this.offset >= this.arr.length;
        }

        next() {
            if (this.done()) {
                throw Error('out of bounds read');
            }
            return this.arr[this.offset++];
        }

        nextVector() {
            const x = this.next();
            const y = this.next();
            const z = this.next();
            return new Vector(x, y, z);
        }
    }

    async function fetchTree(url, treeType) {
        let readFn;
        if (treeType === 'bool') {
            readFn = readBoolTree;
        } else if (treeType === 'coord') {
            readFn = readCoordTree;
        } else if (treeType === 'bounded') {
            readFn = readBoundedSolidTree;
        } else {
            throw Error('unsupported tree type: ' + treeType);
        }
        const buf = await (await fetch(url)).arrayBuffer();
        return readFn(new FloatReader(buf));
    }

    function readBoolTree(floatReader) {
        return readTree(floatReader, (f) => f.next() !== 0);
    }

    function readCoordTree(floatReader) {
        return readTree(floatReader, (f) => f.nextVector());
    }

    function readBoundedSolidTree(floatReader) {
        const min = floatReader.nextVector();
        const max = floatReader.nextVector();
        let tree = readBoolTree(floatReader);

        // Apply bounds as branches of the tree.
        for (let axis = 0; axis < 3; ++axis) {
            const ax = Vector.axis(axis);
            tree = new Tree(ax, min.getAxis(axis), Tree.newLeaf(false), tree);
            tree = new Tree(ax, max.getAxis(axis), tree, Tree.newLeaf(false));
        }
        return [tree, min, max];
    }

    function readTree(floatReader, leafFn) {
        const axis = floatReader.nextVector();
        if (axis.x === 0 && axis.y === 0 && axis.z === 0) {
            const leaf = leafFn(floatReader);
            return new Tree(null, null, null, null, leaf);
        }
        const threshold = floatReader.next();
        const left = readTree(floatReader, leafFn);
        const right = readTree(floatReader, leafFn);
        return new Tree(axis, threshold, left, right, null);
    }

    function flipToLittleEndian(input) {
        if (!isBigEndian()) {
            return input;
        }
        let arr = new Uint8Array(input);
        const output = new ArrayBuffer(arr.length);
        const out = new Uint8Array(output);
        for (let i = 0; i < arr.length; i += 4) {
            const w = arr[i];
            const x = arr[i + 1];
            const y = arr[i + 2];
            const z = arr[i + 3];
            out[i] = z;
            out[i + 1] = y;
            out[i + 2] = x;
            out[i + 3] = w;
        }
        return output;
    }

    function isBigEndian() {
        const x = new ArrayBuffer(4);
        new Float32Array(x)[0] = 1;
        return new Uint8Array(x)[0] != 0;
    }

    self.treed['Tree'] = Tree;
    self.treed['fetchTree'] = fetchTree;

})();
