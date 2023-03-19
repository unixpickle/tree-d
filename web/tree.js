(function () {

    class Vector {
        constructor(x, y, z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        static zero() {
            return new Vector(0, 0, 0);
        }

        static axis(idx) {
            if (idx === 0) {
                return new Vector(1, 0, 0);
            } else if (idx === 1) {
                return new Vector(0, 1, 0);
            } else if (idx === 2) {
                return new Vector(0, 0, 1);
            }
            // Avoid exception for optimization.
            return new Vector(0, 0, 0);
        }

        dot(v1) {
            return this.x * v1.x + this.y * v1.y + this.z * v1.z;
        }

        norm() {
            return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
        }

        scale(s) {
            return new Vector(this.x * s, this.y * s, this.z * s);
        }

        add(v) {
            return new Vector(this.x + v.x, this.y + v.y, this.z + v.z);
        }

        getAxis(axis) {
            if (axis === 0) {
                return this.x;
            } else if (axis === 1) {
                return this.y;
            } else if (axis === 2) {
                return this.z;
            }
            // Avoid exception for optimization.
            return 0;
        }
    }

    class Ray {
        constructor(origin, direction) {
            this.origin = origin;
            this.direction = direction;
        }

        at(t) {
            return this.origin.add(this.direction.scale(t));
        }
    }

    class ChangePoint {
        constructor(point, normal, t) {
            this.point = point;
            this.normal = normal;
            this.t = t;
        }
    }

    class Tree {
        constructor(axis, threshold, left, right, leaf) {
            this.axis = axis;
            this.threshold = threshold;
            this.left = left;
            this.right = right;
            this.leaf = leaf;
        }

        static newLeaf(value) {
            return new Tree(null, null, null, null, value);
        }

        isLeaf() {
            return this.axis !== null;
        }

        nextChange(ray) {
            if (this.isLeaf()) {
                return null;
            }
            const dirDot = ray.direction.dot(this.axis);
            const absDirDot = Math.abs(dirDot);
            if (absDirDot < this.axis.norm() * ray.direction.norm() * 1e-8) {
                return null;
            }

            const curDot = this.axis.dot(ray.direction);
            const child = curDot >= this.threshold ? this.right : this.left;
            const normal = child === this.right ? this.axis : this.axis.scale(-1);
            const thisT = (this.threshold - curDot) / dirDot;

            // This edge case might seem extremely unusual, but it actually occurs
            // naturally for trees with tight bounding boxes.
            if (this.threshold == curDot) {
                const maxT = 1e8;
                const maxDot = this.axis.dot(ray.at(maxT));
                if ((curDot >= this.threshold) != (maxDot >= this.threshold)) {
                    const changeT = this._changeT(ray, thisT, maxT);
                    return new ChangePoint(ray.at(changeT), normal, changeT);
                }
            }

            const childRes = child.nextChange(ray);
            if (thisT <= 0) {
                return childRes;
            } else if (childRes !== null && thisT > childRes.t) {
                return childRes;
            } else {
                const changeT = this._changeT(origin, direction, thisT, thisT * 2);
                return new ChangePoint(ray.at(changeT), normal, changeT);
            }
        }

        _changeT(ray, minT, maxT) {
            const orig = this.axis.dot(ray.origin) < this.threshold;
            const x = ray.at(minT);
            if (this.axis.dot(x) < this.threshold != orig) {
                return minT;
            }
            if (this.axis.dot(ray.at(maxT)) < this.threshold == orig) {
                console.error("impossible situation encountered: collision was expected");
                return maxT;
            }
            for (let i = 0; i < 32; ++i) {
                const midT = (minT + maxT) / 2;
                if (this.axis.dot(ray.at(midT)) < this.threshold != orig) {
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
        return tree;
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
        return new Tree(new Vector(x, y, z), threshold, left, right, null);
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
});
