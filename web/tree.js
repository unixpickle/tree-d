(function () {

    class Vector {
        constructor(x, y, z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        dot(v1) {
            return this.x * v1.x + this.y * v1.y + this.z * v1.z;
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
