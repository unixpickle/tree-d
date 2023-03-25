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

        static fromArray(x) {
            return new Vector(x[0], x[1], x[2]);
        }

        toArray() {
            return [this.x, this.y, this.z];
        }

        dot(v1) {
            return this.x * v1.x + this.y * v1.y + this.z * v1.z;
        }

        norm() {
            return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
        }

        normalize() {
            return this.scale(1 / this.norm());
        }

        scale(s) {
            return new Vector(this.x * s, this.y * s, this.z * s);
        }

        add(v) {
            return new Vector(this.x + v.x, this.y + v.y, this.z + v.z);
        }

        sub(v) {
            return this.add(v.scale(-1));
        }

        mid(v) {
            return this.add(v).scale(0.5);
        }

        absMax() {
            return Math.max(Math.abs(this.x), Math.abs(this.y), Math.abs(this.z));
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

        addT(t) {
            return new ChangePoint(this.point, this.normal, this.t + t);
        }
    }

    class Camera {
        constructor(origin, x, y, z, fov) {
            this.origin = origin;
            this.x = x;
            this.y = y;
            this.z = z;
            this.fov = fov;
        }

        static undump(obj) {
            return new Camera(
                Vector.fromArray(obj.origin),
                Vector.fromArray(obj.x),
                Vector.fromArray(obj.y),
                Vector.fromArray(obj.z),
                obj.fov,
            );
        }

        dump() {
            return {
                origin: this.origin.toArray(),
                x: this.x.toArray(),
                y: this.y.toArray(),
                z: this.z.toArray(),
                fov: this.fov,
            };
        }

        pixelRays(size) {
            const z = this.z.scale(1 / Math.tan(this.fov / 2));
            const result = [];
            for (let y = 0; y < size; ++y) {
                const yFrac = 2 * y / size - 1;
                for (let x = 0; x < size; ++x) {
                    const xFrac = 2 * x / size - 1;
                    const dir = this.x.scale(xFrac).add(this.y.scale(yFrac)).add(z).normalize();
                    result.push(new Ray(this.origin, dir));
                }
            }
            return result;
        }
    }

    window.treed = window['treed'] || {};
    window.treed['Vector'] = Vector;
    window.treed['Ray'] = Ray;
    window.treed['ChangePoint'] = ChangePoint;
    window.treed['Camera'] = Camera;

})();
