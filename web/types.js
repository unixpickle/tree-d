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

        reflect(c1) {
            const n = this.normalize();
            return c1.add(n.scale(-2 * n.dot(c1))).scale(-1);
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

        orthoBasis() {
            const absX = Math.abs(this.x);
            const absY = Math.abs(this.y);
            const absZ = Math.abs(this.z);

            // Create the first basis vector by swapping two
            // coordinates and negating one of them.
            // For numerical stability, we involve the component
            // with the largest absolute value.
            let basis1 = Vector.zero();
            if (absX > absY && absX > absZ) {
                basis1.x = this.y / absX;
                basis1.y = -this.x / absX;
            } else {
                basis1.y = this.z;
                basis1.z = -this.y;
                if (absY > absZ) {
                    basis1.y /= absY;
                    basis1.z /= absY;
                } else {
                    basis1.y /= absZ;
                    basis1.z /= absZ;
                }
            }

            // Create the second basis vector using a cross product.
            const basis2 = new Vector(
                basis1.y * this.z - basis1.z * this.y,
                basis1.z * this.x - basis1.x * this.z,
                basis1.x * this.y - basis1.y * this.x,
            ).normalize();
            basis1 = basis1.normalize();
            const basis0 = this.normalize();

            return new Matrix(
                basis0.x, basis1.x, basis2.x,
                basis0.y, basis1.y, basis2.y,
                basis0.z, basis1.z, basis2.z,
            );
        }
    }

    class Matrix {
        constructor(a, b, c, d, e, f, g, h, i) {
            this.a = a;
            this.b = b;
            this.c = c;
            this.d = d;
            this.e = e;
            this.f = f;
            this.g = g;
            this.h = h;
            this.i = i;
        }

        static identity() {
            return new Matrix(1, 0, 0, 0, 1, 0, 0, 0, 1);
        }

        static rotation(v, theta) {
            const basis = v.orthoBasis();
            const raw = new Matrix(
                1, 0, 0,
                0, Math.cos(theta), -Math.sin(theta),
                0, Math.sin(theta), Math.cos(theta),
            );
            return basis.mul(raw.mul(basis.t()));
        }

        apply(v) {
            return new Vector(
                this.a * v.x + this.b * v.y + this.c * v.z,
                this.d * v.x + this.e * v.y + this.f * v.z,
                this.g * v.x + this.h * v.y + this.i * v.z,
            );
        }

        mul(m) {
            return new Matrix(
                this.a * m.a + this.b * m.d + this.c * m.g,
                this.a * m.b + this.b * m.e + this.c * m.h,
                this.a * m.c + this.b * m.f + this.c * m.i,

                this.d * m.a + this.e * m.d + this.f * m.g,
                this.d * m.b + this.e * m.e + this.f * m.h,
                this.d * m.c + this.e * m.f + this.f * m.i,

                this.g * m.a + this.h * m.d + this.i * m.g,
                this.g * m.b + this.h * m.e + this.i * m.h,
                this.g * m.c + this.h * m.f + this.i * m.i,
            );
        }

        t() {
            return new Matrix(
                this.a, this.d, this.g,
                this.b, this.e, this.h,
                this.c, this.f, this.i,
            );
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

    self.treed = self['treed'] || {};
    self.treed['Vector'] = Vector;
    self.treed['Matrix'] = Matrix;
    self.treed['Ray'] = Ray;
    self.treed['ChangePoint'] = ChangePoint;
    self.treed['Camera'] = Camera;

})();
