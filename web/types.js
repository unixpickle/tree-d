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

        addT(t) {
            return new ChangePoint(this.point, this.normal, this.t + t);
        }
    }

    window.treed = window['treed'] || {};
    window.treed['Vector'] = Vector;
    window.treed['Ray'] = Vector;
    window.treed['ChangePoint'] = ChangePoint;

})();
