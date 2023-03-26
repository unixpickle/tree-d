(function () {

    const Vector = self.treed.Vector;
    const Matrix = self.treed.Matrix;
    const Camera = self.treed.Camera;

    class App {
        constructor() {
            this.renderer = new Renderer();
            this.renderer.onError = (e) => {
                alert(e);
            };
            this.canvas = document.getElementById('canvas');
            this.matrix = Matrix.identity();
            this.rerender();
            this.setupPointerEvents();
        }

        camera() {
            const mat = this.matrix;
            return new Camera(
                mat.apply(new Vector(0, 3, 0)),
                mat.apply(new Vector(1, 0, 0)),
                mat.apply(new Vector(0, 0, -1)),
                mat.apply(new Vector(0, -1, 0)),
                0.69,
            );
        }

        rerender() {
            this.renderer.request('/data/corgi.bin', '/data/corgi_normals.bin', this.camera());
        }

        setupPointerEvents() {
            this.canvas.addEventListener('mousedown', (e) => {
                const p1 = eventPosition(e);
                const initMatrix = this.matrix;
                const mousemove = (e) => {
                    const p2 = eventPosition(e);
                    const offset = p2.sub(p1);
                    if (offset.norm() === 0) {
                        this.matrix = initMatrix;
                    } else {
                        const axis = offset.normalize();
                        const distance = offset.norm();
                        const transAxis = new Vector(-axis.z, 0, axis.x);
                        this.matrix = initMatrix.mul(Matrix.rotation(transAxis, distance / 100).t());
                        this.rerender();
                    }
                };
                window.addEventListener('mousemove', mousemove);
                window.addEventListener('mouseup', (_) => {
                    window.removeEventListener('mousemove', mousemove);
                });
            });
        }
    }

    function eventPosition(e) {
        return new Vector(e.clientX, 0, -e.clientY);
    }

    class Renderer {
        constructor() {
            this.worker = new Worker('/worker.js');
            this.container = document.getElementById('canvas-container');

            this.handlingRequest = false;
            this.nextRequest = null;
            this.lastModelPath = null;
            this.lastNormalsPath = null;
            this.sendCanvas = document.getElementById('canvas').transferControlToOffscreen();

            this.onError = (e) => null;

            this.worker.onmessage = (event) => {
                const d = event.data;
                if (d['error']) {
                    this.onError(d.error);
                } else {
                    this.lastModelPath = d.modelPath;
                    this.lastNormalsPath = d.normalsPath;
                    this.handlingRequest = false;
                    this._sendNext();
                }
            };
        }

        request(modelPath, normalsPath, camera) {
            this.nextRequest = {
                modelPath: modelPath,
                normalsPath: normalsPath,
                camera: camera,
            };
            if (!this.handlingRequest) {
                this._sendNext();
            }
        }

        _sendNext() {
            this.container.classList.remove('loading');
            if (this.nextRequest === null) {
                return;
            }
            if (this.nextRequest.modelPath !== this.lastModelPath ||
                this.nextRequest.normalsPath !== this.lastNormalsPath) {
                this.container.classList.add('loading');
            }
            this.worker.postMessage({
                modelPath: this.nextRequest.modelPath,
                normalsPath: this.nextRequest.normalsPath,
                camera: this.nextRequest.camera.dump(),
                canvas: this.sendCanvas,
            }, this.sendCanvas ? [this.sendCanvas] : []);
            this.handlingRequest = true;
            this.nextRequest = null;
            this.sendCanvas = null;
        }
    }

    window.app = new App();

})();
