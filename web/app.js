(function () {

    const Vector = self.treed.Vector;
    const Matrix = self.treed.Matrix;
    const Camera = self.treed.Camera;

    const MODELS = [
        {
            name: 'Corgi',
            model: 'data/corgi.bin',
            normals: 'data/corgi_normals.bin',
            source: 'https://www.thingiverse.com/thing:2806745',
        },
    ]

    class App {
        constructor() {
            this.renderer = new Renderer();
            this.renderer.onError = (e) => {
                alert(e);
            };
            this.canvas = document.getElementById('canvas');
            this.matrix = Matrix.identity();

            this.normalsCheckbox = document.getElementById('use-normals');
            this.normalsCheckbox.onchange = (_) => {
                this.rerender();
            };

            this.modelPicker = document.getElementById('model-picker');
            this.modelLink = document.getElementById('model-link');
            MODELS.forEach((model, i) => {
                const option = document.createElement('option');
                option.textContent = model.name;
                option.value = model.name;
                option.defaultSelected = i == 0;
                this.modelPicker.appendChild(option);
            });
            this.modelPicker.value = MODELS[0].name;
            this.modelPicker.onchange = (_) => {
                this.matrix = Matrix.identity();
                this.modelLink.href = this.currentModel().source;
                this.rerender();
            };
            this.modelLink.href = this.currentModel().source;

            this.rerender();
            this.setupPointerEvents();
        }

        currentModel() {
            return MODELS.filter((model) => model.name == this.modelPicker.value)[0];
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
            const model = this.currentModel();
            const options = {
                useNormals: this.normalsCheckbox.checked,
            };
            this.renderer.request(model.model, model.normals, this.camera(), options);
        }

        setupPointerEvents() {
            const groups = [
                ['mousedown', 'mousemove', 'mouseup'],
                ['touchstart', 'touchmove', 'touchend'],
            ];
            groups.forEach(([start, move, end]) => {
                this.canvas.addEventListener(start, (e) => {
                    e.preventDefault();
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
                    window.addEventListener(move, mousemove);
                    window.addEventListener(end, (_) => {
                        window.removeEventListener(move, mousemove);
                    });
                });
            });
        }
    }

    function eventPosition(e) {
        if (e['changedTouches'] && e.changedTouches.length) {
            e = e.changedTouches[0];
        }
        return new Vector(e.clientX, 0, -e.clientY);
    }

    class Renderer {
        constructor() {
            this.worker = new Worker('worker.js');
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

        request(modelPath, normalsPath, camera, options) {
            this.nextRequest = {
                modelPath: modelPath,
                normalsPath: normalsPath,
                camera: camera,
                options: options,
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
                options: this.nextRequest.options,
                canvas: this.sendCanvas,
            }, this.sendCanvas ? [this.sendCanvas] : []);
            this.handlingRequest = true;
            this.nextRequest = null;
            this.sendCanvas = null;
        }
    }

    window.app = new App();

})();
