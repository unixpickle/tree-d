(function () {

    const Vector = self.treed.Vector;
    const Matrix = self.treed.Matrix;
    const Camera = self.treed.Camera;
    const MODELS = self.treed.models;

    const DL_RENDER_WIDTH = 1024;
    const DL_RENDER_HEIGHT = 1024;

    class App {
        constructor() {
            this.renderer = new UIRenderer();
            this.renderer.onError = (e) => {
                alert(e);
            };
            this.downloadRenderer = new DownloadRenderer();
            this.downloadRenderer.onError = (e) => {
                alert(e);
            }

            this.canvas = document.getElementById('canvas');

            this.normalsCheckbox = document.getElementById('use-normals');
            this.normalsCheckbox.onchange = (_) => this.rerender();

            this.rayHeatmapCheckbox = document.getElementById('ray-heatmap');
            this.rayHeatmapCheckbox.onchange = (_) => this.rerender();
            this.rayHeatmapMax = document.getElementById('heatmap-max');
            this.rayHeatmapMax.onchange = (_) => this.rerender();

            this.downloadButton = document.getElementById('download-button');
            this.downloadButton.addEventListener('click', () => this.download());

            this.modelPicker = document.getElementById('model-picker');
            this.lodPicker = document.getElementById('lod-picker');
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
                this.populateModelInfo();
                this.matrix = this.initMatrix();
                this.rerender();
            };
            this.lodPicker.onchange = (_) => this.rerender();
            this.populateModelInfo();

            this.rerender();
            this.setupPointerEvents();
        }

        populateModelInfo() {
            const model = this.currentModel();
            this.matrix = this.initMatrix();
            this.modelLink.href = model.source;
            this.lodPicker.innerHTML = '';
            model.metadata.lods.forEach((lod, i) => {
                const option = document.createElement('option');
                if (i === 0) {
                    option.innerText = 'Full (' + lod['num_leaves'] + ' leaves)';
                } else {
                    option.innerText = lod['num_leaves'] + ' leaves';
                }
                option.value = model.path + '/' + lod.filename;
                this.lodPicker.appendChild(option);
                if (i === 0) {
                    this.lodPicker.value = option.value;
                }
            });
        }

        currentModel() {
            return MODELS.filter((model) => model.name == this.modelPicker.value)[0];
        }

        currentLodPath() {
            return this.lodPicker.value;
        }

        initMatrix() {
            return this.currentModel().initMatrix;
        }

        camera() {
            return this.baseCamera().apply(this.matrix);
        }

        baseCamera() {
            const initialPos = new Vector(0, 4, 0);
            const z = initialPos.scale(-1).normalize();
            const y = new Vector(0, 0, -1).projectOut(z).normalize();
            const x = z.normalize().cross(y);
            return new Camera(
                initialPos,
                x,
                y,
                z,
                0.69,
            );
        }

        rerender() {
            this._requestRender(this.renderer);
        }

        download() {
            this._requestRender(this.downloadRenderer);
        }

        _requestRender(renderer) {
            const model = this.currentModel();
            const options = {
                useNormals: this.normalsCheckbox.checked,
                maxChanges: this.rayHeatmapCheckbox.checked ? this.rayHeatmapMax.value : null,
            };
            const normalsPath = model.path + '/' + model.metadata.normals.filename;
            renderer.request(this.currentLodPath(), normalsPath, this.camera(), options);
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

    class RendererBase {
        constructor(canvas) {
            this.worker = new Worker('worker.js');

            this.handlingRequest = false;
            this.nextRequest = null;
            this.lastModelPath = null;
            this.lastNormalsPath = null;
            this.sendCanvas = canvas.transferControlToOffscreen();

            this.onError = (e) => null;

            this.worker.onmessage = (event) => {
                const d = event.data;
                if (d['error']) {
                    this.onError(d.error);
                } else {
                    this.handleResult(d);
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
            if (this.nextRequest === null) {
                this.stopLoading();
                return;
            }

            if (this.nextRequest.modelPath !== this.lastModelPath ||
                this.nextRequest.normalsPath !== this.lastNormalsPath) {
                this.startLoading();
            } else {
                this.stopLoading();
            }

            this.worker.postMessage({
                modelPath: this.nextRequest.modelPath,
                normalsPath: this.nextRequest.normalsPath,
                camera: this.nextRequest.camera.dump(),
                options: this.nextRequest.options,
                returnImage: this.returnImage(),
                canvas: this.sendCanvas,
            }, this.sendCanvas ? [this.sendCanvas] : []);
            this.handlingRequest = true;
            this.nextRequest = null;
            this.sendCanvas = null;
        }

        returnImage() {
            return false;
        }

        handleResult(d) {
            this.lastModelPath = d.modelPath;
            this.lastNormalsPath = d.normalsPath;
            this.handlingRequest = false;
            this._sendNext();
        }

        startLoading() {
        }

        stopLoading() {
        }
    }

    class UIRenderer extends RendererBase {
        constructor() {
            super(document.getElementById('canvas'));
            this.container = document.getElementById('canvas-container');
        }

        startLoading() {
            this.container.classList.add('loading');
        }

        stopLoading() {
            this.container.classList.remove('loading');
        }
    }

    class DownloadRenderer extends RendererBase {
        constructor() {
            const canvas = document.createElement('canvas');
            canvas.width = DL_RENDER_WIDTH;
            canvas.height = DL_RENDER_HEIGHT;
            super(canvas);
        }

        returnImage() {
            return true;
        }

        handleResult(d) {
            super.handleResult(d);
            downloadBlob(d.image, 'rendered_tree.png');
        }
    }

    function downloadBlob(blob, filename) {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        document.body.appendChild(a);
        a.style.display = 'none';
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }

    window.app = new App();

})();
