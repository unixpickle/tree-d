(function () {

    class App {
        constructor() {
            this.renderer = new Renderer();
            this.renderer.onError = (e) => {
                alert(e);
            };
            const camera = new window.treed.Camera(
                new window.treed.Vector(0, 3, 0),
                new window.treed.Vector(1, 0, 0),
                new window.treed.Vector(0, 0, -1),
                new window.treed.Vector(0, -1, 0),
                0.69,
            );
            this.renderer.request('/data/corgi.bin', '/data/corgi_normals.bin', camera);
        }
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
