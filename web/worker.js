importScripts(
    'types.js',
    'tree.js',
    'render.js',
);

const Camera = self.treed.Camera;
const fetchTrees = self.treed.fetchTrees;
const renderTree = self.treed.renderTree;
const renderTreeChanges = self.treed.renderTreeChanges;

let canvas = null;
let currentModel = null;
let currentNormals = null;
let currentModelPath = null;
let currentNormalsPath = null;
let currentTransform = (x) => x;

onmessage = (event) => {
    const d = event.data;
    canvas = d.canvas || canvas;
    renderModel(d.modelPath, d.normalsPath, Camera.undump(d.camera), d.options).then((_) => {
        return d.returnImage ? canvas.convertToBlob() : null;
    }).then((image) => {
        postMessage({
            modelPath: d.modelPath,
            normalsPath: d.normalsPath,
            camera: d.camera,
            options: d.options,
            image: image,
        });
    }).catch((e) => {
        postMessage({ error: e.toString() });
    });
}

async function renderModel(modelPath, normalsPath, camera, options) {
    if (modelPath !== currentModelPath) {
        const [[rawTree, min, max]] = await fetchTrees(modelPath, 'bounded');
        currentModelPath = modelPath;
        const scale = 2 / (max.sub(min).absMax());
        const translate = min.mid(max).scale(-1);
        currentTransform = (x) => x.translate(translate).scale(scale);
        currentModel = currentTransform(rawTree);
    }
    if (normalsPath !== currentNormalsPath) {
        const rawTrees = await fetchTrees(normalsPath, 'coord');
        currentNormals = rawTrees.map(currentTransform);
        currentNormalsPath = normalsPath;
    }
    if (options.maxChanges) {
        renderTreeChanges(canvas, camera, currentModel, options.maxChanges);
    } else {
        renderTree(canvas, camera, currentModel, options.useNormals ? currentNormals : null);
    }
}
