importScripts(
    '/types.js',
    '/tree.js',
    '/render.js',
);

const Camera = self.treed.Camera;
const fetchTree = self.treed.fetchTree;
const renderTree = self.treed.renderTree;

let canvas = null;
let currentModel = null;
let currentNormals = null;
let currentModelPath = null;
let currentNormalsPath = null;
let currentTransform = (x) => x;

onmessage = (event) => {
    const d = event.data;
    canvas = d.canvas || canvas;
    renderModel(d.modelPath, d.normalsPath, Camera.undump(d.camera)).then((_) => {
        postMessage({ modelPath: d.modelPath, normalsPath: d.normalsPath, camera: d.camera });
    }).catch((e) => {
        postMessage({ error: e.toString() });
    });
}

async function renderModel(modelPath, normalsPath, camera) {
    if (modelPath !== currentModelPath) {
        const [rawTree, min, max] = await fetchTree(modelPath, 'bounded');
        currentModelPath = modelPath;
        const scale = 2 / (max.sub(min).absMax());
        const translate = min.mid(max).scale(-1);
        currentTransform = (x) => x.translate(translate).scale(scale);
        currentModel = currentTransform(rawTree);
    }
    if (normalsPath !== currentNormalsPath) {
        const rawTree = await fetchTree(normalsPath, 'coord');
        currentNormals = currentTransform(rawTree);
        currentNormalsPath = normalsPath;
    }
    renderTree(canvas, camera, currentModel, currentNormals);
}
