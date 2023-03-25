importScripts(
    '/types.js',
    '/tree.js',
    '/render.js',
);

const Camera = window.treed.Camera;
const fetchTree = window.treed.fetchTree;
const renderTree = window.treed.renderTree;

let canvas = null;
let currentModel = null;
let currentNormals = null;
let currentModelPath = null;
let currentNormalsPath = null;

function onmessage(event) {
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
        currentModel = await fetchTree(modelPath, 'bounded');
        currentModelPath = modelPath;
    }
    if (normalsPath !== currentNormalsPath) {
        currentNormals = await fetchTree(modelPath, 'coord');
        currentNormalsPath = normalsPath;
    }
    renderTree(canvas, camera, currentModel, currentNormals);
}
