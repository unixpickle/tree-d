const Matrix = self.treed.Matrix;
const Vector = self.treed.Vector;

self.treed.models = [
    // {
    //     name: 'Moai',
    //     source: 'https://www.thingiverse.com/thing:2493386',
    //     initMatrix: Matrix.rotation(new Vector(0, 0, 1), Math.PI),
    // },
    // {
    //     name: 'Corgi',
    //     source: 'https://www.thingiverse.com/thing:2806745',
    //     initMatrix: Matrix.rotation(new Vector(0, 0, 1), Math.PI / 2),
    // },
    {
        name: 'Curvy Thing',
        path: 'data/curvy_thing',
        source: 'https://github.com/unixpickle/model3d/tree/18bdf6c73a91e699501b0c1b441388d1a4350c19/examples/decoration/curvy_thing',
        initMatrix: Matrix.identity(),
        metadata: { "normals": { "num_leaves": 3672, "filename": "normals.bin", "file_size": 146864 }, "lods": [{ "num_leaves": 2011, "filename": "full.bin", "file_size": 64360 }, { "num_leaves": 1024, "filename": "lod_1024.bin", "file_size": 32776 }, { "num_leaves": 512, "filename": "lod_512.bin", "file_size": 16392 }, { "num_leaves": 255, "filename": "lod_255.bin", "file_size": 8168 }] },
    },
];
