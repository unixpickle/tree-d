const Matrix = self.treed.Matrix;
const Vector = self.treed.Vector;

self.treed.models = [
    {
        name: 'Moai',
        path: 'data/moai',
        source: 'https://www.thingiverse.com/thing:2493386',
        initMatrix: Matrix.rotation(new Vector(0, 0, 1), Math.PI),
        metadata: { "normals": { "num_leaves": 10933, "filename": "normals.bin", "file_size": 437304 }, "lods": [{ "num_leaves": 5805, "filename": "full.bin", "file_size": 185768 }, { "num_leaves": 4096, "filename": "lod_4096.bin", "file_size": 131080 }, { "num_leaves": 2047, "filename": "lod_2047.bin", "file_size": 65512 }, { "num_leaves": 1024, "filename": "lod_1024.bin", "file_size": 32776 }, { "num_leaves": 512, "filename": "lod_512.bin", "file_size": 16392 }, { "num_leaves": 256, "filename": "lod_256.bin", "file_size": 8200 }] },
    },
    {
        name: 'Corgi',
        path: 'data/corgi',
        source: 'https://www.thingiverse.com/thing:2806745',
        initMatrix: Matrix.rotation(new Vector(0, 0, 1), Math.PI / 2),
        metadata: { "normals": { "num_leaves": 6739, "filename": "normals.bin", "file_size": 269544 }, "lods": [{ "num_leaves": 5130, "filename": "full.bin", "file_size": 164168 }, { "num_leaves": 4095, "filename": "lod_4095.bin", "file_size": 131048 }, { "num_leaves": 2048, "filename": "lod_2048.bin", "file_size": 65544 }, { "num_leaves": 1023, "filename": "lod_1023.bin", "file_size": 32744 }, { "num_leaves": 512, "filename": "lod_512.bin", "file_size": 16392 }, { "num_leaves": 256, "filename": "lod_256.bin", "file_size": 8200 }] },
    },
    {
        name: 'Curvy Thing',
        path: 'data/curvy_thing',
        source: 'https://github.com/unixpickle/model3d/tree/18bdf6c73a91e699501b0c1b441388d1a4350c19/examples/decoration/curvy_thing',
        initMatrix: Matrix.identity(),
        metadata: { "normals": { "num_leaves": 3672, "filename": "normals.bin", "file_size": 146864 }, "lods": [{ "num_leaves": 2011, "filename": "full.bin", "file_size": 64360 }, { "num_leaves": 1024, "filename": "lod_1024.bin", "file_size": 32776 }, { "num_leaves": 512, "filename": "lod_512.bin", "file_size": 16392 }, { "num_leaves": 255, "filename": "lod_255.bin", "file_size": 8168 }] },
    },
];
