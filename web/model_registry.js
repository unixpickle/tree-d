const Matrix = self.treed.Matrix;
const Vector = self.treed.Vector;

self.treed.models = [
    {
        name: 'Moai',
        path: 'data/moai',
        source: 'https://www.thingiverse.com/thing:2493386',
        initMatrix: Matrix.rotation(new Vector(0, 0, 1), Math.PI * 1.15).mul(Matrix.rotation(new Vector(1, 0, 0), -0.4)),
        metadata: { "normals": { "num_leaves": 26865, "filename": "normals.bin", "file_size": 1074584 }, "lods": [{ "num_leaves": 61100, "filename": "full.bin", "file_size": 1955208 }, { "num_leaves": 4096, "filename": "lod_4096.bin", "file_size": 131080 }, { "num_leaves": 2048, "filename": "lod_2048.bin", "file_size": 65544 }, { "num_leaves": 1024, "filename": "lod_1024.bin", "file_size": 32776 }, { "num_leaves": 512, "filename": "lod_512.bin", "file_size": 16392 }, { "num_leaves": 256, "filename": "lod_256.bin", "file_size": 8200 }] },
    },
    {
        name: 'Corgi',
        path: 'data/corgi',
        source: 'https://www.thingiverse.com/thing:2806745',
        initMatrix: Matrix.rotation(new Vector(0, 0, 1), Math.PI * 0.8).mul(Matrix.rotation(new Vector(1, 0, 0), -0.3)),
        metadata: { "normals": { "num_leaves": 16927, "filename": "normals.bin", "file_size": 677064 }, "lods": [{ "num_leaves": 2029, "filename": "full.bin", "file_size": 64936 }, { "num_leaves": 1024, "filename": "lod_1024.bin", "file_size": 32776 }, { "num_leaves": 511, "filename": "lod_511.bin", "file_size": 16360 }, { "num_leaves": 255, "filename": "lod_255.bin", "file_size": 8168 }] },
    },
    {
        name: 'Curvy Thing',
        path: 'data/curvy_thing',
        source: 'https://github.com/unixpickle/model3d/tree/18bdf6c73a91e699501b0c1b441388d1a4350c19/examples/decoration/curvy_thing',
        initMatrix: Matrix.rotation(new Vector(0, 0, 1), Math.PI * 0.2).mul(Matrix.rotation(new Vector(1, 0, 0), -0.4)),
        metadata: { "normals": { "num_leaves": 11473, "filename": "normals.bin", "file_size": 458904 }, "lods": [{ "num_leaves": 39101, "filename": "full.bin", "file_size": 1251240 }, { "num_leaves": 4096, "filename": "lod_4096.bin", "file_size": 131080 }, { "num_leaves": 2048, "filename": "lod_2048.bin", "file_size": 65544 }, { "num_leaves": 1023, "filename": "lod_1023.bin", "file_size": 32744 }, { "num_leaves": 512, "filename": "lod_512.bin", "file_size": 16392 }, { "num_leaves": 255, "filename": "lod_255.bin", "file_size": 8168 }] },
    },
];
