const Vector = self.treed['Vector'];

(function () {

    const AMBIENT = 0.3;
    const DIFFUSE = 0.5;
    const SPECULAR = 0.2;

    function renderTree(canvas, camera, tree, normalMap) {
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(canvas.width, canvas.height);
        const lightDir = camera.origin.normalize().scale(-1);
        camera.pixelRays(canvas.width).forEach((ray, i) => {
            const result = tree.castRay(ray);
            imageData.data[i * 4 + 3] = 255;
            if (result !== null) {
                const normal = normalMap === null ? result.normal : normalMap.reduce(
                    (acc, cur) => acc.add(cur.predict(result.point)),
                    Vector.zero(),
                );
                const diffuse = Math.abs(lightDir.dot(normal));
                const refDot = Math.abs(normal.reflect(ray.direction).dot(lightDir));
                const specular = Math.pow(refDot, 10);
                const brightness = AMBIENT + DIFFUSE * diffuse + SPECULAR * specular;
                const pixel = Math.round(Math.pow(brightness, 2.2) * 255);
                imageData.data[i * 4] = pixel;
                imageData.data[i * 4 + 1] = pixel;
                imageData.data[i * 4 + 2] = pixel;
            }
        });
        ctx.putImageData(imageData, 0, 0);
    }

    function renderTreeChanges(canvas, camera, tree, maxChanges) {
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(canvas.width, canvas.height);
        camera.pixelRays(canvas.width).forEach((ray, i) => {
            const [_, changes] = tree.castRayChanges(ray);
            imageData.data[i * 4 + 3] = 255;
            const brightness = Math.min(1.0, changes / maxChanges);
            imageData.data[i * 4] = Math.floor(brightness * 255.999);
            imageData.data[i * 4 + 1] = Math.floor((1 - brightness) * 255.49);
            imageData.data[i * 4 + 2] = 128;
        });
        ctx.putImageData(imageData, 0, 0);
    }

    self.treed['renderTree'] = renderTree;
    self.treed['renderTreeChanges'] = renderTreeChanges;

})();