(function () {

    const AMBIENT = 0.4;
    const DIFFUSE = 0.5;
    const SPECULAR = 0.1;

    function renderTree(canvas, camera, tree, normalMap) {
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(canvas.width, canvas.height);
        const lightDir = camera.origin.normalize().scale(-1);
        camera.pixelRays(canvas.width).forEach((ray, i) => {
            const result = tree.castRay(ray);
            imageData.data[i * 4 + 3] = 255;
            if (result !== null) {
                const normal = normalMap ? normalMap.predict(result.point) : result.normal;
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

    self.treed['renderTree'] = renderTree;

})();