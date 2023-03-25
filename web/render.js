(function () {

    const Vector = self.treed['Vector'];

    function renderTree(canvas, camera, tree, normalMap) {
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(canvas.width, canvas.height);
        const lightDir = camera.origin.normalize().scale(-1);
        camera.pixelRays(canvas.width).forEach((ray, i) => {
            const result = tree.castRay(ray);
            imageData.data[i * 4 + 3] = 255;
            if (result !== null) {
                const normal = normalMap ? normalMap.predict(result.point) : result.normal;
                const brightness = Math.abs(lightDir.dot(normal));
                const pixel = Math.round(brightness * 255);
                imageData.data[i * 4] = pixel;
                imageData.data[i * 4 + 1] = pixel;
                imageData.data[i * 4 + 2] = pixel;
            }
        });
        ctx.putImageData(imageData, 0, 0);
    }

    self.treed['renderTree'] = renderTree;

})();