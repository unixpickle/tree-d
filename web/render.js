(function () {

    const Vector = window.treed['Vector'];

    function renderTree(canvas, camera, tree) {
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(canvas.width, canvas.height);
        const lightDir = camera.origin.normalize().scale(-1);
        camera.pixelRays(canvas.width).forEach((ray, i) => {
            const result = tree.castRay(ray);
            imageData.data[i * 4 + 3] = 255;
            if (result !== null) {
                const brightness = Math.abs(lightDir.dot(result.normal));
                const pixel = Math.round(brightness * 255);
                imageData.data[i * 4] = pixel;
                imageData.data[i * 4 + 1] = pixel;
                imageData.data[i * 4 + 2] = pixel;
            }
        });
        ctx.putImageData(imageData, 0, 0);
    }

    window.treed['renderTree'] = renderTree;

})();