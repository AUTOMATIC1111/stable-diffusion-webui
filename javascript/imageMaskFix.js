/**
 * temporary fix for https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/668
 * @see https://github.com/gradio-app/gradio/issues/1721
 */
function imageMaskResize() {
    const canvases = gradioApp().querySelectorAll('#img2maskimg .touch-none canvas');
    if (!canvases.length) {
        window.removeEventListener('resize', imageMaskResize);
        return;
    }

    const wrapper = canvases[0].closest('.touch-none');
    const previewImage = wrapper.previousElementSibling;

    if (!previewImage.complete) {
        previewImage.addEventListener('load', imageMaskResize);
        return;
    }

    const w = previewImage.width;
    const h = previewImage.height;
    const nw = previewImage.naturalWidth;
    const nh = previewImage.naturalHeight;
    const portrait = nh > nw;

    const wW = Math.min(w, portrait ? h / nh * nw : w / nw * nw);
    const wH = Math.min(h, portrait ? h / nh * nh : w / nw * nh);

    wrapper.style.width = `${wW}px`;
    wrapper.style.height = `${wH}px`;
    wrapper.style.left = `0px`;
    wrapper.style.top = `0px`;

    canvases.forEach(c => {
        c.style.width = c.style.height = '';
        c.style.maxWidth = '100%';
        c.style.maxHeight = '100%';
        c.style.objectFit = 'contain';
    });
}

onUiUpdate(imageMaskResize);
window.addEventListener('resize', imageMaskResize);
