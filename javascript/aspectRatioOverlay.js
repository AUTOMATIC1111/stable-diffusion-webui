let currentWidth = null;
let currentHeight = null;
let arFrameTimeout = null;
const gradio = gradioApp(); // cache gradioApp reference

// Cache AR preview div globally
let arPreviewRect = gradio.querySelector('#imageARPreview');
if (!arPreviewRect) {
    arPreviewRect = document.createElement('div');
    arPreviewRect.id = "imageARPreview";
    gradio.appendChild(arPreviewRect);
}

function dimensionChange(e, isWidth, isHeight) {
    if (isWidth) currentWidth = +e.target.value;
    if (isHeight) currentHeight = +e.target.value;

    if (gradio.querySelector("#tab_img2img").style.display !== "block") return;

    const tabIndex = get_tab_index('mode_img2img');
    const tabSelectors = [
        '#img2img_image div[data-testid=image] img',
        '#img2img_sketch div[data-testid=image] img',
        '#img2maskimg div[data-testid=image] img',
        '#inpaint_sketch div[data-testid=image] img'
    ];

    const targetElement = gradio.querySelector(tabSelectors[tabIndex]);
    if (!targetElement || !currentWidth || !currentHeight) return;

    const viewportRect = targetElement.getBoundingClientRect();
    const viewportScale = Math.min(targetElement.clientWidth / targetElement.naturalWidth,
                                   targetElement.clientHeight / targetElement.naturalHeight);

    const scaledX = targetElement.naturalWidth * viewportScale;
    const scaledY = targetElement.naturalHeight * viewportScale;

    const centreX = viewportRect.left + window.scrollX + targetElement.clientWidth / 2;
    const centreY = viewportRect.top + window.scrollY + targetElement.clientHeight / 2;

    const arScale = Math.min(scaledX / currentWidth, scaledY / currentHeight);

    const arWidth = currentWidth * arScale;
    const arHeight = currentHeight * arScale;

    arPreviewRect.style.top = (centreY - arHeight / 2) + 'px';
    arPreviewRect.style.left = (centreX - arWidth / 2) + 'px';
    arPreviewRect.style.width = arWidth + 'px';
    arPreviewRect.style.height = arHeight + 'px';
    arPreviewRect.style.display = 'block';

    if (arFrameTimeout) cancelAnimationFrame(arFrameTimeout);
    arFrameTimeout = requestAnimationFrame(() => {
        setTimeout(() => {
            arPreviewRect.style.display = 'none';
        }, 2000);
    });
}

onAfterUiUpdate(function() {
    arPreviewRect.style.display = 'none';

    const tabImg2img = gradio.querySelector("#tab_img2img");
    if (tabImg2img && tabImg2img.style.display === "block") {
        gradio.querySelectorAll('input').forEach(input => {
            const isWidth = input.parentElement.id === "img2img_width";
            const isHeight = input.parentElement.id === "img2img_height";

            if ((isWidth || isHeight) && !input.classList.contains('scrollwatch')) {
                input.addEventListener('input', e => dimensionChange(e, isWidth, isHeight));
                input.classList.add('scrollwatch');
            }

            if (isWidth) currentWidth = +input.value;
            if (isHeight) currentHeight = +input.value;
        });
    }
});
