let newWidth = null;
let newHeight = null;
let arFrameTimeout = setTimeout(function() {}, 0);

function handleDimensionChange(e, isWidth, isHeight) {
    if (isWidth) {
        newWidth = e.target.value * 1.0;
    }
    if (isHeight) {
        newHeight = e.target.value * 1.0;
    }

    const inImg2img = gradioApp().querySelector("#tab_img2img").style.display == "block";

    if (!inImg2img) {
        return;
    }

    let targetElement = null;

    const tabIndex = get_tab_index('mode_img2img');
    if (tabIndex === 0) {
        targetElement = gradioApp().querySelector('#img2img_image div[data-testid=image] img');
    } else if (tabIndex === 1) {
        targetElement = gradioApp().querySelector('#img2img_sketch div[data-testid=image] img');
    } else if (tabIndex === 2) {
        targetElement = gradioApp().querySelector('#img2maskimg div[data-testid=image] img');
    } else if (tabIndex === 3) {
        targetElement = gradioApp().querySelector('#inpaint_sketch div[data-testid=image] img');
    }

    if (targetElement) {
        let arPreviewRect = gradioApp().querySelector('#imageARPreview');
        if (!arPreviewRect) {
            arPreviewRect = document.createElement('div');
            arPreviewRect.id = "imageARPreview";
            gradioApp().appendChild(arPreviewRect);
        }

        const viewportOffset = targetElement.getBoundingClientRect();
        const viewportScale = Math.min(targetElement.clientWidth / targetElement.naturalWidth, targetElement.clientHeight / targetElement.naturalHeight);
        const scaledX = targetElement.naturalWidth * viewportScale;
        const scaledY = targetElement.naturalHeight * viewportScale;
        const cleintRectTop = (viewportOffset.top + window.scrollY);
        const cleintRectLeft = (viewportOffset.left + window.scrollX);
        const cleintRectCentreY = cleintRectTop + (targetElement.clientHeight / 2);
        const cleintRectCentreX = cleintRectLeft + (targetElement.clientWidth / 2);
        const arScale = Math.min(scaledX / newWidth, scaledY / newHeight);
        const arScaledX = newWidth * arScale;
        const arScaledY = newHeight * arScale;
        const arRectTop = cleintRectCentreY - (arScaledY / 2);
        const arRectLeft = cleintRectCentreX - (arScaledX / 2);
        const arRectWidth = arScaledX;
        const arRectHeight = arScaledY;

        arPreviewRect.style.top = arRectTop + 'px';
        arPreviewRect.style.left = arRectLeft + 'px';
        arPreviewRect.style.width = arRectWidth + 'px';
        arPreviewRect.style.height = arRectHeight + 'px';

        clearTimeout(arFrameTimeout);
        arFrameTimeout = setTimeout(function() {
            arPreviewRect.style.display = 'none';
        }, 2000);

        arPreviewRect.style.display = 'block';
    }
}

onAfterUiUpdate(function() {
    const arPreviewRect = gradioApp().querySelector('#imageARPreview');
    if (arPreviewRect) {
        arPreviewRect.style.display = 'none';
    }
    const tabImg2img = gradioApp().querySelector("#tab_img2img");
    if (tabImg2img) {
        const inImg2img = tabImg2img.style.display == "block";
        if (inImg2img) {
            const inputs = gradioApp().querySelectorAll('input');
            inputs.forEach(function(e) {
                const isWidth = e.parentElement.id == "img2img_width";
                const isHeight = e.parentElement.id == "img2img_height";

                if ((isWidth || isHeight) && !e.classList.contains('scrollwatch')) {
                    e.addEventListener('input', function(e) {
                        handleDimensionChange(e, isWidth, isHeight);
                    });
                    e.classList.add('scrollwatch');
                }
                if (isWidth) {
                    newWidth = e.value * 1.0;
                }
                if (isHeight) {
                    newHeight = e.value * 1.0;
                }
            });
        }
    }
});
