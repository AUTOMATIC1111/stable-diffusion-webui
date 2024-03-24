let currentWidth;
let currentHeight;
let arFrameTimeout;

function dimensionChange(e, is_width, is_height) {
    if (is_width) {
        currentWidth = e.target.value * 1.0;
    }
    if (is_height) {
        currentHeight = e.target.value * 1.0;
    }

    var inImg2img = gradioApp().querySelector("#tab_img2img").style.display == "block";

    if (!inImg2img) {
        return;
    }

    var targetElement = null;

    var tabIndex = get_tab_index('mode_img2img');
    if (tabIndex == 0) { // img2img
        targetElement = gradioApp().querySelector('#img2img_image div[data-testid=image] canvas');
    } else if (tabIndex == 1) { //Sketch
        targetElement = gradioApp().querySelector('#img2img_sketch div[data-testid=image] canvas');
    } else if (tabIndex == 2) { // Inpaint
        targetElement = gradioApp().querySelector('#img2maskimg div[data-testid=image] canvas');
    } else if (tabIndex == 3) { // Inpaint sketch
        targetElement = gradioApp().querySelector('#inpaint_sketch div[data-testid=image] canvas');
    } else if (tabIndex == 4) { // Inpaint upload
        targetElement = gradioApp().querySelector('#img_inpaint_base div[data-testid=image] img');
    }

    if (targetElement) {
        var arPreviewRect = gradioApp().querySelector('#imageARPreview');
        if (!arPreviewRect) {
            arPreviewRect = document.createElement('div');
            arPreviewRect.id = "imageARPreview";
            gradioApp().appendChild(arPreviewRect);
        }

        var viewportOffset = targetElement.getBoundingClientRect();
        var viewportscale = Math.min(targetElement.clientWidth / targetElement.width, targetElement.clientHeight / targetElement.height);

        var scaledx = targetElement.width * viewportscale;
        var scaledy = targetElement.height * viewportscale;

        var clientRectTop = (viewportOffset.top + window.scrollY);
        var clientRectLeft = (viewportOffset.left + window.scrollX);
        var clientRectCentreY = clientRectTop + (targetElement.clientHeight / 2);
        var clientRectCentreX = clientRectLeft + (targetElement.clientWidth / 2);

        var arscale = Math.min(scaledx / currentWidth, scaledy / currentHeight);
        var arscaledx = currentWidth * arscale;
        var arscaledy = currentHeight * arscale;

        var arRectTop = clientRectCentreY - (arscaledy / 2);
        var arRectLeft = clientRectCentreX - (arscaledx / 2);
        var arRectWidth = arscaledx;
        var arRectHeight = arscaledy;

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
    var arPreviewRect = gradioApp().querySelector('#imageARPreview');
    if (arPreviewRect) {
        arPreviewRect.style.display = 'none';
    }

    var tabImg2img = gradioApp().querySelector("#tab_img2img");
    if (tabImg2img) {
        if (tabImg2img.style.display == "block") {
            let inputs = gradioApp().querySelectorAll('input');
            inputs.forEach(function(e) {
                var is_width = e.parentElement.id == "img2img_width";
                var is_height = e.parentElement.id == "img2img_height";

                if ((is_width || is_height) && !e.classList.contains('scrollwatch')) {
                    e.addEventListener('input', function(e) {
                        dimensionChange(e, is_width, is_height);
                    });
                    e.classList.add('scrollwatch');
                }
                if (is_width) {
                    currentWidth = e.value * 1.0;
                }
                if (is_height) {
                    currentHeight = e.value * 1.0;
                }
            });
        }
    }
});
