/* global gradioApp, onUiUpdate, get_tab_index */

let currentWidth = null;
let currentHeight = null;
let arFrameTimeout = setTimeout(() => {}, 0);

function dimensionChange(e, is_width, is_height) {
  if (is_width) currentWidth = e.target.value * 1.0;
  if (is_height) currentHeight = e.target.value * 1.0;
  const inImg2img = gradioApp().querySelector('#tab_img2img').style.display === 'block';
  if (!inImg2img) return;
  let targetElement = null;
  const tabIndex = get_tab_index('mode_img2img');
  if (tabIndex === 0) targetElement = gradioApp().querySelector('#img2img_image div[data-testid=image] img'); // img2img
  else if (tabIndex === 1) targetElement = gradioApp().querySelector('#img2img_sketch div[data-testid=image] img'); // Sketch
  else if (tabIndex === 2) targetElement = gradioApp().querySelector('#img2maskimg div[data-testid=image] img'); // Inpaint
  else if (tabIndex === 3) targetElement = gradioApp().querySelector('#inpaint_sketch div[data-testid=image] img'); // Inpaint sketch

  if (targetElement) {
    let arPreviewRect = gradioApp().querySelector('#imageARPreview');
    if (!arPreviewRect) {
      arPreviewRect = document.createElement('div');
      arPreviewRect.id = 'imageARPreview';
      gradioApp().appendChild(arPreviewRect);
    }

    const viewportOffset = targetElement.getBoundingClientRect();
    const viewportscale = Math.min(targetElement.clientWidth / targetElement.naturalWidth, targetElement.clientHeight / targetElement.naturalHeight);
    const scaledx = targetElement.naturalWidth * viewportscale;
    const scaledy = targetElement.naturalHeight * viewportscale;
    const cleintRectTop = (viewportOffset.top + window.scrollY);
    const cleintRectLeft = (viewportOffset.left + window.scrollX);
    const cleintRectCentreY = cleintRectTop + (targetElement.clientHeight / 2);
    const cleintRectCentreX = cleintRectLeft + (targetElement.clientWidth / 2);
    const arscale = Math.min(scaledx / currentWidth, scaledy / currentHeight);
    const arscaledx = currentWidth * arscale;
    const arscaledy = currentHeight * arscale;
    const arRectTop = cleintRectCentreY - (arscaledy / 2);
    const arRectLeft = cleintRectCentreX - (arscaledx / 2);
    const arRectWidth = arscaledx;
    const arRectHeight = arscaledy;
    arPreviewRect.style.top = `${arRectTop}px`;
    arPreviewRect.style.left = `${arRectLeft}px`;
    arPreviewRect.style.width = `${arRectWidth}px`;
    arPreviewRect.style.height = `${arRectHeight}px`;

    clearTimeout(arFrameTimeout);
    arFrameTimeout = setTimeout(() => {
      arPreviewRect.style.display = 'none';
    }, 2000);
    arPreviewRect.style.display = 'block';
  }
}

onUiUpdate(() => {
  const arPreviewRect = gradioApp().querySelector('#imageARPreview');
  if (arPreviewRect) arPreviewRect.style.display = 'none';
  const tabImg2img = gradioApp().querySelector('#tab_img2img');
  if (tabImg2img) {
    const inImg2img = tabImg2img.style.display === 'block';
    if (inImg2img) {
      const inputs = gradioApp().querySelectorAll('input');
      inputs.forEach((e) => {
        const is_width = e.parentElement.id === 'img2img_width';
        const is_height = e.parentElement.id === 'img2img_height';
        if ((is_width || is_height) && !e.classList.contains('scrollwatch')) {
          e.addEventListener('input', (evt) => { dimensionChange(evt, is_width, is_height); });
          e.classList.add('scrollwatch');
        }
        if (is_width) currentWidth = e.value * 1.0;
        if (is_height) currentHeight = e.value * 1.0;
      });
    }
  }
});
