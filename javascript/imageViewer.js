// A full size 'lightbox' preview modal shown when left clicking on gallery previews
let previewTimestamp = Date.now();
let previewDrag = false;
let modalPreviewZone;

function closeModal(force = false) {
  if (force) gradioApp().getElementById('lightboxModal').style.display = 'none';
  if (previewDrag) return;
  if ((Date.now() - previewTimestamp) < 250) return;
  gradioApp().getElementById('lightboxModal').style.display = 'none';
}

function modalImageSwitch(offset) {
  const galleryButtons = all_gallery_buttons();
  if (galleryButtons.length > 1) {
    const currentButton = selected_gallery_button();
    let result = -1;
    galleryButtons.forEach((v, i) => {
      if (v === currentButton) result = i;
    });
    const negmod = (n, m) => ((n % m) + m) % m;
    if (result !== -1) {
      const nextButton = galleryButtons[negmod((result + offset), galleryButtons.length)];
      nextButton.click();
      const modalImage = gradioApp().getElementById('modalImage');
      const modal = gradioApp().getElementById('lightboxModal');
      modalImage.onload = () => modalPreviewZone.focus();
      modalImage.src = nextButton.children[0].src;
      if (modalImage.style.display === 'none') modal.style.setProperty('background-image', `url(${modalImage.src})`);
    }
  }
}

function modalSaveImage(event) {
  if (gradioApp().getElementById('tab_txt2img').style.display !== 'none') gradioApp().getElementById('save_txt2img').click();
  else if (gradioApp().getElementById('tab_img2img').style.display !== 'none') gradioApp().getElementById('save_img2img').click();
  else if (gradioApp().getElementById('tab_process').style.display !== 'none') gradioApp().getElementById('save_extras').click();
}

function modalKeyHandler(event) {
  switch (event.key) {
    case 's':
      modalSaveImage();
      break;
    case 'ArrowLeft':
      modalImageSwitch(-1);
      break;
    case 'ArrowRight':
      modalImageSwitch(1);
      break;
    case 'Escape':
      closeModal(true);
      break;
  }
  event.stopPropagation();
}

function showModal(event) {
  const source = event.target || event.srcElement;
  const modalImage = gradioApp().getElementById('modalImage');
  const lb = gradioApp().getElementById('lightboxModal');
  modalImage.onload = () => modalPreviewZone.focus();
  modalImage.src = source.src;
  if (modalImage.style.display === 'none') lb.style.setProperty('background-image', `url(${source.src})`);
  lb.style.display = 'flex';
  lb.onkeydown = modalKeyHandler;
  previewTimestamp = Date.now();
  event.stopPropagation();
}

function modalDownloadImage() {
  const link = document.createElement('a');
  link.style.display = 'none';
  link.href = gradioApp().getElementById('modalImage').src;
  link.download = 'image';
  document.body.appendChild(link);
  link.click();
  setTimeout(() => {
    URL.revokeObjectURL(link.href);
    link.parentNode.removeChild(link);
  }, 0);
}

function modalZoomSet(modalImage, enable) {
  localStorage.setItem('modalZoom', enable ? 'yes' : 'no');
  if (modalImage) modalImage.classList.toggle('modalImageFullscreen', !!enable);
}

function setupImageForLightbox(e) {
  if (e.dataset.modded) return;
  e.dataset.modded = true;
  e.style.cursor = 'pointer';
  e.style.userSelect = 'none';
  e.addEventListener('mousedown', (evt) => {
    if (evt.button !== 0) return;
    const initialZoom = (localStorage.getItem('modalZoom') || true) === 'yes';
    modalZoomSet(gradioApp().getElementById('modalImage'), initialZoom);
    evt.preventDefault();
    showModal(evt);
  }, true);
}

function modalZoomToggle(event) {
  const modalImage = gradioApp().getElementById('modalImage');
  modalZoomSet(modalImage, !modalImage.classList.contains('modalImageFullscreen'));
  event.stopPropagation();
}

function modalTileToggle(event) {
  const modalImage = gradioApp().getElementById('modalImage');
  const modal = gradioApp().getElementById('lightboxModal');
  const isTiling = modalImage.style.display === 'none';
  if (isTiling) {
    modalImage.style.display = 'block';
    modal.style.setProperty('background-image', 'none');
  } else {
    modalImage.style.display = 'none';
    modal.style.setProperty('background-image', `url(${modalImage.src})`);
  }
  event.stopPropagation();
}

let imageViewerInitialized = false;

function initImageViewer() {
  const fullImgPreview = gradioApp().querySelectorAll('.gradio-gallery > div > img');
  if (fullImgPreview.length > 0) fullImgPreview.forEach(setupImageForLightbox);
  if (imageViewerInitialized) return;
  imageViewerInitialized = true;

  // main elements
  const modal = document.createElement('div');
  modal.id = 'lightboxModal';
  // modal.addEventListener('keydown', modalKeyHandler, true);

  modalPreviewZone = document.createElement('div');
  modalPreviewZone.className = 'lightboxModalPreviewZone';

  const modalImage = document.createElement('img');
  modalImage.id = 'modalImage';
  // modalImage.addEventListener('keydown', modalKeyHandler, true);
  modalPreviewZone.appendChild(modalImage);
  panzoom(modalImage, {
    zoomSpeed: 0.05, minZoom: 0.25, maxZoom: 4.0, filterKey: (/* e, dx, dy, dz */) => true,
  });

  // toolbar
  const modalZoom = document.createElement('span');
  modalZoom.id = 'modal_zoom';
  modalZoom.className = 'cursor';
  modalZoom.innerHTML = '🔍';
  modalZoom.title = 'Toggle zoomed view';
  modalZoom.addEventListener('click', modalZoomToggle, true);

  const modalTile = document.createElement('span');
  modalTile.id = 'modal_tile';
  modalTile.className = 'cursor';
  modalTile.innerHTML = '🖽';
  modalTile.title = 'Preview tiling';
  modalTile.addEventListener('click', modalTileToggle, true);

  const modalSave = document.createElement('span');
  modalSave.id = 'modal_save';
  modalSave.className = 'cursor';
  modalSave.innerHTML = '💾';
  modalSave.title = 'Save Image';
  modalSave.addEventListener('click', modalSaveImage, true);

  const modalDownload = document.createElement('span');
  modalDownload.id = 'modal_download';
  modalDownload.className = 'cursor';
  modalDownload.innerHTML = '📷';
  modalDownload.title = 'Download Image';
  modalDownload.addEventListener('click', modalDownloadImage, true);

  const modalClose = document.createElement('span');
  modalClose.id = 'modal_close';
  modalClose.className = 'cursor';
  modalClose.innerHTML = '🗙';
  modalClose.title = 'Close';
  modalClose.addEventListener('click', closeModal, true);

  // handlers
  modalPreviewZone.addEventListener('mousedown', () => { previewDrag = false; });
  modalPreviewZone.addEventListener('touchstart', () => { previewDrag = false; }, { passive: true });
  modalPreviewZone.addEventListener('mousemove', () => { previewDrag = true; });
  modalPreviewZone.addEventListener('touchmove', () => { previewDrag = true; }, { passive: true });
  modalPreviewZone.addEventListener('scroll', () => { previewDrag = true; });
  modalPreviewZone.addEventListener('mouseup', () => closeModal());
  modalPreviewZone.addEventListener('touchend', () => closeModal());

  const modalPrev = document.createElement('a');
  modalPrev.className = 'modalPrev';
  modalPrev.innerHTML = '&#10094;';
  modalPrev.addEventListener('click', () => modalImageSwitch(-1), true);
  // modalPrev.addEventListener('keydown', modalKeyHandler, true);

  const modalNext = document.createElement('a');
  modalNext.className = 'modalNext';
  modalNext.innerHTML = '&#10095;';
  modalNext.addEventListener('click', () => modalImageSwitch(1), true);
  // modalNext.addEventListener('keydown', modalKeyHandler, true);

  const modalControls = document.createElement('div');
  modalControls.className = 'modalControls gradio-container';

  // build interface
  modal.appendChild(modalPrev);
  modal.appendChild(modalPreviewZone);
  modal.appendChild(modalNext);
  modal.append(modalControls);
  modalControls.appendChild(modalZoom);
  modalControls.appendChild(modalTile);
  modalControls.appendChild(modalSave);
  modalControls.appendChild(modalDownload);
  modalControls.appendChild(modalClose);

  gradioApp().appendChild(modal);
  console.log('initImageViewer');
}

onAfterUiUpdate(initImageViewer);
