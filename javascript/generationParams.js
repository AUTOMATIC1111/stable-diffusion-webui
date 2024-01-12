// attaches listeners to the txt2img and img2img galleries to update displayed generation param text when the image changes

function attachGalleryListeners(tab_name) {
  const gallery = gradioApp().querySelector(`#${tab_name}_gallery`);
  if (!gallery) return null;
  gallery.addEventListener('click', () => setTimeout(() => {
    log('galleryItemSelected:', tab_name);
    gradioApp().getElementById(`${tab_name}_generation_info_button`)?.click();
  }, 500));
  gallery?.addEventListener('keydown', (e) => {
    if (e.keyCode === 37 || e.keyCode === 39) gradioApp().getElementById(`${tab_name}_generation_info_button`).click(); // left or right arrow
  });
  return gallery;
}

let txt2img_gallery;
let img2img_gallery;
let modal;
let generationParamsInitialized = false;

async function initiGenerationParams() {
  if (generationParamsInitialized) return;
  if (!modal) modal = gradioApp().getElementById('lightboxModal');
  if (!modal) return;

  const modalObserver = new MutationObserver((mutations) => {
    mutations.forEach((mutationRecord) => {
      let selectedTab = gradioApp().querySelector('#tabs div button.selected')?.innerText;
      if (!selectedTab) selectedTab = gradioApp().querySelector('#tabs div button')?.innerText;
      if (mutationRecord.target.style.display === 'none' && (selectedTab === 'txt2img' || selectedTab === 'img2img')) { gradioApp().getElementById(`${selectedTab}_generation_info_button`)?.click(); }
    });
  });

  if (!txt2img_gallery) txt2img_gallery = attachGalleryListeners('txt2img');
  if (!img2img_gallery) img2img_gallery = attachGalleryListeners('img2img');
  if (txt2img_gallery && img2img_gallery) generationParamsInitialized = true;
  else return;
  modalObserver.observe(modal, { attributes: true, attributeFilter: ['style'] });
  log('initGenerationParams');
}

onAfterUiUpdate(initiGenerationParams);
