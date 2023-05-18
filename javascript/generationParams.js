/* global gradioApp, onUiUpdate */
// attaches listeners to the txt2img and img2img galleries to update displayed generation param text when the image changes

function attachGalleryListeners(tab_name) {
  const gallery = gradioApp().querySelector(`#${tab_name}_gallery`);
  gallery?.addEventListener('click', () => setTimeout(() => {
    gradioApp()
      .getElementById(`${tab_name}_generation_info_button`)
      ?.click();
  }, 500));
  gallery?.addEventListener('keydown', (e) => {
    if (e.keyCode == 37 || e.keyCode == 39) { // left or right arrow
      gradioApp()
        .getElementById(`${tab_name}_generation_info_button`)
        .click();
    }
  });
  return gallery;
}

let txt2img_gallery;
let img2img_gallery;
let modal;

onUiUpdate(() => {
  if (!txt2img_gallery) txt2img_gallery = attachGalleryListeners('txt2img');
  if (!img2img_gallery) img2img_gallery = attachGalleryListeners('img2img');
  if (!modal) {
    modal = gradioApp().getElementById('lightboxModal');
    modalObserver.observe(modal, { attributes: true, attributeFilter: ['style'] });
  }
});

let modalObserver = new MutationObserver((mutations) => {
  mutations.forEach((mutationRecord) => {
    let selectedTab = gradioApp().querySelector('#tabs div button.selected')?.innerText;
    if (!selectedTab) selectedTab = gradioApp().querySelector('#tabs div button')?.innerText;
    if (mutationRecord.target.style.display === 'none' && (selectedTab === 'txt2img' || selectedTab === 'img2img')) { gradioApp().getElementById(`${selectedTab}_generation_info_button`)?.click(); }
  });
});

