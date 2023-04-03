// attaches listeners to the txt2img and img2img galleries to update displayed generation param text when the image changes

let txt2img_gallery, img2img_gallery, modal = undefined;
onUiUpdate(function(){
	if (!txt2img_gallery) {
		txt2img_gallery = attachGalleryListeners("txt2img")
	}
	if (!img2img_gallery) {
		img2img_gallery = attachGalleryListeners("img2img")
	}
	if (!modal) {
		modal = gradioApp().getElementById('lightboxModal')
		modalObserver.observe(modal,  { attributes : true, attributeFilter : ['style'] });
	}
});

let modalObserver = new MutationObserver(function(mutations) {
	mutations.forEach(function(mutationRecord) {
		let selectedTab = gradioApp().querySelector('#tabs div button.selected')?.innerText
		if (mutationRecord.target.style.display === 'none' && (selectedTab === 'txt2img' || selectedTab === 'img2img'))
			gradioApp().getElementById(selectedTab+"_generation_info_button")?.click()
	});
});

function attachGalleryListeners(tab_name) {
	gallery = gradioApp().querySelector('#'+tab_name+'_gallery')
	gallery?.addEventListener('click', () => gradioApp().getElementById(tab_name+"_generation_info_button").click());
	gallery?.addEventListener('keydown', (e) => {
		if (e.keyCode == 37 || e.keyCode == 39) // left or right arrow
			gradioApp().getElementById(tab_name+"_generation_info_button").click()
	});
	return gallery;
}
