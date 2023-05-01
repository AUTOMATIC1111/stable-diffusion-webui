// A full size 'lightbox' preview modal shown when left clicking on gallery previews
/*
function closeModal() {
    gradioApp().getElementById("lightboxModal").style.display = "none";
}

function showModal(event) {
    const source = event.target || event.srcElement;
    const modalImage = gradioApp().getElementById("modalImage")
    const lb = gradioApp().getElementById("lightboxModal")
    modalImage.src = source.src
    if (modalImage.style.display === 'none') {
        lb.style.setProperty('background-image', 'url(' + source.src + ')');
    }
    lb.style.display = "flex";
    lb.focus()

    const tabTxt2Img = gradioApp().getElementById("tab_txt2img")
    const tabImg2Img = gradioApp().getElementById("tab_img2img")
    // show the save button in modal only on txt2img or img2img tabs
    if (tabTxt2Img.style.display != "none" || tabImg2Img.style.display != "none") {
        gradioApp().getElementById("modal_save").style.display = "inline"
    } else {
        gradioApp().getElementById("modal_save").style.display = "none"
    }
    event.stopPropagation()
}

function negmod(n, m) {
    return ((n % m) + m) % m;
}

function updateOnBackgroundChange() {
    const modalImage = gradioApp().getElementById("modalImage")
    if (modalImage && modalImage.offsetParent) {
        let currentButton = selected_gallery_button();

        if (currentButton?.children?.length > 0 && modalImage.src != currentButton.children[0].src) {
            modalImage.src = currentButton.children[0].src;
            if (modalImage.style.display === 'none') {
                modal.style.setProperty('background-image', `url(${modalImage.src})`)
            }
        }
    }
}

function modalImageSwitch(offset) {
    var galleryButtons = all_gallery_buttons();

    if (galleryButtons.length > 1) {
        var currentButton = selected_gallery_button();

        var result = -1
        galleryButtons.forEach(function(v, i) {
            if (v == currentButton) {
                result = i
            }
        })

        if (result != -1) {
            nextButton = galleryButtons[negmod((result + offset), galleryButtons.length)]
            nextButton.click()
            const modalImage = gradioApp().getElementById("modalImage");
            const modal = gradioApp().getElementById("lightboxModal");
            modalImage.src = nextButton.children[0].src;
            if (modalImage.style.display === 'none') {
                modal.style.setProperty('background-image', `url(${modalImage.src})`)
            }
            setTimeout(function() {
                modal.focus()
            }, 10)
        }
    }
}

function saveImage(){
    const tabTxt2Img = gradioApp().getElementById("tab_txt2img")
    const tabImg2Img = gradioApp().getElementById("tab_img2img")
    const saveTxt2Img = "save_txt2img"
    const saveImg2Img = "save_img2img"
    if (tabTxt2Img.style.display != "none") {
        gradioApp().getElementById(saveTxt2Img).click()
    } else if (tabImg2Img.style.display != "none") {
        gradioApp().getElementById(saveImg2Img).click()
    } else {
        console.error("missing implementation for saving modal of this type")
    }
}

function modalSaveImage(event) {
    saveImage()
    event.stopPropagation()
}

function modalNextImage(event) {
    modalImageSwitch(1)
    event.stopPropagation()
}

function modalPrevImage(event) {
    modalImageSwitch(-1)
    event.stopPropagation()
}

function modalKeyHandler(event) {
    switch (event.key) {
        case "s":
            saveImage()
            break;
        case "ArrowLeft":
            modalPrevImage(event)
            break;
        case "ArrowRight":
            modalNextImage(event)
            break;
        case "Escape":
            closeModal();
            break;
    }
}

function setupImageForLightbox(e) {
	if (e.dataset.modded)
		return;

	e.dataset.modded = true;
	e.style.cursor='pointer'
	e.style.userSelect='none'

	var isFirefox = navigator.userAgent.toLowerCase().indexOf('firefox') > -1

	// For Firefox, listening on click first switched to next image then shows the lightbox.
	// If you know how to fix this without switching to mousedown event, please.
	// For other browsers the event is click to make it possiblr to drag picture.
	var event = isFirefox ? 'mousedown' : 'click'

	e.addEventListener(event, function (evt) {
		if(!opts.js_modal_lightbox || evt.button != 0) return;

		modalZoomSet(gradioApp().getElementById('modalImage'), opts.js_modal_lightbox_initially_zoomed)
		evt.preventDefault()
		showModal(evt)
	}, true);

}

function modalZoomSet(modalImage, enable) {
    if (enable) {
        modalImage.classList.add('modalImageFullscreen');
    } else {
        modalImage.classList.remove('modalImageFullscreen');
    }
}

function modalZoomToggle(event) {
    modalImage = gradioApp().getElementById("modalImage");
    modalZoomSet(modalImage, !modalImage.classList.contains('modalImageFullscreen'))
    event.stopPropagation()
}

function modalTileImageToggle(event) {
    const modalImage = gradioApp().getElementById("modalImage");
    const modal = gradioApp().getElementById("lightboxModal");
    const isTiling = modalImage.style.display === 'none';
    if (isTiling) {
        modalImage.style.display = 'block';
        modal.style.setProperty('background-image', 'none')
    } else {
        modalImage.style.display = 'none';
        modal.style.setProperty('background-image', `url(${modalImage.src})`)
    }

    event.stopPropagation()
}

function galleryImageHandler(e) {
    //if (e && e.parentElement.tagName == 'BUTTON') {
        e.onclick = showGalleryImage;
    //}
}

onUiUpdate(function() {
    fullImg_preview = gradioApp().querySelectorAll('.gradio-gallery > div > img')
    if (fullImg_preview != null) {
        fullImg_preview.forEach(setupImageForLightbox);
    }
    updateOnBackgroundChange();
})

document.addEventListener("DOMContentLoaded", function() {
    //const modalFragment = document.createDocumentFragment();
    const modal = document.createElement('div')
    modal.onclick = closeModal;
    modal.id = "lightboxModal";
    modal.tabIndex = 0
    modal.addEventListener('keydown', modalKeyHandler, true)

    const modalControls = document.createElement('div')
    modalControls.className = 'modalControls gradio-container';
    modal.append(modalControls);

    const modalZoom = document.createElement('span')
    modalZoom.className = 'modalZoom cursor';
    modalZoom.innerHTML = '&#10529;'
    modalZoom.addEventListener('click', modalZoomToggle, true)
    modalZoom.title = "Toggle zoomed view";
    modalControls.appendChild(modalZoom)

    const modalTileImage = document.createElement('span')
    modalTileImage.className = 'modalTileImage cursor';
    modalTileImage.innerHTML = '&#8862;'
    modalTileImage.addEventListener('click', modalTileImageToggle, true)
    modalTileImage.title = "Preview tiling";
    modalControls.appendChild(modalTileImage)

    const modalSave = document.createElement("span")
    modalSave.className = "modalSave cursor"
    modalSave.id = "modal_save"
    modalSave.innerHTML = "&#x1F5AB;"
    modalSave.addEventListener("click", modalSaveImage, true)
    modalSave.title = "Save Image(s)"
    modalControls.appendChild(modalSave)

    const modalClose = document.createElement('span')
    modalClose.className = 'modalClose cursor';
    modalClose.innerHTML = '&times;'
    modalClose.onclick = closeModal;
    modalClose.title = "Close image viewer";
    modalControls.appendChild(modalClose)

    const modalImage = document.createElement('img')
    modalImage.id = 'modalImage';
    modalImage.onclick = closeModal;
    modalImage.tabIndex = 0
    modalImage.addEventListener('keydown', modalKeyHandler, true)
    modal.appendChild(modalImage)

    const modalPrev = document.createElement('a')
    modalPrev.className = 'modalPrev';
    modalPrev.innerHTML = '&#10094;'
    modalPrev.tabIndex = 0
    modalPrev.addEventListener('click', modalPrevImage, true);
    modalPrev.addEventListener('keydown', modalKeyHandler, true)
    modal.appendChild(modalPrev)

    const modalNext = document.createElement('a')
    modalNext.className = 'modalNext';
    modalNext.innerHTML = '&#10095;'
    modalNext.tabIndex = 0
    modalNext.addEventListener('click', modalNextImage, true);
    modalNext.addEventListener('keydown', modalKeyHandler, true)

    modal.appendChild(modalNext)

    gradioApp().appendChild(modal)


    document.body.appendChild(modal);

});
*/
//called from progressbar
function showGalleryImage() {
	//need to clean up the old code 
}

let like;
let tile;

let slide = 0;
let gallery = [];
let fullImg_src;
//let control = ["pan","undo","like","tile","page","fullscreen","autofit","zoom-in","zoom-out","clear","close","download","prev","next"];
let control = ["like","tile","page","fullscreen","autofit","zoom-in","zoom-out","clear","close","download","prev","next"];

let img_browser;
let img_file_name;

let spl_pane;
let spl_zoom_out;
let spl_zoom_in;
let spotlight_gallery;


function tile_zoom_update(val){
	let current_tile_state_size = gallery[slide].tile_size;
	current_tile_state_size += (val * 5);
	current_tile_state_size = Math.max(5, Math.min(100,current_tile_state_size));
	spl_pane.style.setProperty('background-size', current_tile_state_size+"%");
	gallery[slide].tile_size = current_tile_state_size;	
}

function tile_wheel(event){
	let delta = event["deltaY"];
	delta = (delta < 0 ? 1 : delta ? -1 : 0) * 0.5;
	tile_zoom_update(delta);
}
function tile_zoom_in(event){
	tile_zoom_update(1);
}
function tile_zoom_out(event){
	tile_zoom_update(-1);
}

function removeTile(){
	spl_pane.removeEventListener("wheel", tile_wheel);
	spl_zoom_out.removeEventListener("click", tile_zoom_out);
	spl_zoom_in.removeEventListener("click", tile_zoom_in);
	
	spl_pane.classList.remove("hide");
	spl_pane.style.setProperty('background-image', 'none');
	spotlight_gallery.zoom(0.0);
}

function addTile(spl_img){
	spl_pane.addEventListener("wheel", tile_wheel);
	spl_zoom_out.addEventListener("click", tile_zoom_out);
	spl_zoom_in.addEventListener("click", tile_zoom_in);
	
	const current_tile_state_size = gallery[slide].tile_size;
	spl_pane.classList.add("hide");	
	spl_pane.style.setProperty('background-position', "center");
	spl_pane.style.setProperty('background-size', current_tile_state_size+"%");
	if(spl_img){
		spl_pane.style.setProperty('background-image', `url(${spl_img.src})`);
	}
}

function tile_handler(event) {
	
	const current_tile_state = !gallery[slide].tile;	
    gallery[slide].tile = current_tile_state;
	
    this.classList.toggle("on");

    if(current_tile_state){	
		const spl_img = gradioApp().querySelector("#spotlight-gal .spl-pane img");
		addTile(spl_img);
    } else {			
		removeTile();	
    }
}
function like_handler(event){
 
    const current_like_state = !gallery[slide].like;
    gallery[slide].like = current_like_state;
    this.classList.toggle("on");
  
    if(current_like_state){
		// add to favorites ...
		//img_file_name.value = gallery[slide].src;
		//console.log(gallery[slide].src);
    }
    else{
      // remove from favorites ...
    }
}


function createGallerySpotlight() {

	//console.log("clicked");
	slide = 0;
	gallery = [];


	gradioApp().querySelectorAll("#"+selectedTabItemId+' .thumbnails img').forEach(function (elem, i){
		elem.setAttribute("gal-id", i);
		//if(fullImg_src == elem.src) slide = parseInt(i+1);
		if(elem.parentElement.className.indexOf("selected") != -1) slide = parseInt(i+1);
		//console.log(slide);
		gallery[i] = {
			src: elem.src,
			title: "Seed:" + elem.src,
			//description: "This is a description.",
			like: false,
			tile:false,
			tile_size: 50,	
		}
	})
	
	const options = {
		
		class: "sd-gallery",
		index: slide,
		//control: ["like","page","theme","fullscreen","autofit","zoom-in","zoom-out","close","download","play","prev","next"],
		control: control,	
		//animation: animation,
		onshow: function(index){
			
		},		
		onchange: function(index, options){
			slide = index - 1;
			tile.classList.toggle("on", gallery[slide].tile);
			//if(img_browser){
				like.classList.toggle("on", gallery[slide].like);
				
			//}
			
			spl_pane = gradioApp().querySelector("#spotlight-gal .spl-pane:nth-child("+index+")");
			spl_zoom_out = gradioApp().querySelector("#spotlight-gal .spl-zoom-out");
			spl_zoom_in = gradioApp().querySelector("#spotlight-gal .spl-zoom-in");
			
			const current_tile_state = gallery[slide].tile;
			if(current_tile_state){
				addTile();
			}else{
				removeTile();	
			}

		},		
		onclose: function(index){
			gradioApp().querySelector("#"+selectedTabItemId+' .thumbnails .thumbnail-item:nth-child('+(slide+1)+')').click();			
		}
	};

	//assign(options, modifier);
	

	spotlight_gallery.show(gallery, options);		
	spotlight_gallery.panzoom(true);
	
}

function fullImg_click_handler(e){					
	e.stopPropagation();
	e.preventDefault();
	createGallerySpotlight();
}


let intervalUiUpdateIViewer;
function onUiHeaderTabUpdate(){
	if(intervalUiUpdateIViewer != null) clearInterval(intervalUiUpdateIViewer);
	intervalUiUpdateIViewer = setInterval(onUiUpdateIViewer, 500);
}


let fullImg_preview;
function onUiUpdateIViewer(){
	clearInterval(intervalUiUpdateIViewer);
	//update_performant_inputs(selectedTabItemId);
	
	//fullImg_preview = gradioApp().querySelector('#'+selectedTabItemId+' [id$="2img_results"] .modify-upload + img.w-full.object-contain');
	fullImg_preview = gradioApp().querySelector('#'+selectedTabItemId+' .preview > img');	
	if(opts.js_modal_lightbox && fullImg_preview ) {

		fullImg_src = fullImg_preview.src;
		fullImg_preview.removeEventListener('click', fullImg_click_handler );
		fullImg_preview.addEventListener('click', fullImg_click_handler, true );//bubbling phase
		
		/*
		// this is an idea to integrate image browser extension seamlesly, 
		// without the need to change to the image browser tab extension users will be able to review images after generation
		// and add them to favorites or delete the ones that don't like on the spot
		const img_browser = gradioApp().querySelector('[id$="2img_images_history"]');
		const tbname = selectedTabItemId.split("_")[1];
		if(img_browser && tbname ==("txt2img" || "img2img")){
			const images_history = gradioApp().querySelector('[id$="'+tbname+'_images_history"]');
			const history_button_panel = images_history.querySelector('[id$="'+tbname+'_images_history_button_panel"]');
			const labels = images_history.querySelectorAll('label.block span');
			for(let i=0;i<labels.length;i++){
				//console.log(labels[i].innerHTML)
				if(labels[i].innerHTML == 'File Name'){
					img_file_name = labels[i].parentElement.querySelector("textarea");
					console.log(img_file_name.value);
					break;
				}
			}
		}
		*/
	}
}

onUiUpdate(function() {
	if(intervalUiUpdateIViewer != null) clearInterval(intervalUiUpdateIViewer);
	intervalUiUpdateIViewer = setInterval(onUiUpdateIViewer, 500);
})

onUiLoaded(function(){
	spotlight_gallery = new Spotlight();
	spotlight_gallery.init(gradioApp().querySelector('.gradio-container'), "-gal");
	tile = spotlight_gallery.addControl("tile", tile_handler);
	like = spotlight_gallery.addControl("like", like_handler);
})

document.addEventListener("DOMContentLoaded", function() {
	
});