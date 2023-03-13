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
        let allcurrentButtons = gradioApp().querySelectorAll(".gallery-item.transition-all.\\!ring-2")
        let currentButton = null
        allcurrentButtons.forEach(function(elem) {
            if (elem.parentElement.offsetParent) {
                currentButton = elem;
            }
        })

        if (currentButton?.children?.length > 0 && modalImage.src != currentButton.children[0].src) {
            modalImage.src = currentButton.children[0].src;
            if (modalImage.style.display === 'none') {
                modal.style.setProperty('background-image', `url(${modalImage.src})`)
            }
        }
    }
}

function modalImageSwitch(offset) {
    var allgalleryButtons = gradioApp().querySelectorAll(".gallery-item.transition-all")
    var galleryButtons = []
    allgalleryButtons.forEach(function(elem) {
        if (elem.parentElement.offsetParent) {
            galleryButtons.push(elem);
        }
    })

    if (galleryButtons.length > 1) {
        var allcurrentButtons = gradioApp().querySelectorAll(".gallery-item.transition-all.\\!ring-2")
        var currentButton = null
        allcurrentButtons.forEach(function(elem) {
            if (elem.parentElement.offsetParent) {
                currentButton = elem;
            }
        })

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

function showGalleryImage() {
    setTimeout(function() {
        fullImg_preview = gradioApp().querySelectorAll('img.w-full.object-contain')

        if (fullImg_preview != null) {
            fullImg_preview.forEach(function function_name(e) {
                if (e.dataset.modded)
                    return;
                e.dataset.modded = true;
                if(e && e.parentElement.tagName == 'DIV'){
                    e.style.cursor='pointer'
                    e.style.userSelect='none'

                    var isFirefox = isFirefox = navigator.userAgent.toLowerCase().indexOf('firefox') > -1

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
            });
        }

    }, 100);
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
    if (e && e.parentElement.tagName == 'BUTTON') {
        e.onclick = showGalleryImage;
    }
}

onUiUpdate(function() {
    fullImg_preview = gradioApp().querySelectorAll('img.w-full')
    if (fullImg_preview != null) {
        fullImg_preview.forEach(galleryImageHandler);
    }
    updateOnBackgroundChange();
})

document.addEventListener("DOMContentLoaded", function() {
    const modalFragment = document.createDocumentFragment();
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


    gradioApp().getRootNode().appendChild(modal)

    document.body.appendChild(modalFragment);

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

function tile_handler(event) {
	
	const current_tile_state = !gallery[slide].tile;
    gallery[slide].tile = current_tile_state;
    this.classList.toggle("on");
    
    const spl_img = document.querySelector("#spotlight .spl-pane img");
	const spl_pane = spl_img.parentElement;
	
    if(current_tile_state){ 
		spl_pane.classList.add("hide");
        spl_pane.style.setProperty('background-image', `url(${spl_img.src})`);
		spl_pane.style.setProperty('background-position', "center");		
    } else {		
		spl_pane.classList.remove("hide");
        spl_pane.style.setProperty('background-image', 'none');
		Spotlight.autofit(true);
    }
}

function like_handler(event){
 
    const current_like_state = !gallery[slide].like;
    gallery[slide].like = current_like_state;
    this.classList.toggle("on");
  
    if(current_like_state){
      // add to favorites ...
    }
    else{
      // remove from favorites ...
    }
}
function createGallerySpotlight(src) {
	
	slide = 0;
	gallery = [];
	
	gradioApp().querySelectorAll('.grid img.w-full.object-contain').forEach(function (elem, i){
		elem.setAttribute("gal-id", i);
		if(src == elem.src) slide = (i+1);
		gallery[i] = {
			src: elem.src,
			like: false,
			tile:false
		}
	})
	
	const options = {
		
		class: "sd-gallery",
		index: slide,
		//control: ["like","page","theme","fullscreen","autofit","zoom-in","zoom-out","close","download","play","prev","next"],
		control: ["tile","like","page","fullscreen","autofit","zoom-in","zoom-out","close","download","prev","next"],
		//animation: animation,
		//onshow: function(index){
			//like = Spotlight.addControl("like", handler);
		//},		
		onchange: function(index, options){
			slide = index - 1;
			like.classList.toggle("on", gallery[slide].like);
			tile.classList.toggle("on", gallery[slide].tile);
		},		
		onclose: function(index){
			//Spotlight.removeControl("like");
			gradioApp().querySelector('.grid .gallery-item:nth-child('+(slide+1)+')').click();		
		}
	};
	
	//assign(options, modifier);
	Spotlight.show(gallery, options);
}


function fullImg_click_handler(e){					
	e.stopPropagation();
	e.preventDefault();
	createGallerySpotlight(fullImg_src);
}

onUiUpdate(function() {
	
	const fullImg_preview = gradioApp().querySelector('.modify-upload + img.w-full.object-contain');	
	if(opts.js_modal_lightbox && fullImg_preview ) {
		//console.log("GALLERY UPDATED");
		fullImg_src = fullImg_preview.src;
		fullImg_preview.removeEventListener('click', fullImg_click_handler );
		fullImg_preview.addEventListener('click', fullImg_click_handler, true );//bubbling phase	
	}
})

document.addEventListener("DOMContentLoaded", function() {
	
	const head = document.head;
	const link = document.createElement("link");

	link.type = "text/css";
	link.rel = "stylesheet";
	link.href = "file=html/spotlight.css";
	head.appendChild(link);
	
	Spotlight.init();
	like = Spotlight.addControl("like", like_handler);
	tile = Spotlight.addControl("tile", tile_handler);
	
});