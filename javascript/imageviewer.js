// A full size 'lightbox' preview modal shown when left clicking on gallery previews

function closeModal() {
  gradioApp().getElementById("lightboxModal").style.display = "none";
}

function showModal(event) {
  var source = event.target || event.srcElement;
  gradioApp().getElementById("modalImage").src = source.src
  var lb = gradioApp().getElementById("lightboxModal")
  lb.style.display = "block";
  lb.focus()
  event.stopPropagation()
}

function negmod(n, m) {
  return ((n % m) + m) % m;
}

function modalImageSwitch(offset){
  var galleryButtons = gradioApp().querySelectorAll(".gallery-item.transition-all")

  if(galleryButtons.length>1){
      var currentButton  = gradioApp().querySelector(".gallery-item.transition-all.\\!ring-2")

      var result = -1
      galleryButtons.forEach(function(v, i){ if(v==currentButton) { result = i } })

      if(result != -1){
        nextButton = galleryButtons[negmod((result+offset),galleryButtons.length)]
        nextButton.click()
        gradioApp().getElementById("modalImage").src = nextButton.children[0].src
        setTimeout( function(){gradioApp().getElementById("lightboxModal").focus()},10)
      }
  }
}

function modalNextImage(event){
  modalImageSwitch(1)
  event.stopPropagation()
}

function modalPrevImage(event){
  modalImageSwitch(-1)  
  event.stopPropagation()
}

function modalKeyHandler(event){
    switch (event.key) {
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

function showGalleryImage(){
    setTimeout(function() {
        fullImg_preview = gradioApp().querySelectorAll('img.w-full.object-contain')
        
        if(fullImg_preview != null){
            fullImg_preview.forEach(function function_name(e) {
                if(e && e.parentElement.tagName == 'DIV'){

                    e.style.cursor='pointer'

                    e.addEventListener('click', function (evt) {
                        if(!opts.js_modal_lightbox) return;
                        showModal(evt)
                    },true);
                }
            });
        }

    }, 100);
}

function modalZoomToggle(event){
    modalImage = gradioApp().getElementById("modalImage");
    if( !modalImage.classList.contains('modalImageFullscreen') ){
        modalImage.classList.add('modalImageFullscreen');
    }else{
        modalImage.classList.remove('modalImageFullscreen');
    }
    event.stopPropagation()
}

function galleryImageHandler(e){
    if(e && e.parentElement.tagName == 'BUTTON'){
        e.onclick = showGalleryImage;
    }
}

onUiUpdate(function(){
	fullImg_preview = gradioApp().querySelectorAll('img.w-full')
	    if(fullImg_preview != null){
		fullImg_preview.forEach(galleryImageHandler);
	}

    if(Object.keys(opts).length != 0) return;

    json_elem = gradioApp().getElementById('settings_json')
    if(json_elem == null) return;

    textarea = json_elem.querySelector('textarea')
    jsdata = textarea.value
    opts = JSON.parse(jsdata)

    if(!window.lightbox_settings_applied){
        window.lightbox_settings_applied = true;
        if(opts.js_modal_lightbox_initialy_zoomed){
            var imageModal = gradioApp().getElementById('modalImage');
            if(imageModal){
                imageModal.className = 'modalImageFullscreen';
            }
        }
    }
})

document.addEventListener("DOMContentLoaded", function() {
    const modalFragment = document.createDocumentFragment();
    const modal = document.createElement('div')
    modal.onclick = closeModal;
    modal.id = "lightboxModal";
    modal.tabIndex=0
    modal.addEventListener('keydown', modalKeyHandler, true)

    const modalClose = document.createElement('span')
    modalClose.className = 'modalClose cursor';
    modalClose.innerHTML = '&times;'
    modalClose.onclick = closeModal;
    modal.appendChild(modalClose)

    const modalZoom = document.createElement('span')
    modalZoom.className = 'modalZoom cursor';
    modalZoom.innerHTML = '&#10529;'
    modalZoom.addEventListener('click', modalZoomToggle, true)
    modal.appendChild(modalZoom)

    const modalImage = document.createElement('img')
    modalImage.id = 'modalImage';
    modalImage.onclick = closeModal;
    modalImage.tabIndex=0
    modalImage.addEventListener('keydown', modalKeyHandler, true)
    modal.appendChild(modalImage)

    const modalPrev = document.createElement('a')
    modalPrev.className = 'modalPrev';
    modalPrev.innerHTML = '&#10094;'
    modalPrev.tabIndex=0
    modalPrev.addEventListener('click',modalPrevImage,true);
    modalPrev.addEventListener('keydown', modalKeyHandler, true)
    modal.appendChild(modalPrev)

    const modalNext = document.createElement('a')
    modalNext.className = 'modalNext';
    modalNext.innerHTML = '&#10095;'
    modalNext.tabIndex=0
    modalNext.addEventListener('click',modalNextImage,true);
    modalNext.addEventListener('keydown', modalKeyHandler, true)

    modal.appendChild(modalNext)


    gradioApp().getRootNode().appendChild(modal)
    
    document.body.appendChild(modalFragment);
	
});
