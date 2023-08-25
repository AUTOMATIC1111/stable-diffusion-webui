// A full size 'lightbox' preview modal shown when left clicking on gallery previews
function closeModal() {
    gradioApp().getElementById("lightboxModal").style.display = "none";
}

function showModal(event) {
    const source = event.target || event.srcElement;
    const modalImage = gradioApp().getElementById("modalImage");
    const lb = gradioApp().getElementById("lightboxModal");
    modalImage.src = source.src;
    if (modalImage.style.display === 'none') {
        lb.style.setProperty('background-image', 'url(' + source.src + ')');
    }
    lb.style.display = "flex";
    lb.focus();

    const tabTxt2Img = gradioApp().getElementById("tab_txt2img");
    const tabImg2Img = gradioApp().getElementById("tab_img2img");
    // show the save button in modal only on txt2img or img2img tabs
    if (tabTxt2Img.style.display != "none" || tabImg2Img.style.display != "none") {
        gradioApp().getElementById("modal_save").style.display = "inline";
    } else {
        gradioApp().getElementById("modal_save").style.display = "none";
    }
    event.stopPropagation();
}

function negmod(n, m) {
    return ((n % m) + m) % m;
}

function updateOnBackgroundChange() {
    const modalImage = gradioApp().getElementById("modalImage");
    if (modalImage && modalImage.offsetParent) {
        let currentButton = selected_gallery_button();

        if (currentButton?.children?.length > 0 && modalImage.src != currentButton.children[0].src) {
            modalImage.src = currentButton.children[0].src;
            if (modalImage.style.display === 'none') {
                const modal = gradioApp().getElementById("lightboxModal");
                modal.style.setProperty('background-image', `url(${modalImage.src})`);
            }
        }
    }
}

function modalImageSwitch(offset) {
    var galleryButtons = all_gallery_buttons();

    if (galleryButtons.length > 1) {
        var currentButton = selected_gallery_button();

        var result = -1;
        galleryButtons.forEach(function(v, i) {
            if (v == currentButton) {
                result = i;
            }
        });

        if (result != -1) {
            var nextButton = galleryButtons[negmod((result + offset), galleryButtons.length)];
            nextButton.click();
            const modalImage = gradioApp().getElementById("modalImage");
            const modal = gradioApp().getElementById("lightboxModal");
            modalImage.src = nextButton.children[0].src;
            if (modalImage.style.display === 'none') {
                modal.style.setProperty('background-image', `url(${modalImage.src})`);
            }
            setTimeout(function() {
                modal.focus();
            }, 10);
        }
    }
}

function saveImage() {
    const tabTxt2Img = gradioApp().getElementById("tab_txt2img");
    const tabImg2Img = gradioApp().getElementById("tab_img2img");
    const saveTxt2Img = "save_txt2img";
    const saveImg2Img = "save_img2img";
    if (tabTxt2Img.style.display != "none") {
        gradioApp().getElementById(saveTxt2Img).click();
    } else if (tabImg2Img.style.display != "none") {
        gradioApp().getElementById(saveImg2Img).click();
    } else {
        console.error("missing implementation for saving modal of this type");
    }
}

function modalSaveImage(event) {
    saveImage();
    event.stopPropagation();
}

function modalNextImage(event) {
    modalImageSwitch(1);
    event.stopPropagation();
}

function modalPrevImage(event) {
    modalImageSwitch(-1);
    event.stopPropagation();
}

function modalKeyHandler(event) {
    switch (event.key) {
    case "s":
        saveImage();
        break;
    case "ArrowLeft":
        modalPrevImage(event);
        break;
    case "ArrowRight":
        modalNextImage(event);
        break;
    case "Escape":
        closeModal();
        break;
    }
}

function setupImageForLightbox(e) {
    if (e.dataset.modded) {
        return;
    }

    e.dataset.modded = true;
    e.style.cursor = 'pointer';
    e.style.userSelect = 'none';

    var isFirefox = navigator.userAgent.toLowerCase().indexOf('firefox') > -1;

    // For Firefox, listening on click first switched to next image then shows the lightbox.
    // If you know how to fix this without switching to mousedown event, please.
    // For other browsers the event is click to make it possiblr to drag picture.
    var event = isFirefox ? 'mousedown' : 'click';

    e.addEventListener(event, function(evt) {
        if (evt.button == 1) {
            open(evt.target.src);
            evt.preventDefault();
            return;
        }
        if (!opts.js_modal_lightbox || evt.button != 0) return;

        modalZoomSet(gradioApp().getElementById('modalImage'), opts.js_modal_lightbox_initially_zoomed);
        evt.preventDefault();
        showModal(evt);
    }, true);

}

function modalZoomSet(modalImage, enable) {
    if (modalImage) modalImage.classList.toggle('modalImageFullscreen', !!enable);
}

function modalZoomToggle(event) {
    var modalImage = gradioApp().getElementById("modalImage");
    modalZoomSet(modalImage, !modalImage.classList.contains('modalImageFullscreen'));
    event.stopPropagation();
}

function modalTileImageToggle(event) {
    const modalImage = gradioApp().getElementById("modalImage");
    const modal = gradioApp().getElementById("lightboxModal");
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

onAfterUiUpdate(function() {
    var fullImg_preview = gradioApp().querySelectorAll('.gradio-gallery > div > img');
    if (fullImg_preview != null) {
        fullImg_preview.forEach(setupImageForLightbox);
    }
    updateOnBackgroundChange();
});

document.addEventListener("DOMContentLoaded", function() {
    //const modalFragment = document.createDocumentFragment();
    const modal = document.createElement('div');
    modal.onclick = closeModal;
    modal.id = "lightboxModal";
    modal.tabIndex = 0;
    modal.addEventListener('keydown', modalKeyHandler, true);

    const modalControls = document.createElement('div');
    modalControls.className = 'modalControls gradio-container';
    modal.append(modalControls);

    const modalZoom = document.createElement('a');
    modalZoom.className = 'modalZoom cursor';
    modalZoom.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>';
    modalZoom.addEventListener('click', modalZoomToggle, true);
    modalZoom.title = "Toggle zoomed view";
    modalControls.appendChild(modalZoom);

    const modalTileImage = document.createElement('a');
    modalTileImage.className = 'modalTileImage cursor';
    modalTileImage.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>';
    modalTileImage.addEventListener('click', modalTileImageToggle, true);
    modalTileImage.title = "Preview tiling";
    modalControls.appendChild(modalTileImage);

    const modalSave = document.createElement("a");
    modalSave.className = "modalSave cursor";
    modalSave.id = "modal_save";
    modalSave.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>';
    modalSave.addEventListener("click", modalSaveImage, true);
    modalSave.title = "Save image(s)";
    modalControls.appendChild(modalSave);

    const modalClose = document.createElement('a');
    modalClose.className = 'modalClose cursor';
    modalClose.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';
    modalClose.onclick = closeModal;
    modalClose.title = "Close image viewer";
    modalControls.appendChild(modalClose);

    const modalImage = document.createElement('img');
    modalImage.id = 'modalImage';
    modalImage.onclick = closeModal;
    modalImage.tabIndex = 0;
    modalImage.addEventListener('keydown', modalKeyHandler, true);
    modal.appendChild(modalImage);

    const modalPrev = document.createElement('a');
    modalPrev.className = 'modalPrev';
    modalPrev.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 18 9 12 15 6"/></svg>';
    modalPrev.tabIndex = 0;
    modalPrev.addEventListener('click', modalPrevImage, true);
    modalPrev.addEventListener('keydown', modalKeyHandler, true);
    modal.appendChild(modalPrev);

    const modalNext = document.createElement('a');
    modalNext.className = 'modalNext';
    modalNext.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>';
    modalNext.tabIndex = 0;
    modalNext.addEventListener('click', modalNextImage, true);
    modalNext.addEventListener('keydown', modalKeyHandler, true);

    modal.appendChild(modalNext);

    try {
        gradioApp().appendChild(modal);
    } catch (e) {
        gradioApp().body.appendChild(modal);
    }

    document.body.appendChild(modal);

});
