// code related to showing and updating progressbar shown as the image is being made
global_progressbars = {}
galleries = {}
galleryObservers = {}

// this tracks launches of window.setTimeout for progressbar to prevent starting a new timeout when the previous is still running
timeoutIds = {}

function check_progressbar(id_part, id_progressbar, id_progressbar_span, id_skip, id_interrupt, id_preview, id_gallery){
    // gradio 3.8's enlightened approach allows them to create two nested div elements inside each other with same id
    // every time you use gr.HTML(elem_id='xxx'), so we handle this here
    var progressbar = gradioApp().querySelector("#"+id_progressbar+" #"+id_progressbar)
    var progressbarParent
    if(progressbar){
        progressbarParent = gradioApp().querySelector("#"+id_progressbar)
    } else{
        progressbar = gradioApp().getElementById(id_progressbar)
        progressbarParent = null
    }

    var skip = id_skip ? gradioApp().getElementById(id_skip) : null
    var interrupt = gradioApp().getElementById(id_interrupt)

    if(opts.show_progress_in_title && progressbar && progressbar.offsetParent){
        if(progressbar.innerText){
            let newtitle = '[' + progressbar.innerText.trim() + '] Stable Diffusion';
            if(document.title != newtitle){
                document.title =  newtitle;
            }
        }else{
            let newtitle = 'Stable Diffusion'
            if(document.title != newtitle){
                document.title =  newtitle;
            }
        }
    }

	if(progressbar!= null && progressbar != global_progressbars[id_progressbar]){
	    global_progressbars[id_progressbar] = progressbar

        var mutationObserver = new MutationObserver(function(m){
            if(timeoutIds[id_part]) return;

            preview = gradioApp().getElementById(id_preview)
            gallery = gradioApp().getElementById(id_gallery)

            if(preview != null && gallery != null){
                preview.style.width = gallery.clientWidth + "px"
                preview.style.height = gallery.clientHeight + "px"
                if(progressbarParent) progressbar.style.width = progressbarParent.clientWidth + "px"

				//only watch gallery if there is a generation process going on
                check_gallery(id_gallery);

                var progressDiv = gradioApp().querySelectorAll('#' + id_progressbar_span).length > 0;
                if(progressDiv){
                    timeoutIds[id_part] = window.setTimeout(function() {
                        timeoutIds[id_part] = null
                        requestMoreProgress(id_part, id_progressbar_span, id_skip, id_interrupt)
                    }, 500)
                } else{
                    if (skip) {
                        skip.style.display = "none"
                    }
                    interrupt.style.display = "none"

                    //disconnect observer once generation finished, so user can close selected image if they want
                    if (galleryObservers[id_gallery]) {
                        galleryObservers[id_gallery].disconnect();
                        galleries[id_gallery] = null;
                    }
                }
            }

        });
        mutationObserver.observe( progressbar, { childList:true, subtree:true })
	}
}

function check_gallery(id_gallery){
    let gallery = gradioApp().getElementById(id_gallery)
    // if gallery has no change, no need to setting up observer again.
    if (gallery && galleries[id_gallery] !== gallery){
        galleries[id_gallery] = gallery;
        if(galleryObservers[id_gallery]){
            galleryObservers[id_gallery].disconnect();
        }
        let prevSelectedIndex = selected_gallery_index();
        galleryObservers[id_gallery] = new MutationObserver(function (){
            let galleryButtons = gradioApp().querySelectorAll('#'+id_gallery+' .gallery-item')
            let galleryBtnSelected = gradioApp().querySelector('#'+id_gallery+' .gallery-item.\\!ring-2')
            if (prevSelectedIndex !== -1 && galleryButtons.length>prevSelectedIndex && !galleryBtnSelected) {
                // automatically re-open previously selected index (if exists)
                activeElement = gradioApp().activeElement;
                let scrollX = window.scrollX;
                let scrollY = window.scrollY;

                galleryButtons[prevSelectedIndex].click();
                showGalleryImage();

                // When the gallery button is clicked, it gains focus and scrolls itself into view
                // We need to scroll back to the previous position
                setTimeout(function (){
                    window.scrollTo(scrollX, scrollY);
                }, 50);

                if(activeElement){
                    // i fought this for about an hour; i don't know why the focus is lost or why this helps recover it
                    // if someone has a better solution please by all means
                    setTimeout(function (){
                        activeElement.focus({
                            preventScroll: true // Refocus the element that was focused before the gallery was opened without scrolling to it
                        })
                    }, 1);
                }
            }
        })
        galleryObservers[id_gallery].observe( gallery, { childList:true, subtree:false })
    }
}

onUiUpdate(function(){
    check_progressbar('txt2img', 'txt2img_progressbar', 'txt2img_progress_span', 'txt2img_skip', 'txt2img_interrupt', 'txt2img_preview', 'txt2img_gallery')
    check_progressbar('img2img', 'img2img_progressbar', 'img2img_progress_span', 'img2img_skip', 'img2img_interrupt', 'img2img_preview', 'img2img_gallery')
    check_progressbar('ti', 'ti_progressbar', 'ti_progress_span', '', 'ti_interrupt', 'ti_preview', 'ti_gallery')
})

function requestMoreProgress(id_part, id_progressbar_span, id_skip, id_interrupt){
    btn = gradioApp().getElementById(id_part+"_check_progress");
    if(btn==null) return;

    btn.click();
    var progressDiv = gradioApp().querySelectorAll('#' + id_progressbar_span).length > 0;
    var skip = id_skip ? gradioApp().getElementById(id_skip) : null
    var interrupt = gradioApp().getElementById(id_interrupt)
    if(progressDiv && interrupt){
        if (skip) {
            skip.style.display = "block"
        }
        interrupt.style.display = "block"
    }
}

function requestProgress(id_part){
    btn = gradioApp().getElementById(id_part+"_check_progress_initial");
    if(btn==null) return;

    btn.click();
}
