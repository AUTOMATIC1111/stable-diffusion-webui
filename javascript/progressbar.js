// code related to showing and updating progressbar shown as the image is being made


galleries = {}
storedGallerySelections = {}
galleryObservers = {}

function rememberGallerySelection(id_gallery){
    storedGallerySelections[id_gallery] = getGallerySelectedIndex(id_gallery)
}

function getGallerySelectedIndex(id_gallery){
    let galleryButtons = gradioApp().querySelectorAll('#'+id_gallery+' .gallery-item')
    let galleryBtnSelected = gradioApp().querySelector('#'+id_gallery+' .gallery-item.\\!ring-2')

     let currentlySelectedIndex = -1
     galleryButtons.forEach(function(v, i){ if(v==galleryBtnSelected) { currentlySelectedIndex = i } })

     return currentlySelectedIndex
}

// this is a workaround for https://github.com/gradio-app/gradio/issues/2984
function check_gallery(id_gallery){
    let gallery = gradioApp().getElementById(id_gallery)
    // if gallery has no change, no need to setting up observer again.
    if (gallery && galleries[id_gallery] !== gallery){
        galleries[id_gallery] = gallery;
        if(galleryObservers[id_gallery]){
            galleryObservers[id_gallery].disconnect();
        }

        storedGallerySelections[id_gallery] = -1

        galleryObservers[id_gallery] = new MutationObserver(function (){
            let galleryButtons = gradioApp().querySelectorAll('#'+id_gallery+' .gallery-item')
            let galleryBtnSelected = gradioApp().querySelector('#'+id_gallery+' .gallery-item.\\!ring-2')
            let currentlySelectedIndex = getGallerySelectedIndex(id_gallery)
            prevSelectedIndex = storedGallerySelections[id_gallery]
            storedGallerySelections[id_gallery] = -1

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
    check_gallery('txt2img_gallery')
    check_gallery('img2img_gallery')
})

function request(url, data, handler, errorHandler){
    var xhr = new XMLHttpRequest();
    var url = url;
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                try {
                    var js = JSON.parse(xhr.responseText);
                    handler(js)
                } catch (error) {
                    console.error(error);
                    errorHandler()
                }
            } else{
                errorHandler()
            }
        }
    };
    var js = JSON.stringify(data);
    xhr.send(js);
}

function pad2(x){
    return x<10 ? '0'+x : x
}

function formatTime(secs){
    if(secs > 3600){
        return pad2(Math.floor(secs/60/60)) + ":" + pad2(Math.floor(secs/60)%60) + ":" + pad2(Math.floor(secs)%60)
    } else if(secs > 60){
        return pad2(Math.floor(secs/60)) + ":" + pad2(Math.floor(secs)%60)
    } else{
        return Math.floor(secs) + "s"
    }
}

function setTitle(progress){
    var title = 'Stable Diffusion'

    if(opts.show_progress_in_title && progress){
        title = '[' + progress.trim() + '] ' + title;
    }

    if(document.title != title){
        document.title =  title;
    }
}


function randomId(){
    return "task(" + Math.random().toString(36).slice(2, 7) + Math.random().toString(36).slice(2, 7) + Math.random().toString(36).slice(2, 7)+")"
}

// starts sending progress requests to "/internal/progress" uri, creating progressbar above progressbarContainer element and
// preview inside gallery element. Cleans up all created stuff when the task is over and calls atEnd.
// calls onProgress every time there is a progress update
function requestProgress(id_task, progressbarContainer, gallery, atEnd, onProgress){
    var dateStart = new Date()
    var wasEverActive = false
    var parentProgressbar = progressbarContainer.parentNode
    var parentGallery = gallery ? gallery.parentNode : null

    var divProgress = document.createElement('div')
    divProgress.className='progressDiv'
    divProgress.style.display = opts.show_progressbar ? "block" : "none"
    var divInner = document.createElement('div')
    divInner.className='progress'

    divProgress.appendChild(divInner)
    parentProgressbar.insertBefore(divProgress, progressbarContainer)

    if(parentGallery){
        var livePreview = document.createElement('div')
        livePreview.className='livePreview'
        parentGallery.insertBefore(livePreview, gallery)
    }

    var removeProgressBar = function(){
        setTitle("")
        parentProgressbar.removeChild(divProgress)
        if(parentGallery) parentGallery.removeChild(livePreview)
        atEnd()
    }

    var fun = function(id_task, id_live_preview){
        request("./internal/progress", {"id_task": id_task, "id_live_preview": id_live_preview}, function(res){
            if(res.completed){
                removeProgressBar()
                return
            }

            var rect = progressbarContainer.getBoundingClientRect()

            if(rect.width){
                divProgress.style.width = rect.width + "px";
            }

            progressText = ""

            divInner.style.width = ((res.progress || 0) * 100.0) + '%'
            divInner.style.background = res.progress ? "" : "transparent"

            if(res.progress > 0){
                progressText = ((res.progress || 0) * 100.0).toFixed(0) + '%'
            }

            if(res.eta){
                progressText += " ETA: " + formatTime(res.eta)
            }


            setTitle(progressText)

            if(res.textinfo && res.textinfo.indexOf("\n") == -1){
                progressText = res.textinfo + " " + progressText
            }

            divInner.textContent = progressText

            var elapsedFromStart = (new Date() - dateStart) / 1000

            if(res.active) wasEverActive = true;

            if(! res.active && wasEverActive){
                removeProgressBar()
                return
            }

            if(elapsedFromStart > 5 && !res.queued && !res.active){
                removeProgressBar()
                return
            }


            if(res.live_preview && gallery){
                var rect = gallery.getBoundingClientRect()
                if(rect.width){
                    livePreview.style.width = rect.width + "px"
                    livePreview.style.height = rect.height + "px"
                }

                var img = new Image();
                img.onload = function() {
                    livePreview.appendChild(img)
                    if(livePreview.childElementCount > 2){
                        livePreview.removeChild(livePreview.firstElementChild)
                    }
                }
                img.src = res.live_preview;
            }


            if(onProgress){
                onProgress(res)
            }

            setTimeout(() => {
                fun(id_task, res.id_live_preview);
            }, opts.live_preview_refresh_period || 500)
        }, function(){
            removeProgressBar()
        })
    }

    fun(id_task, 0)
}
