// code related to showing and updating progressbar shown as the image is being made
global_progressbars = {}

function check_progressbar(id_part, id_progressbar, id_progressbar_span, id_interrupt, id_preview, id_gallery){
    var progressbar = gradioApp().getElementById(id_progressbar)
    var interrupt = gradioApp().getElementById(id_interrupt)
	if(progressbar!= null && progressbar != global_progressbars[id_progressbar]){
	    global_progressbars[id_progressbar] = progressbar

        var mutationObserver = new MutationObserver(function(m){
            preview = gradioApp().getElementById(id_preview)
            gallery = gradioApp().getElementById(id_gallery)

            if(preview != null && gallery != null){
                preview.style.width = gallery.clientWidth + "px"
                preview.style.height = gallery.clientHeight + "px"

                var progressDiv = gradioApp().querySelectorAll('#' + id_progressbar_span).length > 0;
                if(!progressDiv){
                    interrupt.style.display = "none"
                }
            }

            window.setTimeout(function(){ requestMoreProgress(id_part, id_progressbar_span, id_interrupt) }, 500)
        });
        mutationObserver.observe( progressbar, { childList:true, subtree:true })
	}
}

onUiUpdate(function(){
    check_progressbar('txt2img', 'txt2img_progressbar', 'txt2img_progress_span', 'txt2img_interrupt', 'txt2img_preview', 'txt2img_gallery')
    check_progressbar('img2img', 'img2img_progressbar', 'img2img_progress_span', 'img2img_interrupt', 'img2img_preview', 'img2img_gallery')
})

function requestMoreProgress(id_part, id_progressbar_span, id_interrupt){
    btn = gradioApp().getElementById(id_part+"_check_progress");
    if(btn==null) return;

    btn.click();
    var progressDiv = gradioApp().querySelectorAll('#' + id_progressbar_span).length > 0;
    var interrupt = gradioApp().getElementById(id_interrupt)
    if(progressDiv && interrupt){
        interrupt.style.display = "block"
    }
}

function requestProgress(id_part){
    btn = gradioApp().getElementById(id_part+"_check_progress_initial");
    if(btn==null) return;

    btn.click();
}
