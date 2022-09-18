function gradioApp(){
    return document.getElementsByTagName('gradio-app')[0].shadowRoot;
}

global_progressbar = null

uiUpdateCallbacks = []
function onUiUpdate(callback){
    uiUpdateCallbacks.push(callback)
}

function uiUpdate(root){
	progressbar = root.getElementById('progressbar')
	if(progressbar!= null && progressbar != global_progressbar){
	    global_progressbar = progressbar

        var mutationObserver = new MutationObserver(function(m){
            txt2img_preview = gradioApp().getElementById('txt2img_preview')
            txt2img_gallery = gradioApp().getElementById('txt2img_gallery')

            img2img_preview = gradioApp().getElementById('img2img_preview')
            img2img_gallery = gradioApp().getElementById('img2img_gallery')

            if(txt2img_preview != null && txt2img_gallery != null){
                txt2img_preview.style.width = txt2img_gallery.clientWidth + "px"
                txt2img_preview.style.height = txt2img_gallery.clientHeight + "px"
            }

            if(img2img_preview != null && img2img_gallery != null){
                img2img_preview.style.width = img2img_gallery.clientWidth + "px"
                img2img_preview.style.height = img2img_gallery.clientHeight + "px"
            }
		
            window.setTimeout(requestProgress, 500)
        });
        mutationObserver.observe( progressbar, { childList:true, subtree:true })
	}


	uiUpdateCallbacks.forEach(function(x){
	    x()
	})
}

document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m){
        uiUpdate(gradioApp());
    });
    mutationObserver.observe( gradioApp(), { childList:true, subtree:true })
});

function selected_gallery_index(){
    var gr = gradioApp()
    var buttons = gradioApp().querySelectorAll(".gallery-item")
    var button = gr.querySelector(".gallery-item.\\!ring-2")

    var result = -1
    buttons.forEach(function(v, i){ if(v==button) { result = i } })

    return result
}

function extract_image_from_gallery(gallery){
    if(gallery.length == 1){
        return gallery[0]
    }

    index = selected_gallery_index()

    if (index < 0 || index >= gallery.length){
        return [null]
    }

    return gallery[index];
}

function extract_image_from_gallery_img2img(gallery){
    gradioApp().querySelectorAll('button')[1].click();
    return extract_image_from_gallery(gallery);
}

function extract_image_from_gallery_extras(gallery){
    gradioApp().querySelectorAll('button')[2].click();
    return extract_image_from_gallery(gallery);
}

function requestProgress(){
    btn = gradioApp().getElementById("check_progress");
    if(btn==null) return;

    btn.click();
}

function submit(){
    window.setTimeout(requestProgress, 500)

    res = []
    for(var i=0;i<arguments.length;i++){
        res.push(arguments[i])
    }

    // As it is currently, txt2img and img2img send back the previous output args (txt2img_gallery, generation_info, html_info) whenever you generate a new image.
    // This can lead to uploading a huge gallery of previously generated images, which leads to an unnecessary delay between submitting and beginning to generate.
    // I don't know why gradio is seding outputs along with inputs, but we can prevent sending the image gallery here, which seems to be an issue for some.
    // If gradio at some point stops sending outputs, this may break something
    if(Array.isArray(res[res.length - 3])){
        res[res.length - 3] = null
    }

    return res
}

function ask_for_style_name(_, prompt_text, negative_prompt_text) {
    name_ = prompt('Style name:')
    return name_ === null ? [null, null, null]: [name_, prompt_text, negative_prompt_text]
}
