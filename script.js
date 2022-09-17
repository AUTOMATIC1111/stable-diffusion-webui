

titles = {
    "Sampling steps": "How many times to improve the generated image iteratively; higher values take longer; very low values can produce bad results",
    "Sampling method": "Which algorithm to use to produce the image",
	"GFPGAN": "Restore low quality faces using GFPGAN neural network",
	"Euler a": "Euler Ancestral - very creative, each can get a completely different picture depending on step count, setting steps to higher than 30-40 does not help",
	"DDIM": "Denoising Diffusion Implicit Models - best at inpainting",

	"Batch count": "How many batches of images to create",
	"Batch size": "How many image to create in a single batch",
    "CFG Scale": "Classifier Free Guidance Scale - how strongly the image should conform to prompt - lower values produce more creative results",
    "Seed": "A value that determines the output of random number generator - if you create an image with same parameters and seed as another image, you'll get the same result",

    "Inpaint a part of image": "Draw a mask over an image, and the script will regenerate the masked area with content according to prompt",
    "SD upscale": "Upscale image normally, split result into tiles, improve each tile using img2img, merge whole image back",

    "Just resize": "Resize image to target resolution. Unless height and width match, you will get incorrect aspect ratio.",
    "Crop and resize": "Resize the image so that entirety of target resolution is filled with the image. Crop parts that stick out.",
    "Resize and fill": "Resize the image so that entirety of image is inside target resolution. Fill empty space with image's colors.",

    "Mask blur": "How much to blur the mask before processing, in pixels.",
    "Masked content": "What to put inside the masked area before processing it with Stable Diffusion.",
    "fill": "fill it with colors of the image",
    "original": "keep whatever was there originally",
    "latent noise": "fill it with latent space noise",
    "latent nothing": "fill it with latent space zeroes",
    "Inpaint at full resolution": "Upscale masked region to target resolution, do inpainting, downscale back and paste into original image",

    "Denoising strength": "Determines how little respect the algorithm should have for image's content. At 0, nothing will change, and at 1 you'll get an unrelated image. With values below 1.0, processing will take less steps than the Sampling Steps slider specifies.",
    "Denoising strength change factor": "In loopback mode, on each loop the denoising strength is multiplied by this value. <1 means decreasing variety so your sequence will converge on a fixed picture. >1 means increasing variety so your sequence will become more and more chaotic.",

    "Interrupt": "Stop processing images and return any results accumulated so far.",
    "Save": "Write image to a directory (default - log/images) and generation parameters into csv file.",

    "X values": "Separate values for X axis using commas.",
    "Y values": "Separate values for Y axis using commas.",

    "None": "Do not do anything special",
    "Prompt matrix": "Separate prompts into parts using vertical pipe character (|) and the script will create a picture for every combination of them (except for the first part, which will be present in all combinations)",
    "X/Y plot": "Create a grid where images will have different parameters. Use inputs below to specify which parameters will be shared by columns and rows",
    "Custom code": "Run Python code. Advanced user only. Must run program with --allow-code for this to work",

    "Prompt S/R": "Separate a list of words with commas, and the first word will be used as a keyword: script will search for this word in the prompt, and replace it with others",

    "Tiling": "Produce an image that can be tiled.",
    "Tile overlap": "For SD upscale, how much overlap in pixels should there be between tiles. Tiles overlap so that when they are merged back into one picture, there is no clearly visible seam.",

    "Roll": "Add a random artist to the prompt.",

    "Variation seed": "Seed of a different picture to be mixed into the generation.",
    "Variation strength": "How strong of a variation to produce. At 0, there will be no effect. At 1, you will get the complete picture with variation seed (except for ancestral samplers, where you will just get something).",
    "Resize seed from height": "Make an attempt to produce a picture similar to what would have been produced with same seed at specified resolution",
    "Resize seed from width": "Make an attempt to produce a picture similar to what would have been produced with same seed at specified resolution",

    "Interrogate": "Reconstruct prompt from existing image and put it into the prompt field.",

    "Images filename pattern": "Use following tags to define how filenames for images are chosen: [steps], [cfg], [prompt], [prompt_spaces], [width], [height], [sampler], [seed], [model_hash], [prompt_words], [date]; leave empty for default.",
    "Directory name pattern": "Use following tags to define how subdirectories for images and grids are chosen: [steps], [cfg], [prompt], [prompt_spaces], [width], [height], [sampler], [seed], [model_hash], [prompt_words], [date]; leave empty for default.",

    "Loopback": "Process an image, use it as an input, repeat.",
    "Loops": "How many times to repeat processing an image and using it as input for the next iteration",


    "Style 1": "Style to apply; styles have components for both positive and negative prompts and apply to both",
    "Style 2": "Style to apply; styles have components for both positive and negative prompts and apply to both",
    "Apply style": "Insert selected styles into prompt fields",
    "Create style": "Save current prompts as a style. If you add the token {prompt} to the text, the style use that as placeholder for your prompt when you use the style in the future.",

    "Checkpoint name": "Loads weights from checkpoint before making images. You can either use hash or a part of filename (as seen in settings) for checkpoint name. Recommended to use with Y axis for less switching.",
}

function gradioApp(){
    return document.getElementsByTagName('gradio-app')[0].shadowRoot;
}

global_progressbar = null

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
    }
}

function addTitles(root){
	root.querySelectorAll('span, button, select').forEach(function(span){
		tooltip = titles[span.textContent];

		if(!tooltip){
		    tooltip = titles[span.value];
		}

		if(tooltip){
			span.title = tooltip;
		}
	})

	root.querySelectorAll('select').forEach(function(select){
	    if (select.onchange != null) return;

	    select.onchange = function(){
            select.title = titles[select.value] || "";
	    }
	})

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
	
	fullImg_preview = gradioApp().querySelectorAll('img.w-full')

	    if(fullImg_preview != null){
		fullImg_preview.forEach(galleryImageHandler);
	}
	
}

document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m){
        addTitles(gradioApp());
    });
    mutationObserver.observe( gradioApp(), { childList:true, subtree:true })
	
    const modalFragment = document.createDocumentFragment();
    const modal = document.createElement('div')
    modal.onclick = closeModal;
    
    const modalClose = document.createElement('span')
    modalClose.className = 'modalClose cursor';
    modalClose.innerHTML = '&times;'
    modalClose.onclick = closeModal;
    modal.id = "lightboxModal";
    modal.tabIndex=0
    modal.addEventListener('keydown', modalKeyHandler, true)
    modal.appendChild(modalClose)

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

window.addEventListener('paste', e => {
    const files = e.clipboardData.files;
    if (!files || files.length !== 1) {
        return;
    }
    if (!['image/png', 'image/gif', 'image/jpeg'].includes(files[0].type)) {
        return;
    }
    [...gradioApp().querySelectorAll('input[type=file][accept="image/x-png,image/gif,image/jpeg"]')]
        .filter(input => !input.matches('.\\!hidden input[type=file]'))
        .forEach(input => {
            input.files = files;
            input.dispatchEvent(new Event('change'))
        });
});

function ask_for_style_name(_, prompt_text, negative_prompt_text) {
    name_ = prompt('Style name:')
    return name_ === null ? [null, null, null]: [name_, prompt_text, negative_prompt_text]
}
