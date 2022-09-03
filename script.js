titles = {
    "Sampling steps": "How many times to imptove the generated image itratively; higher values take longer; very low values can produce bad results",
    "Sampling method": "Which algorithm to use to produce the image",
	"GFPGAN": "Restore low quality faces using GFPGAN neural network",
	"Euler a": "Euler Ancestral - very creative, each can get acompletely different pictures depending on step count, setting seps tohigher than 30-40 does not help",
	"DDIM": "Denoising Diffusion Implicit Models - best at inpainting",

	"Batch count": "How many batches of images to create",
	"Batch size": "How many image to create in a single batch",
    "CFG Scale": "Classifier Free Guidance Scale - how strongly the image should conform to prompt - lower values produce more creative results",
    "Seed": "A value that determines the output of random number generator - if you create an image with same parameters and seed as another image, you'll get the same result",

    "Inpaint a part of image": "Draw a mask over an image, and the script will regenerate the masked area with content according to prompt",
    "Loopback": "Process an image, use it as an input, repeat. Batch count determings number of iterations.",
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

    "Denoising Strength": "Determines how little respect the algorithm should have for image's content. At 0, nothing will change, and at 1 you'll get an unrelated image.",

    "Interrupt": "Stop processing images and return any results accumulated so far.",
    "Save": "Write image to a directory (default - log/images) and generation parameters into csv file.",

    "X values": "Separate values for X axis using commas.",
    "Y values": "Separate values for Y axis using commas.",

    "None": "Do not do anything special",
    "Prompt matrix": "Separate prompts into parts using vertical pipe character (|) and the script will create a picture for every combination of them (except for the first part, which will be present in all combinations)",
    "X/Y plot": "Create a grid where images will have different parameters. Use inputs below to specify which parameterswill be shared by columns and rows",
    "Custom code": "Run python code. Advanced user only. Must run program with --allow-code for this to work",

    "Prompt S/R": "Separate a list of words with commas, and the first word will be used as a keyword: script will search for this word in the prompt, and replace it with others",
}

function gradioApp(){
    return document.getElementsByTagName('gradio-app')[0].shadowRoot;
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
}

document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m){
        addTitles(gradioApp());
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
        return []
    }

    return gallery[index];
}
