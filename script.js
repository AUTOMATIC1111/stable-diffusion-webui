console.log("running")

titles = {
    "Sampling steps": "How many times to imptove the generated image itratively; higher values take longer; very low values can produce bad results",
    "Sampling method": "Which algorithm to use to produce the image",
	"GFPGAN": "Restore low quality faces using GFPGAN neural network",
	"Euler a": "Euler Ancestral - very creative, each can get acompletely different pictures depending on step count, setting seps tohigher than 30-40 does not help",
	"DDIM": "Denoising Diffusion Implicit Models - best at inpainting",
	"Prompt matrix": "Separate prompts into part using vertical pipe character (|) and the script will create a picture for every combination of them (except for first part, which will be present in all combinations)",
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
}

function gradioApp(){
    return document.getElementsByTagName('gradio-app')[0];
}

function addTitles(root){
	root.querySelectorAll('span').forEach(function(span){
		tooltip = titles[span.textContent];
		if(tooltip){
			span.title = tooltip;
		}
	})
}

document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m){
        addTitles(gradioApp().shadowRoot);
    });
    mutationObserver.observe( gradioApp().shadowRoot, { childList:true, subtree:true })

});
