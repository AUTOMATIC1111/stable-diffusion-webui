// mouseover tooltips for various UI elements

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
    "\u{1f3b2}\ufe0f": "Set seed to -1, which will cause a new random number to be used every time",
    "\u267b\ufe0f": "Reuse seed from last generation, mostly useful if it was randomed",
    "\u{1f3a8}": "Add a random artist to the prompt.",
    "\u2199\ufe0f": "Read generation parameters from prompt or last generation if prompt is empty into user interface.",
    "\u{1f4c2}": "Open images output directory",
    "\u{1f4be}": "Save style",
    "\u{1f4cb}": "Apply selected styles to current prompt",

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

    "Skip": "Stop processing current image and continue processing.",
    "Interrupt": "Stop processing images and return any results accumulated so far.",
    "Save": "Write image to a directory (default - log/images) and generation parameters into csv file.",

    "X values": "Separate values for X axis using commas.",
    "Y values": "Separate values for Y axis using commas.",

    "None": "Do not do anything special",
    "Prompt matrix": "Separate prompts into parts using vertical pipe character (|) and the script will create a picture for every combination of them (except for the first part, which will be present in all combinations)",
    "X/Y plot": "Create a grid where images will have different parameters. Use inputs below to specify which parameters will be shared by columns and rows",
    "Custom code": "Run Python code. Advanced user only. Must run program with --allow-code for this to work",

    "Prompt S/R": "Separate a list of words with commas, and the first word will be used as a keyword: script will search for this word in the prompt, and replace it with others",
    "Prompt order": "Separate a list of words with commas, and the script will make a variation of prompt with those words for their every possible order",

    "Tiling": "Produce an image that can be tiled.",
    "Tile overlap": "For SD upscale, how much overlap in pixels should there be between tiles. Tiles overlap so that when they are merged back into one picture, there is no clearly visible seam.",

    "Variation seed": "Seed of a different picture to be mixed into the generation.",
    "Variation strength": "How strong of a variation to produce. At 0, there will be no effect. At 1, you will get the complete picture with variation seed (except for ancestral samplers, where you will just get something).",
    "Resize seed from height": "Make an attempt to produce a picture similar to what would have been produced with same seed at specified resolution",
    "Resize seed from width": "Make an attempt to produce a picture similar to what would have been produced with same seed at specified resolution",

    "Interrogate": "Reconstruct prompt from existing image and put it into the prompt field.",

    "Images filename pattern": "Use following tags to define how filenames for images are chosen: [steps], [cfg], [prompt], [prompt_no_styles], [prompt_spaces], [width], [height], [styles], [sampler], [seed], [model_hash], [prompt_words], [date], [datetime], [datetime<Format>], [datetime<Format><Time Zone>], [job_timestamp]; leave empty for default.",
    "Directory name pattern": "Use following tags to define how subdirectories for images and grids are chosen: [steps], [cfg], [prompt], [prompt_no_styles], [prompt_spaces], [width], [height], [styles], [sampler], [seed], [model_hash], [prompt_words], [date], [datetime], [datetime<Format>], [datetime<Format><Time Zone>], [job_timestamp]; leave empty for default.",
    "Max prompt words": "Set the maximum number of words to be used in the [prompt_words] option; ATTENTION: If the words are too long, they may exceed the maximum length of the file path that the system can handle",

    "Loopback": "Process an image, use it as an input, repeat.",
    "Loops": "How many times to repeat processing an image and using it as input for the next iteration",

    "Style 1": "Style to apply; styles have components for both positive and negative prompts and apply to both",
    "Style 2": "Style to apply; styles have components for both positive and negative prompts and apply to both",
    "Apply style": "Insert selected styles into prompt fields",
    "Create style": "Save current prompts as a style. If you add the token {prompt} to the text, the style use that as placeholder for your prompt when you use the style in the future.",

    "Checkpoint name": "Loads weights from checkpoint before making images. You can either use hash or a part of filename (as seen in settings) for checkpoint name. Recommended to use with Y axis for less switching.",
    "Inpainting conditioning mask strength": "Only applies to inpainting models. Determines how strongly to mask off the original image for inpainting and img2img. 1.0 means fully masked, which is the default behaviour. 0.0 means a fully unmasked conditioning. Lower values will help preserve the overall composition of the image, but will struggle with large changes.",

    "vram": "Torch active: Peak amount of VRAM used by Torch during generation, excluding cached data.\nTorch reserved: Peak amount of VRAM allocated by Torch, including all active and cached data.\nSys VRAM: Peak amount of VRAM allocation across all applications / total GPU VRAM (peak utilization%).",

    "Highres. fix": "Use a two step process to partially create an image at smaller resolution, upscale, and then improve details in it without changing composition",
    "Scale latent": "Uscale the image in latent space. Alternative is to produce the full image from latent representation, upscale that, and then move it back to latent space.",

    "Eta noise seed delta": "If this values is non-zero, it will be added to seed and used to initialize RNG for noises when using samplers with Eta. You can use this to produce even more variation of images, or you can use this to match images of other software if you know what you are doing.",
    "Do not add watermark to images": "If this option is enabled, watermark will not be added to created images. Warning: if you do not add watermark, you may be behaving in an unethical manner.",

    "Filename word regex": "This regular expression will be used extract words from filename, and they will be joined using the option below into label text used for training. Leave empty to keep filename text as it is.",
    "Filename join string": "This string will be used to join split words into a single line if the option above is enabled.",

    "Quicksettings list": "List of setting names, separated by commas, for settings that should go to the quick access bar at the top, rather than the usual setting tab. See modules/shared.py for setting names. Requires restarting to apply.",

    "Weighted sum": "Result = A * (1 - M) + B * M",
    "Add difference": "Result = A + (B - C) * M",

    "Learning rate": "how fast should the training go. Low values will take longer to train, high values may fail to converge (not generate accurate results) and/or may break the embedding (This has happened if you see Loss: nan in the training info textbox. If this happens, you need to manually restore your embedding from an older not-broken backup).\n\nYou can set a single numeric value, or multiple learning rates using the syntax:\n\n   rate_1:max_steps_1, rate_2:max_steps_2, ...\n\nEG:   0.005:100, 1e-3:1000, 1e-5\n\nWill train with rate of 0.005 for first 100 steps, then 1e-3 until 1000 steps, then 1e-5 for all remaining steps.",
}


onUiUpdate(function(){
	gradioApp().querySelectorAll('span, button, select, p').forEach(function(span){
		tooltip = titles[span.textContent];

		if(!tooltip){
		    tooltip = titles[span.value];
		}

		if(!tooltip){
			for (const c of span.classList) {
				if (c in titles) {
					tooltip = titles[c];
					break;
				}
			}
		}

		if(tooltip){
			span.title = tooltip;
		}
	})

	gradioApp().querySelectorAll('select').forEach(function(select){
	    if (select.onchange != null) return;

	    select.onchange = function(){
            select.title = titles[select.value] || "";
	    }
	})
})
