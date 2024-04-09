{
	"about": "This file is used by Web UI to show the index of available extensions. It's in JSON format and is not meant to be viewed by users directly. If you edit the file you must ensure that it's still a valid JSON.",
	"tags": {
		"script": "a general extension that adds functionality",
		"localization": "a localization extension that translates web ui into another language",
		"tab": "adds a tab",
		"ads": "contains ads",
		"installed": "an extension that is already installed",
		"training": "an extension that allows for new types of training"
	},
	"extensions": [
		{
			"name": "Aesthetic Gradients",
			"url": "https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients.git",
			"description": "Create an embedding from one or few pictures and use it to apply their style to generated images.",
			"added": "2022-11-01",
			"tags": ["script", "training"]
		},
		{
			"name": "Wildcards",
			"url": "https://github.com/AUTOMATIC1111/stable-diffusion-webui-wildcards.git",
			"description": "Sample extension. Allows you to use __name__ syntax in your prompt to get a random line from a file named name.txt in the wildcards directory. Also see Dynamic Prompts for similar functionality.",
			"added": "2022-11-01",
			"tags": ["script"]
		},
		{
			"name": "Dynamic Prompts",
			"url": "https://github.com/adieyal/sd-dynamic-prompts.git",
			"description": "Implements an expressive template language for random or combinatorial prompt generation along with features to support deep wildcard directory structures.",
			"added": "2022-11-01",
			"tags": ["script"]
		},
		{
			"name": "Dreambooth",
			"url": "https://github.com/d8ahazard/sd_dreambooth_extension.git",
			"description": "Dreambooth training based on Shivam Shiaro's repo, optimized for lower-VRAM GPUs.",
			"added": "2022-11-07",
			"tags": ["script", "tab", "training"]
		},
		{
			"name": "Smart Process",
			"url": "https://github.com/d8ahazard/sd_smartprocess.git",
			"description": "Smart pre-process including auto subject identification, caption subject swapping, and upscaling/facial restoration.",
			"added": "2022-11-12",
			"tags": ["script", "tab"]
		},
		{
			"name": "Image browser",
			"url": "https://github.com/AlUlkesh/stable-diffusion-webui-images-browser.git",
			"description": "Provides an interface to browse created images in the web browser. Note: new ext. maintainer, uninstall previous ext. if needed.",
			"added": "2022-11-01",
			"tags": ["script", "tab"]
		},
		{
			"name": "Inspiration",
			"url": "https://github.com/yfszzx/stable-diffusion-webui-inspiration.git",
			"description": "Randomly display the pictures of the artist's or artistic genres typical style, more pictures of this artist or genre is displayed after selecting. So you don't have to worry about how hard it is to choose the right style of art when you create.",
			"added": "2022-11-01",
			"tags": ["script", "tab"]
		},
		{
			"name": "Deforum",
			"url": "https://github.com/deforum-art/deforum-for-automatic1111-webui.git",
			"description": "The official port of Deforum, an extensive script for 2D and 3D animations, supporting keyframable sequences, dynamic math parameters (even inside the prompts), dynamic masking, depth estimation and warping.",
			"added": "2022-11-01",
			"tags": ["script", "tab"]
		},
        {
			"name": "Animator",
			"url": "https://github.com/Animator-Anon/animator_extension.git",
			"description": "A basic img2img script that will dump frames and build a video file. Suitable for creating interesting zoom-in warping movies. This is intended to be a versatile toolset to help you automate some img2img tasks.",
			"added": "2023-01-11",
			"tags": ["script", "tab"]
		},
		{
			"name": "gif2gif",
			"url": "https://github.com/LonicaMewinsky/gif2gif.git",
			"description": "A script for img2img that extract a gif frame by frame for img2img generation and recombine them back into an animated gif",
			"added": "2023-02-09",
			"tags": ["script"]
		},
		{
			"name": "Artists to study",
			"url": "https://github.com/camenduru/stable-diffusion-webui-artists-to-study.git",
			"description": "Shows a gallery of generated pictures by artists separated into categories.",
			"added": "2022-11-01",
			"tags": ["script", "tab"]
		},
		{
			"name": "Aesthetic Image Scorer",
			"url": "https://github.com/tsngo/stable-diffusion-webui-aesthetic-image-scorer.git",
			"description": "Calculates aesthetic score for generated images using CLIP+MLP Aesthetic Score Predictor based on Chad Scorer",
			"added": "2022-11-01",
			"tags": ["script"]
		},
		{
			"name": "Dataset Tag Editor",
			"url": "https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor.git",
			"description": "Feature-rich UI tab that allows image viewing, search-filtering and editing.",
			"added": "2022-11-01",
			"tags": ["script", "tab"]
		},
		{
			"name": "auto-sd-paint-ext",
			"url": "https://github.com/Interpause/auto-sd-paint-ext.git",
			"description": "Krita Plugin.",
			"added": "2022-11-04",
			"tags": ["script"]
		},
		{
			"name": "training-picker",
			"url": "https://github.com/Maurdekye/training-picker.git",
			"description": "Adds a tab to the webui that allows the user to automatically extract keyframes from video, and manually extract 512x512 crops of those frames for use in model training.",
			"added": "2022-11-06",
			"tags": ["script"]
		},
		{
			"name": "Unprompted",
			"url": "https://github.com/ThereforeGames/unprompted.git",
			"description": "Allows you to include various shortcodes in your prompts. You can pull text from files, set up your own variables, process text through conditional functions, and so much more - it's like wildcards on steroids. It now includes txt2img2img and txt2mask custom-script features.",
			"added": "2022-11-04",
			"tags": ["script", "ads"]
		},
		{
			"name": "StylePile",
			"url": "https://github.com/some9000/StylePile.git",
			"description": "An easy way to mix and match elements to prompts that affect the style of the result.",
			"added": "2022-11-24",
			"tags": ["script"]
		},
		{
			"name": "Booru tag autocompletion",
			"url": "https://github.com/DominikDoom/a1111-sd-webui-tagcomplete.git",
			"description": "Displays autocompletion hints for tags from image booru boards such as Danbooru. Uses local tag CSV files and includes a config for customization.",
			"added": "2022-11-04",
			"tags": ["script"]
		},
		{
			"name": "novelai-2-local-prompt",
			"url": "https://github.com/animerl/novelai-2-local-prompt.git",
			"description": "Add a button to convert the prompts used in NovelAI for use in the WebUI. In addition, add a button that allows you to recall a previously used prompt.",
			"added": "2022-11-05",
			"tags": ["script"]
		},
		{
			"name": "tokenizer",
			"url": "https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer.git",
			"description": "Adds a tab that lets you preview how CLIP model would tokenize your text.",
			"added": "2022-11-05",
			"tags": ["script", "tab"]
		},
		{
			"name": "Embeddings editor",
			"url": "https://github.com/CodeExplode/stable-diffusion-webui-embedding-editor.git",
			"description": "Allows you to manually edit textual inversion embeddings using sliders.",
			"added": "2022-11-06",
			"tags": ["script", "tab"]
		},
		{
			"name": "Randomize",
			"url": "https://github.com/innightwolfsleep/stable-diffusion-webui-randomize.git",
			"description": "Allows for random parameters during txt2img generation. This script is processed for all generations, regardless of the script selected, meaning this script will function with others as well, such as AUTOMATIC1111/stable-diffusion-webui-wildcards",
			"added": "2022-11-11",
			"tags": ["script"]
		},
		{
			"name": "embedding-inspector",
			"url": "https://github.com/tkalayci71/embedding-inspector.git",
			"description": "Inspect any token(a word) or Textual-Inversion embeddings and find out which embeddings are similar. You can mix, modify, or create the embeddings in seconds.",
			"added": "2022-12-06",
			"tags": ["script", "tab"]
		},
		{
			"name": "Asymmetric Tiling",
			"url": "https://github.com/tjm35/asymmetric-tiling-sd-webui.git",
			"description": "An always visible script extension to configure seamless image tiling independently for the X and Y axes.",
			"added": "2023-01-13",
			"tags": ["script"]
		},
		{
			"name": "Latent Mirroring",
			"url": "https://github.com/dfaker/SD-latent-mirroring.git",
			"description": "Applies mirroring and flips to the latent images to produce anything from subtle balanced compositions to perfect reflections",
			"added": "2022-11-06",
			"tags": ["script"]
		},
		{
			"name": "seed travel",
			"url": "https://github.com/yownas/seed_travel.git",
			"description": "Small script for AUTOMATIC1111/stable-diffusion-webui to create images that exists between seeds.",
			"added": "2022-11-09",
			"tags": ["script"]
		},
		{
			"name": "shift-attention",
			"url": "https://github.com/yownas/shift-attention.git",
			"description": "Generate a sequence of images shifting attention in the prompt. This script enables you to give a range to the weight of tokens in a prompt and then generate a sequence of images stepping from the first one to the second.",
			"added": "2022-11-09",
			"tags": ["script"]
		},
		{
			"name": "prompt travel",
			"url": "https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel.git",
			"description": "Extension script for AUTOMATIC1111/stable-diffusion-webui to travel between prompts in latent space.",
			"added": "2022-11-11",
			"tags": ["script"]
		},
		{
			"name": "Sonar",
			"url": "https://github.com/Kahsolt/stable-diffusion-webui-sonar.git",
			"description": "Improve the generated image quality, searches for similar (yet even better!) images in the neighborhood of some known image, focuses on single prompt optimization rather than traveling between multiple prompts.",
			"added": "2023-01-12",
			"tags": ["script"]
		},
		{
			"name": "Detection Detailer",
			"url": "https://github.com/dustysys/ddetailer.git",
			"description": "An object detection and auto-mask extension for Stable Diffusion web UI.",
			"added": "2022-11-09",
			"tags": ["script"]
		},
		{
			"name": "Batch Face Swap",
			"url": "https://github.com/kex0/batch-face-swap.git",
			"description": "Automatically detects faces and replaces them.",
			"added": "2023-01-13",
			"tags": ["script"]
		},
		{
			"name": "conditioning-highres-fix",
			"url": "https://github.com/klimaleksus/stable-diffusion-webui-conditioning-highres-fix.git",
			"description": "This is Extension for rewriting Inpainting conditioning mask strength value relative to Denoising strength at runtime. This is useful for Inpainting models such as sd-v1-5-inpainting.ckpt",
			"added": "2022-11-11",
			"tags": ["script"]
		},
		{
			"name": "Auto TLS-HTTPS",
			"url": "https://github.com/papuSpartan/stable-diffusion-webui-auto-tls-https.git",
			"description": "Allows you to easily, or even completely automatically start using HTTPS.",
			"added": "2022-11-14",
			"tags": ["script"]
		},
		{
			"name": "DreamArtist",
			"url": "https://github.com/7eu7d7/DreamArtist-sd-webui-extension.git",
			"description": "Towards Controllable One-Shot Text-to-Image Generation via Contrastive Prompt-Tuning.",
			"added": "2022-11-15",
			"tags": ["script", "training"]
		},
		{
			"name": "WD 1.4 Tagger",
			"url": "https://github.com/toriato/stable-diffusion-webui-wd14-tagger.git",
			"description": "Interrogates single or multiple image files using various alternative models, similar to deepdanbooru interrogate.",
			"added": "2022-11-20",
			"tags": ["script", "tab"]
		},
		{
			"name": "booru2prompt",
			"url": "https://github.com/Malisius/booru2prompt.git",
			"description": "This SD extension allows you to turn posts from various image boorus into stable diffusion prompts. It does so by pulling a list of tags down from their API. You can copy-paste in a link to the post you want yourself, or use the built-in search feature to do it all without leaving SD.",
			"added": "2022-11-21",
			"tags": ["script", "tab"]
		},
		{
			"name": "Gelbooru Prompt",
			"url": "https://github.com/antis0007/sd-webui-gelbooru-prompt.git",
			"description": "Extension that gets tags for saved gelbooru images in AUTOMATIC1111's Stable Diffusion webui",
			"added": "2022-12-20",
			"tags": ["script"]
		},
		{
			"name": "Merge Board",
			"url": "https://github.com/bbc-mc/sdweb-merge-board.git",
			"description": "Multiple lane merge support(up to 10). Save and Load your merging combination as Recipes, which is simple text.",
			"added": "2022-11-21",
			"tags": ["script", "tab"]
		},
		{
			"name": "Depth Maps",
			"url": "https://github.com/thygate/stable-diffusion-webui-depthmap-script.git",
			"description": "Depth Maps, Stereo Image, 3D Mesh and Video generator extension.",
			"added": "2022-11-30",
			"tags": ["script"]
		},
		{
			"name": "multi-subject-render",
			"url": "https://github.com/Extraltodeus/multi-subject-render.git",
			"description": "It is a depth aware extension that can help to create multiple complex subjects on a single image. It generates a background, then multiple foreground subjects, cuts their backgrounds after a depth analysis, paste them onto the background and finally does an img2img for a clean finish.",
			"added": "2022-11-24",
			"tags": ["script"]
		},
		{
			"name": "depthmap2mask",
			"url": "https://github.com/Extraltodeus/depthmap2mask.git",
			"description": "Create masks for img2img based on a depth estimation made by MiDaS.",
			"added": "2022-11-26",
			"tags": ["script"]
		},
		{
			"name": "ABG_extension",
			"url": "https://github.com/KutsuyaYuki/ABG_extension.git",
			"description": "Automatically remove backgrounds. Uses an onnx model fine-tuned for anime images. Runs on GPU.",
			"added": "2022-12-24",
			"tags": ["script"]
		},
        {
			"name": "Depth Image I/O",
			"url": "https://github.com/AnonymousCervine/depth-image-io-for-SDWebui.git",
			"description": "An extension to allow managing custom depth inputs to Stable Diffusion depth2img models.",
			"added": "2023-01-17",
			"tags": ["script"]
		},
		{
			"name": "Visualize Cross-Attention",
			"url": "https://github.com/benkyoujouzu/stable-diffusion-webui-visualize-cross-attention-extension.git",
			"description": "Generates highlighted sectors of a submitted input image, based on input prompts. Use with tokenizer extension. See the readme for more info.",
			"added": "2022-11-25",
			"tags": ["script", "tab"]
		},
		{
			"name": "DAAM",
			"url": "https://github.com/kousw/stable-diffusion-webui-daam.git",
			"description": "DAAM stands for Diffusion Attentive Attribution Maps. Enter the attention text (must be a string contained in the prompt) and run. An overlapping image with a heatmap for each attention will be generated along with the original image.",
			"added": "2022-12-02",
			"tags": ["script"]
		},
		{
			"name": "Prompt Gallery",
			"url": "https://github.com/dr413677671/PromptGallery-stable-diffusion-webui.git",
			"description": "Build a yaml file filled with prompts of your character, hit generate, and quickly preview them by their word attributes and modifiers.",
			"added": "2022-12-02",
			"tags": ["script", "tab"]
		},
		{
			"name": "Infinity Grid Generator",
			"url": "https://github.com/mcmonkeyprojects/sd-infinity-grid-generator-script.git",
			"description": "Build a yaml file with your chosen parameters, and generate infinite-dimensional grids. Built-in ability to add description text to fields. See readme for usage details.",
			"added": "2022-12-09",
			"tags": ["script"]
		},
		{
			"name": "NSFW checker",
			"url": "https://github.com/AUTOMATIC1111/stable-diffusion-webui-nsfw-censor.git",
			"description": "Replaces NSFW images with black.",
			"added": "2022-12-10",
			"tags": ["script"]
		},
		{
			"name": "Diffusion Defender",
			"url": "https://github.com/WildBanjos/DiffusionDefender.git",
			"description": "Prompt blacklist, find and replace, for semi-private and public instances.",
			"added": "2022-12-20",
			"tags": ["script"]
		},
		{
			"name": "Config-Presets",
			"url": "https://github.com/Zyin055/Config-Presets.git",
			"description": "Adds a configurable dropdown to allow you to change UI preset settings in the txt2img and img2img tabs.",
			"added": "2022-12-13",
			"tags": ["script"]
		},
		{
			"name": "Preset Utilities",
			"url": "https://github.com/Gerschel/sd_web_ui_preset_utils.git",
			"description": "Preset utility tool for ui. Offers compatibility with custom scripts. (to a limit)",
			"added": "2022-12-19",
			"tags": ["script"]
		},
		{
			"name": "DH Patch",
			"url": "https://github.com/d8ahazard/sd_auto_fix.git",
			"description": "Random patches by D8ahazard. Auto-load config YAML files for v2, 2.1 models; patch latent-diffusion to fix attention on 2.1 models (black boxes without no-half), whatever else I come up with.",
			"added": "2022-12-16",
			"tags": ["script"]
		},
		{
			"name": "Riffusion",
			"url": "https://github.com/enlyth/sd-webui-riffusion.git",
			"description": "Use Riffusion model to produce music in gradio. To replicate original interpolation technique, input the prompt travel extension output frames into the riffusion tab.",
			"added": "2022-12-19",
			"tags": ["script", "tab"]
		},
		{
			"name": "Save Intermediate Images",
			"url": "https://github.com/AlUlkesh/sd_save_intermediate_images.git",
			"description": "Save intermediate images during the sampling process. You can also make videos from the intermediate images.",
			"added": "2022-12-22",
			"tags": ["script"]
		},
		{
			"name": "openOutpaint extension",
			"url": "https://github.com/zero01101/openOutpaint-webUI-extension.git",
			"description": "A tab with the full openOutpaint UI. Run with the --api flag.",
			"added": "2022-12-23",
			"tags": ["script", "tab"]
		},
		{
			"name": "model-keyword",
			"url": "https://github.com/mix1009/model-keyword.git",
			"description": "Inserts matching keyword(s) to the prompt automatically. Update this extension to get the latest model+keyword mappings.",
			"added": "2022-12-28",
			"tags": ["script"]
		},
		{
			"name": "Prompt Generator",
			"url": "https://github.com/imrayya/stable-diffusion-webui-Prompt_Generator.git",
			"description": "generate a prompt from a small base prompt using distilgpt2. Adds a tab with additional control of the model.",
			"added": "2022-12-30",
			"tags": ["script", "tab"]
		},
		{
			"name": "Prompt Translator",
			"url": "https://github.com/butaixianran/Stable-Diffusion-Webui-Prompt-Translator",
			"description": "A integrated translator for translating prompts to English using Deepl or Baidu.",
			"added": "2023-02-11",
			"tags": ["script", "tab"]
		},
		{
			"name": "quick-css",
			"url": "https://github.com/Gerschel/sd-web-ui-quickcss.git",
			"description": "Extension for quickly selecting and applying custom.css files, for customizing look and placement of elements in ui.",
			"added": "2022-12-30",
			"tags": ["script", "tab"]
		},
		{
			"name": "Add image number to grid",
			"url": "https://github.com/AlUlkesh/sd_grid_add_image_number.git",
			"description": "Add the image's number to its picture in the grid.",
			"added": "2023-01-01",
			"tags": ["script"]
		},
		{
			"name": "Model Converter",
			"url": "https://github.com/Akegarasu/sd-webui-model-converter.git",
			"description": "Convert models to fp16/bf16 no-ema/ema-only safetensors. Convert/copy/delete any parts of model: unet, text encoder(clip), vae.",
			"added": "2023-01-05",
			"tags": ["script", "tab"]
		},
		{
			"name": "Kohya-ss Additional Networks",
			"url": "https://github.com/kohya-ss/sd-webui-additional-networks.git",
			"description": "Allows the Web UI to use networks (LoRA) trained by their scripts to generate images.",
			"added": "2023-01-06",
			"tags": ["script"]
		},
		{
			"name": "Ultimate SD Upscale",
			"url": "https://github.com/Coyote-A/ultimate-upscale-for-automatic1111.git",
			"description": "More advanced options for SD Upscale, less artifacts than original using higher denoise ratio (0.3-0.5).",
			"added": "2023-01-10",
			"tags": ["script"]
		},
		{
			"name": "Stable Horde Worker",
			"url": "https://github.com/sdwebui-w-horde/sd-webui-stable-horde-worker.git",
			"description": "Worker Client for Stable Horde. Generate pictures for other users with your PC. Please see readme for additional instructions.",
			"added": "2023-01-10",
			"tags": ["script", "tab"]
		},
		{
			"name": "Stable Horde Client",
			"url": "https://github.com/natanjunges/stable-diffusion-webui-stable-horde.git",
			"description": "Stable Horde Client. Generate pictures using other user's PC. Useful if u have no GPU.",
			"added": "2023-01-11",
			"tags": ["script", "tab"]
		},
		{
			"name": "Hypernetwork-Monkeypatch-Extension",
			"url": "https://github.com/aria1th/Hypernetwork-MonkeyPatch-Extension.git",
			"description": "Extension that provides additional training features for hypernetwork training. Also supports using multiple hypernetworks for inference.",
			"added": "2023-01-12",
			"tags": ["script", "tab", "training"]
		},
		{
			"name": "Multiple Hypernetworks",
			"url": "https://github.com/antis0007/sd-webui-multiple-hypernetworks.git",
			"description": "Adds the ability to apply multiple hypernetworks at once. Apply multiple hypernetworks sequentially, with different weights.",
			"added": "2023-01-13",
			"tags": ["script"]
		},
		{
			"name": "Merge Block Weighted",
			"url": "https://github.com/bbc-mc/sdweb-merge-block-weighted-gui.git",
			"description": "Merge models with separate rate for each 25 U-Net block (input, middle, output).",
			"added": "2023-01-13",
			"tags": ["script", "tab"]
		},
		{
			"name": "haku-img",
			"url": "https://github.com/KohakuBlueleaf/a1111-sd-webui-haku-img.git",
			"description": "Image utils extension. Allows blending, layering, hue and color adjustments, blurring and sketch effects, and basic pixelization.",
			"added": "2023-01-17",
			"tags": ["script", "tab"]
		},
		{
			"name": "Promptgen",
			"url": "https://github.com/AUTOMATIC1111/stable-diffusion-webui-promptgen.git",
			"description": "Use transformers models to generate prompts.",
			"added": "2023-01-18",
			"tags": ["script", "tab"]
		},
		{
			"name": "Discord Rich Presence",
			"url": "https://github.com/kabachuha/discord-rpc-for-automatic1111-webui.git",
			"description": "Provides connection to Discord RPC, showing a fancy table in the user profile.",
			"added": "2023-01-20",
			"tags": ["script"]
		},
    {
			"name": "Steps Animation",
			"url": "https://github.com/vladmandic/sd-extension-steps-animation.git",
			"description": "Create animation sequence from denoised intermediate steps.",
			"added": "2023-01-21",
			"tags": ["script"]
		},
		{
			"name": "Aesthetic Scorer",
			"url": "https://github.com/vladmandic/sd-extension-aesthetic-scorer.git",
			"description": "Uses existing CLiP model with an additional small pretrained model to calculate perceived aesthetic score of an image.",
			"added": "2023-01-21",
			"tags": ["script"]
		},
		{
			"name": "System Info",
			"url": "https://github.com/vladmandic/sd-extension-system-info.git",
			"description": "System Info tab for WebUI which shows realtime information of the server.",
			"added": "2023-01-21",
			"tags": ["script", "tab"]
		},
		{
			"name": "Pixelization",
			"url": "https://github.com/AUTOMATIC1111/stable-diffusion-webui-pixelization.git",
			"description": "Using pre-trained models, produce pixel art out of images in the extras tab.",
			"added": "2023-01-23",
			"tags": ["script"]
		},
		{
			"name": "Instruct-pix2pix",
			"url": "https://github.com/Klace/stable-diffusion-webui-instruct-pix2pix.git",
			"description": "Adds a tab for doing img2img editing with the instruct-pix2pix model. Note: No longer required. Author has integrated code to webui, use in img2img.",
			"added": "2023-01-25",
			"tags": ["script", "tab"]
		},
		{
			"name": "Custom Diffusion",
			"url": "https://github.com/guaneec/custom-diffusion-webui.git",
			"description": "Custom Diffusion is, in short, finetuning-lite with TI, instead of tuning the whole model. Similar speed and memory requirements to TI and supposedly gives better results in less steps.",
			"added": "2023-01-28",
			"tags": ["script", "tab", "training"]
		},
		{
			"name": "Fusion",
			"url": "https://github.com/ljleb/prompt-fusion-extension.git",
			"description": "Adds prompt-travel and shift-attention-like interpolations (see exts), but during/within the sampling steps. Always-on + works w/ existing prompt-editing syntax. Various interpolation modes. See their wiki for more info.",
			"added": "2023-01-28",
			"tags": ["script"]
		},
		{
			"name": "cafe-aesthetic",
			"url": "https://github.com/p1atdev/stable-diffusion-webui-cafe-aesthetic.git",
			"description": "Pre-trained model, determines if aesthetic/non-aesthetic, does 5 different style recognition modes, and Waifu confirmation. Also has a tab with Batch processing.",
			"added": "2023-01-28",
			"tags": ["script", "tab"]
		},
		{
			"name": "Dynamic Thresholding",
			"url": "https://github.com/mcmonkeyprojects/sd-dynamic-thresholding.git",
			"description": "Adds customizable dynamic thresholding to allow high CFG Scale values without the burning / 'pop art' effect.",
			"added": "2023-02-01",
			"tags": ["script"]
		},
		{
			"name": "Aspect Ratio selector",
			"url": "https://github.com/alemelis/sd-webui-ar.git",
			"description": "Adds image aspect ratio selector buttons.",
			"added": "2023-02-04",
			"tags": ["script"]
		},
		{
			"name": "Catppuccin Theme",
			"url": "https://github.com/catppuccin/stable-diffusion-webui.git",
			"description": "Adds various custom themes",
			"added": "2023-02-04",
			"tags": ["script"]
		},
		{
			"name": "Embedding Merge",
			"url": "https://github.com/klimaleksus/stable-diffusion-webui-embedding-merge.git",
			"description": "Merging Textual Inversion embeddings at runtime from string literals.",
			"added": "2023-02-09",
			"tags": ["script", "tab"]
		},
		{
			"name": "anti-burn",
			"url": "https://github.com/klimaleksus/stable-diffusion-webui-anti-burn.git",
			"description": "Smoothing generated images by skipping a few very last steps and averaging together some images before them",
			"added": "2023-02-09",
			"tags": ["script", "tab"]
		},
		{
			"name": "zh_CN Localization",
			"url": "https://github.com/dtlnor/stable-diffusion-webui-localization-zh_CN",
			"description": "Simplified Chinese localization",
			"added": "2022-11-06",
			"tags": ["localization"]
		},
		{
			"name": "zh_TW Localization",
			"url": "https://github.com/benlisquare/stable-diffusion-webui-localization-zh_TW",
			"description": "Traditional Chinese localization",
			"added": "2022-11-09",
			"tags": ["localization"]
		},
		{
			"name": "ko_KR Localization",
			"url": "https://github.com/36DB/stable-diffusion-webui-localization-ko_KR",
			"description": "Korean localization",
			"added": "2022-11-06",
			"tags": ["localization"]
		},
		{
			"name": "th_TH Localization",
			"url": "https://github.com/econDS/thai-localization-for-Automatic-stable-diffusion-webui",
			"description": "Thai localization",
			"added": "2022-12-30",
			"tags": ["localization"]
		},
		{
			"name": "es_ES Localization",
			"url": "https://github.com/innovaciones/stable-diffusion-webui-localization-es_ES",
			"description": "Spanish localization",
			"added": "2022-11-09",
			"tags": ["localization"]
		},
		{
			"name": "it_IT Localization",
			"url": "https://github.com/Harvester62/stable-diffusion-webui-localization-it_IT",
			"description": "Italian localization",
			"added": "2022-11-07",
			"tags": ["localization"]
		},
		{
			"name": "de_DE Localization",
			"url": "https://github.com/Strothis/stable-diffusion-webui-de_DE",
			"description": "German localization",
			"added": "2022-11-07",
			"tags": ["localization"]
		},
		{
			"name": "ja_JP Localization",
			"url": "https://github.com/Katsuyuki-Karasawa/stable-diffusion-webui-localization-ja_JP",
			"description": "Japanese localization",
			"added": "2022-11-07",
			"tags": ["localization"]
		},
		{
			"name": "pt_BR Localization",
			"url": "https://github.com/M-art-ucci/stable-diffusion-webui-localization-pt_BR",
			"description": "Brazillian portuguese localization",
			"added": "2022-11-09",
			"tags": ["localization"]
		},
		{
			"name": "tr_TR Localization",
			"url": "https://github.com/camenduru/stable-diffusion-webui-localization-tr_TR",
			"description": "Turkish localization",
			"added": "2022-11-12",
			"tags": ["localization"]
		},
		{
			"name": "no_NO Localization",
			"url": "https://github.com/Cyanz83/stable-diffusion-webui-localization-no_NO",
			"description": "Norwegian localization",
			"added": "2022-11-16",
			"tags": ["localization"]
		},
		{
			"name": "ru_RU Localization",
			"url": "https://github.com/ProfaneServitor/stable-diffusion-webui-localization-ru_RU",
			"description": "Russian localization",
			"added": "2022-11-20",
			"tags": ["localization"]
		},
		{
			"name": "fi_FI Localization",
			"url": "https://github.com/otsoniemi/stable-diffusion-webui-localization-fi_FI",
			"description": "Finnish localization",
			"added": "2022-12-28",
			"tags": ["localization"]
		},
		{
			"name": "old localizations",
			"url": "https://github.com/AUTOMATIC1111/stable-diffusion-webui-old-localizations.git",
			"description": "Old unmaintained localizations that used to be a part of main repository",
			"added": "2022-11-08",
			"tags": ["localization"]
		}
	]
}