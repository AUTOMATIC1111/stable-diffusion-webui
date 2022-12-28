{
	"about": "This file is used by Web UI to show the index of available extensions. It's in JSON format and is not meant to be viewed by users directly. If you edit the file you must ensure that it's still a valid JSON.",
	"tags": {
		"script": "a general extension that adds functionality",
		"localization": "a localization extension that translates web ui into another language",
		"tab": "adds a tab",
		"ads": "contains ads",
		"installed": "an extension that is already installed"
	},
	"extensions": [
		{
			"name": "Aesthetic Gradients",
			"url": "https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients",
			"description": "Create an embedding from one or few pictures and use it to apply their style to generated images.",
			"tags": ["script"]
		},
		{
			"name": "Wildcards",
			"url": "https://github.com/AUTOMATIC1111/stable-diffusion-webui-wildcards",
			"description": "Sample extension. Allows you to use __name__ syntax in your prompt to get a random line from a file named name.txt in the wildcards directory. Also see Dynamic Prompts for similar functionality.",
			"tags": ["script"]
		},
		{
			"name": "Dynamic Prompts",
			"url": "https://github.com/adieyal/sd-dynamic-prompts",
			"description": "Implements an expressive template language for random or combinatorial prompt generation along with features to support deep wildcard directory structures.",
			"tags": ["script"]
		},
		{
			"name": "Dreambooth",
			"url": "https://github.com/d8ahazard/sd_dreambooth_extension",
			"description": "Dreambooth training based on Shivam Shiaro's repo, optimized for lower-VRAM GPUs.",
			"tags": ["script", "tab"]
		},
		{
			"name": "Smart Process",
			"url": "https://github.com/d8ahazard/sd_smartprocess",
			"description": "Smart pre-process including auto subject identification, caption subject swapping, and upscaling/facial restoration.",
			"tags": ["script", "tab"]
		},
		{
			"name": "Image browser",
			"url": "https://github.com/yfszzx/stable-diffusion-webui-images-browser",
			"description": "Provides an interface to browse created images in the web browser.",
			"tags": ["script", "tab"]
		},
		{
			"name": "Inspiration",
			"url": "https://github.com/yfszzx/stable-diffusion-webui-inspiration",
			"description": "Randomly display the pictures of the artist's or artistic genres typical style, more pictures of this artist or genre is displayed after selecting. So you don't have to worry about how hard it is to choose the right style of art when you create.",
			"tags": ["script", "tab"]
		},
		{
			"name": "Deforum",
			"url": "https://github.com/deforum-art/deforum-for-automatic1111-webui",
			"description": "The official port of Deforum, an extensive script for 2D and 3D animations, supporting keyframable sequences, dynamic math parameters (even inside the prompts), dynamic masking, depth estimation and warping.",
			"tags": ["script", "tab"]
		},
		{
			"name": "Artists to study",
			"url": "https://github.com/camenduru/stable-diffusion-webui-artists-to-study",
			"description": "Shows a gallery of generated pictures by artists separated into categories.",
			"tags": ["script", "tab"]
		},
		{
			"name": "Aesthetic Image Scorer",
			"url": "https://github.com/tsngo/stable-diffusion-webui-aesthetic-image-scorer",
			"description": "Calculates aesthetic score for generated images using CLIP+MLP Aesthetic Score Predictor based on Chad Scorer",
			"tags": ["script"]
		},
		{
			"name": "Dataset Tag Editor",
			"url": "https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor",
			"description": "Feature-rich UI tab that allows image viewing, search-filtering and editing.",
			"tags": ["script", "tab"]
		},
		{
			"name": "auto-sd-paint-ext",
			"url": "https://github.com/Interpause/auto-sd-paint-ext",
			"description": "Krita Plugin.",
			"tags": ["script"]
		},
		{
			"name": "training-picker",
			"url": "https://github.com/Maurdekye/training-picker",
			"description": "Adds a tab to the webui that allows the user to automatically extract keyframes from video, and manually extract 512x512 crops of those frames for use in model training.",
			"tags": ["script"]
		},
		{
			"name": "Unprompted",
			"url": "https://github.com/ThereforeGames/unprompted",
			"description": "Allows you to include various shortcodes in your prompts. You can pull text from files, set up your own variables, process text through conditional functions, and so much more - it's like wildcards on steroids. It now includes txt2img2img and txt2mask custom-script features.",
			"tags": ["script", "ads"]
		},
		{
			"name": "StylePile",
			"url": "https://github.com/some9000/StylePile",
			"description": "An easy way to mix and match elements to prompts that affect the style of the result.",
			"tags": ["script"]
		},
		{
			"name": "Booru tag autocompletion",
			"url": "https://github.com/DominikDoom/a1111-sd-webui-tagcomplete",
			"description": "Displays autocompletion hints for tags from image booru boards such as Danbooru. Uses local tag CSV files and includes a config for customization.",
			"tags": ["script"]
		},
		{
			"name": "novelai-2-local-prompt",
			"url": "https://github.com/animerl/novelai-2-local-prompt",
			"description": "Add a button to convert the prompts used in NovelAI for use in the WebUI. In addition, add a button that allows you to recall a previously used prompt.",
			"tags": ["script"]
		},
		{
			"name": "tokenizer",
			"url": "https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer.git",
			"description": "Adds a tab that lets you preview how CLIP model would tokenize your text.",
			"tags": ["script", "tab"]
		},
		{
			"name": "Embeddings editor",
			"url": "https://github.com/CodeExplode/stable-diffusion-webui-embedding-editor.git",
			"description": "Allows you to manually edit textual inversion embeddings using sliders.",
			"tags": ["script", "tab"]
		},
		{
			"name": "embedding-inspector",
			"url": "https://github.com/tkalayci71/embedding-inspector.git",
			"description": "Inspect any token(a word) or Textual-Inversion embeddings and find out which embeddings are similar. You can mix, modify, or create the embeddings in seconds.",
			"tags": ["script", "tab"]
		},
		{
			"name": "Latent Mirroring",
			"url": "https://github.com/dfaker/SD-latent-mirroring",
			"description": "Applies mirroring and flips to the latent images to produce anything from subtle balanced compositions to perfect reflections",
			"tags": ["script"]
		},
		{
			"name": "seed travel",
			"url": "https://github.com/yownas/seed_travel.git",
			"description": "Small script for AUTOMATIC1111/stable-diffusion-webui to create images that exists between seeds.",
			"tags": ["script"]
		},
		{
			"name": "shift-attention",
			"url": "https://github.com/yownas/shift-attention.git",
			"description": "Generate a sequence of images shifting attention in the prompt. This script enables you to give a range to the weight of tokens in a prompt and then generate a sequence of images stepping from the first one to the second.",
			"tags": ["script"]
		},
		{
			"name": "prompt travel",
			"url": "https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel.git",
			"description": "Extension script for AUTOMATIC1111/stable-diffusion-webui to travel between prompts in latent space.",
			"tags": ["script"]
		},
		{
			"name": "Detection Detailer",
			"url": "https://github.com/dustysys/ddetailer.git",
			"description": "An object detection and auto-mask extension for Stable Diffusion web UI.",
			"tags": ["script"]
		},
		{
			"name": "conditioning-highres-fix",
			"url": "https://github.com/klimaleksus/stable-diffusion-webui-conditioning-highres-fix.git",
			"description": "This is Extension for rewriting Inpainting conditioning mask strength value relative to Denoising strength at runtime. This is useful for Inpainting models such as sd-v1-5-inpainting.ckpt",
			"tags": ["script"]
		},
		{
			"name": "Randomize",
			"url": "https://github.com/stysmmaker/stable-diffusion-webui-randomize.git",
			"description": "Allows for random parameters during txt2img generation. This script is processed for all generations, regardless of the script selected, meaning this script will function with others as well, such as AUTOMATIC1111/stable-diffusion-webui-wildcards",
			"tags": ["script"]
		},
		{
			"name": "Auto TLS-HTTPS",
			"url": "https://github.com/papuSpartan/stable-diffusion-webui-auto-tls-https.git",
			"description": "Allows you to easily, or even completely automatically start using HTTPS.",
			"tags": ["script"]
		},
		{
			"name": "DreamArtist",
			"url": "https://github.com/7eu7d7/DreamArtist-sd-webui-extension.git",
			"description": "Towards Controllable One-Shot Text-to-Image Generation via Contrastive Prompt-Tuning.",
			"tags": ["script"]
		},
		{
			"name": "WD 1.4 Tagger",
			"url": "https://github.com/toriato/stable-diffusion-webui-wd14-tagger.git",
			"description": "Interrogates single or multiple image files using various alternative models, similar to deepdanbooru interrogate.",
			"tags": ["script", "tab"]
		},
		{
			"name": "booru2prompt",
			"url": "https://github.com/Malisius/booru2prompt.git",
			"description": "This SD extension allows you to turn posts from various image boorus into stable diffusion prompts. It does so by pulling a list of tags down from their API. You can copy-paste in a link to the post you want yourself, or use the built-in search feature to do it all without leaving SD.",
			"tags": ["script", "tab"]
		},
		{
			"name": "Gelbooru Prompt",
			"url": "https://github.com/antis0007/sd-webui-gelbooru-prompt.git",
			"description": "Extension that gets tags for saved gelbooru images in AUTOMATIC1111's Stable Diffusion webui",
			"tags": ["script"]
		},
		{
			"name": "Merge Board",
			"url": "https://github.com/bbc-mc/sdweb-merge-board.git",
			"description": "Multiple lane merge support(up to 10). Save and Load your merging combination as Recipes, which is simple text.",
			"tags": ["script", "tab"]
		},
		{
			"name": "Depth Maps",
			"url": "https://github.com/thygate/stable-diffusion-webui-depthmap-script.git",
			"description": "Creates depthmaps from the generated images. The result can be viewed on 3D or holographic devices like VR headsets or lookingglass display, used in Render or Game- Engines on a plane with a displacement modifier, and maybe even 3D printed.",
			"tags": ["script"]
		},
		{
			"name": "multi-subject-render",
			"url": "https://github.com/Extraltodeus/multi-subject-render.git",
			"description": "It is a depth aware extension that can help to create multiple complex subjects on a single image. It generates a background, then multiple foreground subjects, cuts their backgrounds after a depth analysis, paste them onto the background and finally does an img2img for a clean finish.",
			"tags": ["script"]
		},
		{
			"name": "depthmap2mask",
			"url": "https://github.com/Extraltodeus/depthmap2mask.git",
			"description": "Create masks for img2img based on a depth estimation made by MiDaS.",
			"tags": ["script"]
		},
		{
			"name": "ABG_extension",
			"url": "https://github.com/KutsuyaYuki/ABG_extension.git",
			"description": "Automatically remove backgrounds. Uses an onnx model fine-tuned for anime images. Runs on GPU.",
			"tags": ["script"]
		},
		{
			"name": "Visualize Cross-Attention",
			"url": "https://github.com/benkyoujouzu/stable-diffusion-webui-visualize-cross-attention-extension.git",
			"description": "Generates highlighted sectors of a submitted input image, based on input prompts. Use with tokenizer extension. See the readme for more info.",
			"tags": ["script", "tab"]
		},
		{
			"name": "DAAM",
			"url": "https://github.com/kousw/stable-diffusion-webui-daam.git",
			"description": "DAAM stands for Diffusion Attentive Attribution Maps. Enter the attention text (must be a string contained in the prompt) and run. An overlapping image with a heatmap for each attention will be generated along with the original image.",
			"tags": ["script"]
		},
		{
			"name": "Prompt Gallery",
			"url": "https://github.com/dr413677671/PromptGallery-stable-diffusion-webui.git",
			"description": "Build a yaml file filled with prompts of your character, hit generate, and quickly preview them by their word attributes and modifiers.",
			"tags": ["script", "tab"]
		},
		{
			"name": "Infinity Grid Generator",
			"url": "https://github.com/mcmonkeyprojects/sd-infinity-grid-generator-script.git",
			"description": "Build a yaml file with your chosen parameters, and generate infinite-dimensional grids. Built-in ability to add description text to fields. See readme for usage details.",
			"tags": ["script"]
		},
		{
			"name": "NSFW checker",
			"url": "https://github.com/AUTOMATIC1111/stable-diffusion-webui-nsfw-censor.git",
			"description": "Replaces NSFW images with black.",
			"tags": ["script"]
		},
		{
			"name": "Diffusion Defender",
			"url": "https://github.com/WildBanjos/DiffusionDefender.git",
			"description": "Prompt blacklist, find and replace, for semi-private and public instances.",
			"tags": ["script"]
		},
		{
			"name": "Config-Presets",
			"url": "https://github.com/Zyin055/Config-Presets.git",
			"description": "Adds a configurable dropdown to allow you to change UI preset settings in the txt2img and img2img tabs.",
			"tags": ["script"]
		},
		{
			"name": "Preset Utilities",
			"url": "https://github.com/Gerschel/sd_web_ui_preset_utils.git",
			"description": "Preset utility tool for ui. Offers compatibility with custom scripts. (to a limit)",
			"tags": ["script"]
		},
{
			"name": "DH Patch",
			"url": "https://github.com/d8ahazard/sd_auto_fix",
			"description": "Random patches by D8ahazard. Auto-load config YAML files for v2, 2.1 models; patch latent-diffusion to fix attention on 2.1 models (black boxes without no-half), whatever else I come up with.",
			"tags": ["script"]
		},
{
			"name": "Riffusion",
			"url": "https://github.com/enlyth/sd-webui-riffusion",
			"description": "Use Riffusion model to produce music in gradio. To replicate original interpolation technique, input the prompt travel extension output frames into the riffusion tab.",
			"tags": ["script", "tab"]
		},
{
			"name": "Save Intermediate Images",
			"url": "https://github.com/AlUlkesh/sd_save_intermediate_images",
			"description": "See PR https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/5464",
			"tags": ["script"]
		},
{
			"name": "openOutpaint extension",
			"url": "https://github.com/zero01101/openOutpaint-webUI-extension",
			"description": "A tab with the full openOutpaint UI. Run with the --api flag.",
			"tags": ["script", "tab"]
		},
{
			"name": "model-keyword",
			"url": "https://github.com/mix1009/model-keyword",
			"description": "Inserts matching keyword(s) to the prompt automatically. Update this extension to get the latest model+keyword mappings.",
			"tags": ["script"]
		},
		{
			"name": "zh_CN Localization",
			"url": "https://github.com/dtlnor/stable-diffusion-webui-localization-zh_CN",
			"description": "Simplified Chinese localization",
			"tags": ["localization"]
		},
		{
			"name": "zh_TW Localization",
			"url": "https://github.com/benlisquare/stable-diffusion-webui-localization-zh_TW",
			"description": "Traditional Chinese localization",
			"tags": ["localization"]
		},
		{
			"name": "ko_KR Localization",
			"url": "https://github.com/36DB/stable-diffusion-webui-localization-ko_KR",
			"description": "Korean localization",
			"tags": ["localization"]
		},
		{
			"name": "es_ES Localization",
			"url": "https://github.com/innovaciones/stable-diffusion-webui-localization-es_ES",
			"description": "Spanish localization",
			"tags": ["localization"]
		},
		{
			"name": "it_IT Localization",
			"url": "https://github.com/Harvester62/stable-diffusion-webui-localization-it_IT",
			"description": "Italian localization",
			"tags": ["localization"]
		},
		{
			"name": "de_DE Localization",
			"url": "https://github.com/Strothis/stable-diffusion-webui-de_DE",
			"description": "German localization",
			"tags": ["localization"]
		},
		{
			"name": "ja_JP Localization",
			"url": "https://github.com/Katsuyuki-Karasawa/stable-diffusion-webui-localization-ja_JP",
			"description": "Japanese localization",
			"tags": ["localization"]
		},
		{
			"name": "pt_BR Localization",
			"url": "https://github.com/M-art-ucci/stable-diffusion-webui-localization-pt_BR",
			"description": "Brazillian portuguese localization",
			"tags": ["localization"]
		},
		{
			"name": "tr_TR Localization",
			"url": "https://github.com/camenduru/stable-diffusion-webui-localization-tr_TR",
			"description": "Turkish localization",
			"tags": ["localization"]
		},
		{
			"name": "no_NO Localization",
			"url": "https://github.com/Cyanz83/stable-diffusion-webui-localization-no_NO",
			"description": "Norwegian localization",
			"tags": ["localization"]
		},
		{
			"name": "ru_RU Localization",
			"url": "https://github.com/ProfaneServitor/stable-diffusion-webui-localization-ru_RU",
			"description": "Russian localization",
			"tags": ["localization"]
		},
		{
			"name": "fi_FI Localization",
			"url": "https://github.com/otsoniemi/stable-diffusion-webui-localization-fi_FI",
			"description": "Finnish localization",
			"tags": ["localization"]
		},
		{
			"name": "old localizations",
			"url": "https://github.com/AUTOMATIC1111/stable-diffusion-webui-old-localizations.git",
			"description": "Old unmaintained localizations that used to be a part of main repository",
			"tags": ["localization"]
		}
	]
}