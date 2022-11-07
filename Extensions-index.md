{
	"about": "This file is used by Web UI to show the index of available extensions. It's in JSON format and is not meant to be viewed by users directly. If you edit the file you must ensure that it's still a valid JSON.",
	"tags": {
		"script": "a general extension that adds functionality",
		"localization": "a localization extension that translates web ui into another language",
		"tab": "adds a tab",
		"ads": "contains ads"
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
			"tags": ["script"]
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
			"description": "Lets you edit captions in training datasets.",
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
			"description": "Allows you to include various shortcodes in your prompts. You can pull text from files, set up your own variables, process text through conditional functions, and so much more - it's like wildcards on steroids.",
			"tags": ["script", "ads"]
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
			"name": "Latent Mirroring",
			"url": "https://github.com/dfaker/SD-latent-mirroring",
			"description": "Applies mirroring and flips to the latent images to produce anything from subtle balanced compositions to perfect reflections",
			"tags": ["script"]
		},
		{
			"name": "Embeddings editor",
			"url": "https://github.com/CodeExplode/stable-diffusion-webui-embedding-editor.git",
			"description": "Allows you to manually edit textual inversion embeddings using sliders.",
			"tags": ["script", "tab"]
		},
		{
			"name": "zh_CN localization",
			"url": "https://github.com/dtlnor/stable-diffusion-webui-localization-zh_CN",
			"description": "Provide Simplified Chinese localization for the WebUI",
			"tags": ["localization"]
		},
		{
			"name": "ko_KR Localization",
			"url": "https://github.com/36DB/stable-diffusion-webui-kr",
			"description": "Provide Korean localization for the WebUI",
			"tags": ["localization"]
		},
		{
			"name": "it_IT Localization",
			"url": "https://github.com/Harvester62/stable-diffusion-webui-localization-it_IT",
			"description": "Provide Italian localization for the WebUI",
			"tags": ["localization"]
		},
		{
			"name": "de_DE Localization",
			"url": "https://github.com/Strothis/stable-diffusion-webui-de_DE",
			"description": "Provide German localization for the WebUI",
			"tags": ["localization"]
		}
	]
}
