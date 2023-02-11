# Prompt / Generation parameters Extensions

## Wildcards
https://github.com/AUTOMATIC1111/stable-diffusion-webui-wildcards

Allows you to use `__name__` syntax in your prompt to get a random line from a file named `name.txt` in the wildcards directory.

## Dynamic Prompts
https://github.com/adieyal/sd-dynamic-prompts

A custom extension for [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that implements an expressive template language for random or combinatorial prompt generation along with features to support deep wildcard directory structures.

More features and additions are shown in the [readme](https://github.com/adieyal/sd-dynamic-prompts).

![image](https://github.com/adieyal/sd-dynamic-prompts/raw/main/images/extension.png)

Using this extension, the prompt:

`A {house|apartment|lodge|cottage} in {summer|winter|autumn|spring} by {2$$artist1|artist2|artist3}`

Will any of the following prompts:

- A house in summer by artist1, artist2
- A lodge in autumn by artist3, artist1
- A cottage in winter by artist2, artist3
- ...

This is especially useful if you are searching for interesting combinations of artists and styles.

You can also pick a random string from a file. Assuming you have the file seasons.txt in WILDCARD_DIR (see below), then:

`__seasons__ is coming`

Might generate the following:

- Winter is coming
- Spring is coming
- ...

You can also use the same wildcard twice

`I love __seasons__ better than __seasons__`

- I love Winter better than Summer
- I love Spring better than Spring

## Unprompted
https://github.com/ThereforeGames/unprompted
 
Supercharge your prompt workflow with this powerful scripting language!

![unprompted_header](https://user-images.githubusercontent.com/95403634/199041569-7c6c5748-e7dc-4068-943f-c2d92745dbb5.png)

**Unprompted** is a highly modular extension for [AUTOMATIC1111's Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that allows you to include various shortcodes in your prompts. You can pull text from files, set up your own variables, process text through conditional functions, and so much more - it's like wildcards on steroids.

While the intended usecase is Stable Diffusion, **this engine is also flexible enough to serve as an all-purpose text generator.**

## Booru tag autocompletion
https://github.com/DominikDoom/a1111-sd-webui-tagcomplete

Displays autocompletion hints for tags from "image booru" boards such as Danbooru. Uses local tag CSV files and includes a config for customization.

![image](https://user-images.githubusercontent.com/20920490/200016417-9451efdb-5d0d-4131-bd9e-39a687be8dd7.png)

## novelai-2-local-prompt
https://github.com/animerl/novelai-2-local-prompt

Add a button to convert the prompts used in NovelAI for use in the WebUI. In addition, add a button that allows you to recall a previously used prompt.

![pic](https://user-images.githubusercontent.com/113022648/197382468-65f4a96d-48af-4890-8fcf-0ec7c3b9ec3a.png)

## StylePile
https://github.com/some9000/StylePile
			
An easy way to mix and match elements to prompts that affect the style of the result.

![image](https://user-images.githubusercontent.com/98228077/208331056-2956d050-a7a4-4b6f-b064-72f6a7d7ee0d.png)

## shift-attention
https://github.com/yownas/shift-attention

Generate a sequence of images shifting attention in the prompt. This script enables you to give a range to the weight of tokens in a prompt and then generate a sequence of images stepping from the first one to the second.

https://user-images.githubusercontent.com/13150150/193368939-c0a57440-1955-417c-898a-ccd102e207a5.mp4

## prompt travel
https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel

Extension script for AUTOMATIC1111/stable-diffusion-webui to travel between prompts in latent space.

<details><summary>Example: (Click to expand:)</summary>
<img src="https://github.com/ClashSAN/bloated-gifs/blob/main/prompt_travel.gif" width="512" height="512" />
</details>

## Sonar
https://github.com/Kahsolt/stable-diffusion-webui-sonar

Improve the generated image quality, searches for similar (yet even better!) images in the neighborhood of some known image, focuses on single prompt optimization rather than traveling between multiple prompts.

![image](https://user-images.githubusercontent.com/98228077/209545702-c796a3f8-4d8c-4e2b-9b2e-920008ec2f32.png)![image](https://user-images.githubusercontent.com/98228077/209545756-31c94fec-d783-447f-8aac-4a5bba43ea15.png)

## Randomize
~~https://github.com/stysmmaker/stable-diffusion-webui-randomize~~
fork: https://github.com/innightwolfsleep/stable-diffusion-webui-randomize

Allows for random parameters during txt2img generation. This script is processed for all generations, regardless of the script selected, meaning this script will function with others as well, such as AUTOMATIC1111/stable-diffusion-webui-wildcards.

## booru2prompt
https://github.com/Malisius/booru2prompt

This SD extension allows you to turn posts from various image boorus into stable diffusion prompts. It does so by pulling a list of tags down from their API. You can copy-paste in a link to the post you want yourself, or use the built-in search feature to do it all without leaving SD.

![image](https://user-images.githubusercontent.com/98228077/208331612-dad61ef7-33dd-4008-9cc7-06b0b0a7cb6d.png)

also see:\
https://github.com/stysmmaker/stable-diffusion-webui-booru-prompt

## gelbooru-prompt
https://github.com/antis0007/sd-webui-gelbooru-prompt

Fetch tags using your image's hash.

## Infinity Grid Generator
https://github.com/mcmonkeyprojects/sd-infinity-grid-generator-script

Build a yaml file with your chosen parameters, and generate infinite-dimensional grids. Built-in ability to add description text to fields. See readme for usage details.

![image](https://user-images.githubusercontent.com/98228077/208332269-88983668-ea7e-45a8-a6d5-cd7a9cb64b3a.png)

## Diffusion Defender
https://github.com/WildBanjos/DiffusionDefender

Prompt blacklist, find and replace, for semi-private and public instances.

## Config-Presets
https://github.com/Zyin055/Config-Presets

Adds a configurable dropdown to allow you to change UI preset settings in the txt2img and img2img tabs.

![image](https://user-images.githubusercontent.com/98228077/208332322-24339554-0274-4add-88a7-d33bba1e3823.png)

## Riffusion
https://github.com/enlyth/sd-webui-riffusion

Use Riffusion model to produce music in gradio. To replicate [original](https://www.riffusion.com/about) interpolation technique, input the [prompt travel extension](https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel) output frames into the riffusion tab.

![image](https://user-images.githubusercontent.com/98228077/209539460-f5c23891-b5e6-46c7-b1a5-b7440a3f031b.png)![image](https://user-images.githubusercontent.com/98228077/209539472-031e623e-f7a2-4da9-9711-8bf73d0cfe6e.png)

## model-keyword
https://github.com/mix1009/model-keyword

Inserts matching keyword(s) to the prompt automatically. Update extension to get the latest model+keyword mappings.

![image](https://user-images.githubusercontent.com/98228077/209717531-e0ae74ab-b753-4ad1-99b2-e1eda3de5433.png)

## Prompt Generator
https://github.com/imrayya/stable-diffusion-webui-Prompt_Generator

Adds a tab to the webui that allows the user to generate a prompt from a small base prompt. Based on [FredZhang7/distilgpt2-stable-diffusion-v2](https://huggingface.co/FredZhang7/distilgpt2-stable-diffusion-v2).

![image](https://user-images.githubusercontent.com/98228077/210076951-36f5d90c-b8c4-4b12-b909-582193deeec1.png)

## Promptgen
https://github.com/AUTOMATIC1111/stable-diffusion-webui-promptgen

Use transformers models to generate prompts.
			
## Fusion
https://github.com/ljleb/prompt-fusion-extension

Adds prompt-travel and shift-attention-like interpolations (see exts), but during/within the sampling steps. Always-on + works w/ existing prompt-editing syntax. Various interpolation modes. See their wiki for more info.

## Prompt Translator
https://github.com/butaixianran/Stable-Diffusion-Webui-Prompt-Translator

A integrated translator for Translate prompt to English using Deepl or Baidu

