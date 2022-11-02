# General info

Extensions are a more convenient form of user scripts.

Extensions all exist in their own subdirectory inside the `extensions` directory. You can use git to install an extension like this:

```
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients extensions/aesthetic-gradients
```

This installs an extension from `https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients` into the `extensions/aesthetic-gradients` directory.

Alternatively you can just copy-paste a directory into `extensions`.

For developing extensions, see [Developing extensions](Developing-extensions).

# Extensions

## Aesthetic Gradients
https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients

Create an embedding from one or few pictures and use it to apply their style to generated images.

![firefox_FgKg9dx9eF](https://user-images.githubusercontent.com/20920490/197466300-6b042bcf-5cba-4600-97d7-ad2652875706.png)

## Wildcards
https://github.com/AUTOMATIC1111/stable-diffusion-webui-wildcards

Allows you to use `__name__` syntax in your prompt to get a random line from a file named `name.txt` in the wildcards directory.

## Dynamic Prompts
https://github.com/adieyal/sd-dynamic-prompts

A custom extension for [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that implements an expressive template language for random or combinatorial prompt generation along with features to support deep wildcard directory structures.

Using this script, the prompt:

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

To install, run the following command from within the webui directory:

`git clone https://github.com/adieyal/sd-dynamic-prompting/ extensions/dynamic-prompts`

If you are upgrading from a version prior to 0.11.0, be sure to delete the old dynamic_prompting.py from the webui's scripts directory and the old dynamic_prompting.js from the webui's javascript directory.

## Image browser
https://github.com/yfszzx/stable-diffusion-webui-images-browser

Provides an interface to browse created images in the web browser.

![68747470733a2f2f73362e6a70672e636d2f323032322f31302f32342f504a6a755a742e706e67](https://user-images.githubusercontent.com/20920490/197518762-a23f3e34-f174-4275-8283-eb8d2ff65ef2.png)

## Inspiration
https://github.com/yfszzx/stable-diffusion-webui-inspiration

Randomly display the pictures of the artist's or artistic genres typical style, more pictures of this artist or genre is displayed after selecting. So you don't have to worry about how hard it is to choose the right style of art when you create.

![68747470733a2f2f73362e6a70672e636d2f323032322f31302f32322f504a596f4e4c2e706e67](https://user-images.githubusercontent.com/20920490/197518700-3f753132-8799-4ad0-8cdf-bcdcbf7798aa.png)

## Deforum
https://github.com/deforum-art/deforum-for-automatic1111-webui


The official port of Deforum, an extensive script for 2D and 3D animations, supporting keyframable sequences, dynamic math parameters (even inside the prompts), dynamic masking, depth estimation and warping.

![ui](https://user-images.githubusercontent.com/20920490/197619558-c088a329-3672-4f0a-8685-cf539996ad1e.png)

## Artists to study
https://github.com/camenduru/stable-diffusion-webui-artists-to-study

https://artiststostudy.pages.dev/ adapted to an extension for [web ui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

To install it, clone the repo into the `extensions` directory and restart the web ui:

`git clone https://github.com/camenduru/stable-diffusion-webui-artists-to-study`

You can add the artist name to the clipboard by clicking on it. (thanks for the idea @gmaciocci)

![picture](https://user-images.githubusercontent.com/54370274/197829512-e7d30d44-2697-4ecd-b9a7-3665217918c7.jpg)

## Aesthetic Image Scorer
https://github.com/tsngo/stable-diffusion-webui-aesthetic-image-scorer

Extension for https://github.com/AUTOMATIC1111/stable-diffusion-webui

Calculates aesthetic score for generated images using [CLIP+MLP Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) based on [Chad Scorer](https://github.com/grexzen/SD-Chad/blob/main/chad_scorer.py)

See [Discussions](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/1831)

Saves score to windows tags with other options planned

![picture](https://github.com/tsngo/stable-diffusion-webui-aesthetic-image-scorer/blob/main/tag_group_by.png)

## Dataset Tag Editor
https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor

[日本語 Readme](https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor/blob/main/README-JP.md)

This is an extension to edit captions in training dataset for [Stable Diffusion web UI by AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

It works well with text captions in comma-separated style (such as the tags generated by DeepBooru interrogator).

Caption in the filenames of images can be loaded, but edited captions can only be saved in the form of text files.

![picture](https://github.com/toshiaki1729/stable-diffusion-webui-dataset-tag-editor/blob/main/ss01.png)

## auto-sd-paint-ext

https://github.com/Interpause/auto-sd-paint-ext

Formerly known as `auto-sd-krita`.

>Extension for AUTOMATIC1111's webUI with Krita Plugin (other drawing studios soon?)

Outdated demo | New UI (TODO: demo image)
--- | ---
![demo image](https://user-images.githubusercontent.com/42513874/194701722-e7a3f7eb-be4a-4f43-93a5-480835c9260f.jpg) | ![demo image 2](https://user-images.githubusercontent.com/42513874/199507299-66729f9b-3581-43a3-b5f4-57eb90b8f981.png)

### Differences

- UI no longer freezes during image update
- Inpainting layer no longer has to be manually hidden, nor use white specifically
- UI has been improved & squeezed further
- Scripts API is now possible

## training-picker
https://github.com/Maurdekye/training-picker

Adds a tab to the webui that allows the user to automatically extract keyframes from video, and manually extract 512x512 crops of those frames for use in model training.

![image](https://user-images.githubusercontent.com/2313721/199614791-1f573573-a2e2-4358-836d-5655825077e1.png)

### Installation

- Install [AUTOMATIC1111's Stable Diffusion Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- Install [ffmpeg](https://ffmpeg.org/) for your operating system
- Clone this repository into the extensions folder inside the webui
- Drop videos you want to extract cropped frames from into the training-picker/videos folder