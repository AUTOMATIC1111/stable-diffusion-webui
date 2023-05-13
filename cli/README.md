# Stable-Diffusion Productivity Scripts

Note: All scripts have built-in `--help` parameter that can be used to get more information

<br>

## Main Scripts

### Generate

Text-to-image with all of the possible parameters  
Supports upsampling, face restoration and grid creation  
> python generate.py

By default uses parameters from  `generate.json`

Parameters that are not specified will be randomized:

- Prompt will be dynamically created from template of random samples: `random.json`
- Sampler/Scheduler will be randomly picked from available ones
- CFG Scale set to 5-10

### Train

Combined pipeline for **embeddings**, **lora**, **lycoris**, **dreambooth** and **hypernetwork**  
Optionally runs several image processing steps before training:

- keep original image
- detect and extract face
- detect and extract body
- detect blur
- detect dynamic range
- attempt to upscale low resolution images
- attempt to restore quality of low quality images
- automatically generate captions using interrogate
- resize image
- square image
- run image segmentation to remove background

> python train.py

<br>

## Auxiliary Scripts

### Benchmark

> python run-benchmark.py

### Create Previews

Create previews for **embeddings**, **lora**, **lycoris**, **dreambooth** and **hypernetwork**

> python create-previews.py

## Image Grid

> python image-grid.py

### Image Watermark

Create invisible image watermark and remove existing EXIF tags  

> python image-watermark.py

### Image Interrogate

Runs CLiP and Booru image interrogation  

> python image-interrogate.py

### Palette Extract

Extract color palette from image(s)  

> python image-palette.py

### Prompt Ideas

Generate complex prompt ideas

> python prompt-ideas.py

### Prompt Promptist

Attempts to beautify the provided prompt  

> python prompt-promptist.py

### Video Extract

Extract frames from video files  

> python video-extract.py

<br>

## Utility Scripts

### SDAPI

Utility module that handles async communication to Automatic API endpoints  
Note: Requires SD API  

Can be used to manually execute specific commands:
> python sdapi.py progress  
> python sdapi.py interrupt
> python sdapi.py shutdown
