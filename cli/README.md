# Stable-Diffusion Productivity Scripts

*Notes*:
- Offline scripts can be used with or without **Automatic WebUI**
- Online scripts rely on **Automatic WebUI** API which should be started with `--api` parameter
- All scripts have built-in `--help` parameter that can be used to get more information

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

End-to-end embedding training
> python train.py

Combined pipeline:
1. Creates embedding  
2. Extracts images if input is movie  
3. Preprocesses images  
4. Runs training  

<br>

## Auxiliary Scripts

### Benchmark

Benchmark your **Automatic WebUI**
> python modules/bench.py

### Embedding Previews

Create previews of embeddings using preview templates
> python modules/embedding-preview.py

## Grid

Create flexible image grids from any number of images
> python modiles/grid.py

### Image Watermark

Create invisible image watermark and remove existing EXIF tags
> python modules/image-watermark.py
### Interrogate

Runs CLiP and Booru image interrogation
> python modules/interrogate.py

### Models Previews

Create previews of models using built-in templates
> python modules/models-preview.py

### Palette Extract

Extract color palette from image(s)
> python modules/palette-extract.py

### Image Process

Run image processing to extract face/body segments and run resolution/blur/dynamic-range checks
> python modules/process.py

### Prompt Ideas

Generate complex prompt ideas
> python modules/prompt-ideas.py

### Prompt Promptist

Attempts to beautify the provided prompt  
> python modules/promptist.py

### Training Loss-Chart

Create loss-chart from training log
> python modules/train-losschart.py

### Training Loss-Rate

Create customizable loss rate to be used in training
> python modules/train-lossrate.py

### Video Extract

Extract frames from video files
> python modules/video-extract.py

<br>

## Utility Scripts
### SDAPI

Utility module that handles async communication to Automatic API endpoints  
Can be used to manually execute specific commands:
> python sdapi.py progress  
> python sdapi.py interrupt
