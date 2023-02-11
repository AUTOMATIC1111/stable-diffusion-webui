# Generation Extensions

## Latent Mirroring
https://github.com/dfaker/SD-latent-mirroring

Applies mirroring and flips to the latent images to produce anything from subtle balanced compositions to perfect reflections

![image](https://user-images.githubusercontent.com/98228077/208331098-3b7fefce-6d38-486d-9543-258f5b2b0fd6.png)

## seed travel
https://github.com/yownas/seed_travel

Small script for AUTOMATIC1111/stable-diffusion-webui to create images that exists between seeds.

<details><summary>Example: (Click to expand:)</summary>
<img src="https://github.com/ClashSAN/bloated-gifs/blob/main/seedtravel.gif" width="512" height="512" />
</details>

## Detection Detailer
https://github.com/dustysys/ddetailer

An object detection and auto-mask extension for Stable Diffusion web UI.

<img src="https://github.com/dustysys/ddetailer/raw/master/misc/ddetailer_example_3.gif"/>

## conditioning-highres-fix
https://github.com/klimaleksus/stable-diffusion-webui-conditioning-highres-fix

This is Extension for rewriting Inpainting conditioning mask strength value relative to Denoising strength at runtime. This is useful for Inpainting models such as sd-v1-5-inpainting.ckpt

![image](https://user-images.githubusercontent.com/98228077/208331374-5a271cf3-cfac-449b-9e09-c63ddc9ca03a.png)

## multi-subject-render
https://github.com/Extraltodeus/multi-subject-render

It is a depth aware extension that can help to create multiple complex subjects on a single image. It generates a background, then multiple foreground subjects, cuts their backgrounds after a depth analysis, paste them onto the background and finally does an img2img for a clean finish.

![image](https://user-images.githubusercontent.com/98228077/208331952-019dfd64-182d-4695-bdb0-4367c81e4c43.png)

## depthmap2mask
https://github.com/Extraltodeus/depthmap2mask

Create masks for img2img based on a depth estimation made by MiDaS.

![image](https://user-images.githubusercontent.com/15731540/204050868-eca8db02-2193-4115-a5b8-e8f5c796e035.png)![image](https://user-images.githubusercontent.com/15731540/204050888-41b00335-50b4-4328-8cfd-8fc5e9cec78b.png)![image](https://user-images.githubusercontent.com/15731540/204050899-8757b774-f2da-4c15-bfa7-90d8270e8287.png)

## Save Intermediate Images
https://github.com/AlUlkesh/sd_save_intermediate_images

Implements saving intermediate images, with more advanced features.

![noisy](https://user-images.githubusercontent.com/98228077/211706803-f747691d-cca8-4692-90ef-f6a2859ed5cb.jpg)
![not](https://user-images.githubusercontent.com/98228077/211706801-fc593dbf-67c4-4983-8a80-c88355ffeba2.jpg)
![image](https://user-images.githubusercontent.com/98228077/217990312-15b4eda2-858a-44b7-91a3-b34af7c487b6.png)

## Enhanced-img2img
https://github.com/OedoSoldier/enhanced-img2img
An extension with support for batched and better inpainting. See [readme](https://github.com/OedoSoldier/enhanced-img2img#usage) for more details.
![image](https://user-images.githubusercontent.com/98228077/217990537-e8bdbc74-7210-4864-8140-a076a342c695.png)

## Kohya-ss Additional Networks
https://github.com/kohya-ss/sd-webui-additional-networks

Allows the Web UI to use networks (LoRA) trained by their scripts to generate images.

## Multiple hypernetworks 
https://github.com/antis0007/sd-webui-multiple-hypernetworks

Extension that allows the use of multiple hypernetworks at once

![image](https://user-images.githubusercontent.com/32306715/212293588-a8b4d1e9-4099-4a2e-a61a-f549a70f6096.png)


## Steps Animation
https://github.com/vladmandic/sd-extension-steps-animation

Extension to create animation sequence from denoised intermediate steps  
Registers a script in **txt2img** and **img2img** tabs

Creating animation has minimum impact on overall performance as it does not require separate runs  
except adding overhead of saving each intermediate step as image plus few seconds to actually create movie file  

Supports **color** and **motion** interpolation to achieve animation of desired duration from any number of interim steps  
Resulting movie fiels are typically very small (*~1MB being average*) due to optimized codec settings  

![screenshot](https://raw.githubusercontent.com/vladmandic/sd-extension-steps-animation/main/steps-animation.jpg)

### [Example](https://user-images.githubusercontent.com/57876960/212490617-f0444799-50e5-485e-bc5d-9c24a9146d38.mp4)

## Instruct-pix2pix
https://github.com/Klace/stable-diffusion-webui-instruct-pix2pix

Adds a tab for doing img2img editing with the instruct-pix2pix model.

## anti-burn
https://github.com/klimaleksus/stable-diffusion-webui-anti-burn

Smoothing generated images by skipping a few very last steps and averaging together some images before them