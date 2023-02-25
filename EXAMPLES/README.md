BASIC SETTINGS

positive prompt: what you want in your image
negative prompt: what you don't want in your image


Sampling method: I still DK what this is. Just use Default: DPM++ SDE Karras

Sampling steps: How many denoising steps used in the diffusion process. Just leave at 20-30 steps default unless you want to experiment

Seed: Default -1 (meaning you use a random new seed every generation). The random noise seed that you can use as a base for diffusion, think of it as just an image with random RGB values at every coordinate pixel, this random noise image will slowly be diffused into an artwork. Different seeds produce different images.

Batch count: How many images you want to generate. Default 1

Batch size: Leave this at 1. This will create more images also but it will eat up ALOT of VRAM. if you set too high you will get "CUDA OUT OF MEMORY ERROR"

CFG scale: Default: 8. how much you want your image to conform with your prompt - the high the number the more it will follow the prompt

Width & Height: Default 512x512. Output image dimensions. Normally set at least one side to a length between 512 & 768, the other side can be larger. If the resolution is too high, distortion and artifacts will appear in the image.

Hires. fix: A fix to the previous problem of distortion and artifacts. able to latently upscale images by x2 (VERY GOOD, but takes longer to generate image)
Default settings for hires. fix: 
Upscaler: Just use "Latent (nearest-exact)". I feel this is the best upscaler, unless you want to experiment with the other options.
Hires steps: Default 0 (This means it will follow the same number of steps from the Sampling steps eg. 20)
Upscale by: Default 2 (512x512 image will become 1024x1024) (setting this value too low or too high will cause distortions)


FILEPATHS

Launch file (double click) - stable-diffusion-webui\webui-user.bat
Stable Diffusion models - stable-diffusion-webui\models\Stable-diffusion
Lora models - stable-diffusion-webui\models\Lora
Output Images - stable-diffusion-webui\Outputs
Styles (prompt templates) - stable-diffusion-webui\styles.csv


