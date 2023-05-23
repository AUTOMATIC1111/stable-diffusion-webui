Making your own inpainting model is very simple:
1. Go to Checkpoint Merger
2. Select "Add Difference"
3. Set "Multiplier" to 1.0
4. Set "A" to the official inpaint model ([SD-v1.5-Inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting/tree/main))
5. Set "B" to your model
6. Set "C" to the standard base model ([SD-v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main))
7. Set name as whatever you want, probably (your model)_inpainting
8. Set other values as preferred, ie probably select "Safetensors" and "Save as float16"
9. Hit merge
10. Use your new model in img2img inpainting tab

The way this works is it literally just takes the inpainting model, and copies over your model's unique data to it.
Notice that the formula is A + (B - C), which you can interpret as equivalent to (A - C) + B. Because 'A' is 1.5-inpaint and 'C' is 1.5, A - C is inpainting logic and nothing more. so the formula is (Inpainting logic) + (Your Model).

![screenshot](https://github.com/AUTOMATIC1111/stable-diffusion-webui/assets/40751091/4bdbab38-9237-48ea-9698-a036a5c96585)
