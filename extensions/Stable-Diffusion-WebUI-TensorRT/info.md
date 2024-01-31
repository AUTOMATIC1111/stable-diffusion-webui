# TensorRT Extension

This extension enables the best performance on NVIDIA RTX GPUs for Stable Diffusion with TensorRT.

## How to use

1. Click on the “Generate Default Engines” button. This step takes 2-10 minutes depending on your GPU. You can generate engines for other combinations.
2. Go to Settings → User Interface → Quick Settings List, add sd_unet. Apply these settings, then reload the UI.
3. Back in the main UI, select “Automatic” from the sd_unet dropdown menu at the top of the page if not already selected.
4. You can now start generating images accelerated by TRT. If you need to create more Engines, go to the TensorRT tab.

Happy prompting!

### LoRA

To use LoRA / LyCORIS checkpoints they first need to be converted to a TensorRT format. This can be done in the TensorRT extension in the Export LoRA tab.
1. Select a LoRA checkpoint from the dropdown.
2. Export. (This will not generate an engine but only convert the weights in ~20s)
3. You can use the exported LoRAs as usual using the prompt embedding.


## More Information

TensorRT uses optimized engines for specific resolutions and batch sizes. You can generate as many optimized engines as desired. Types:
- The "Export Default Engines” selection adds support for resolutions between `512 x 512` and 768x768 for Stable Diffusion 1.5 and 2.1 with batch sizes 1 to 4. For SDXL, this selection generates an engine supporting a resolution of `1024 x 1024` with a batch size of `1`.
- Static engines support a single specific output resolution and batch size.
- Dynamic engines support a range of resolutions and batch sizes, at a small cost in performance. Wider ranges will use more VRAM.
- The first time generating an engine for a checkpoint will take longer. Additional engines generated for the same checkpoint will be much faster. 

Each preset can be adjusted with the “Advanced Settings” option. More detailed instructions can be found [here](https://nvidia.custhelp.com/app/answers/detail/a_id/5487/~/tensorrt-extension-for-stable-diffusion-web-ui).

For more information, please visit the TensorRT Extension GitHub page [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui-tensorrt).
