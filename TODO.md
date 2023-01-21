# Publish extensions

- `sd-extension-aesthetic-scorer`
- `sd-extension-steps-animation`
- `sd-extension-system-info`

# Investigating

Things I'm actively investigating...

## Embeddings

Need to study more to determine best out-of-the-box settings:
- Train object
- Train style
- Impact of Hires fix:
  - e.g 25 steps and denoising strength 0.25-0.7
- Impact of non-square target resolution

## Hypernetworks

## LORA


# Ideas

Things I'm looking into...

## Automatic

Stuff to be fixed...

- Reconnect WebUI

Tech that can be integrated as part of the core workflow...

- Custom watermark
- [Embedding mixing](https://github.com/tkalayci71/embedding-inspector)
- [DAAM](https://github.com/kousw/stable-diffusion-webui-daam)
- [Merge without distortion](https://github.com/ogkalu2/Merge-Stable-Diffusion-models-without-distortion)
- [Weighted merges](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui/tree/master)
- [Prune models](https://github.com/Akegarasu/sd-webui-model-converter)
- [Use scripts from API](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/6469)
- [Aesthetic gradients](https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients)
- [Face swap](https://github.com/kex0/batch-face-swap)
- [Custom diffusion](https://github.com/guaneec/custom-diffusion-webui)
- [LORA](https://github.com/cloneofsimo/lora)
  - <https://github.com/kohya-ss/sd-webui-additional-networks>
  - <https://github.com/kohya-ss/sd-scripts>
- [Hypernetworks](https://civitai.com/models/4086/luisap-tutorial-hypernetwork-monkeypatch-method)
  - <https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/2670#discussioncomment-4372336>
  - <https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/2670#discussioncomment-4582025>

## Video Generation

- [Deforum](https://github.com/deforum-art/deforum-for-automatic1111-webui)
- [Latent blending](https://github.com/lunarring/latentblending/)
- [VIDM: video implicit diffusion models](https://github.com/MKFMIKU/vidm)
- [Tune-a-Video](https://github.com/showlab/Tune-A-Video)
- [Animator extension](https://github.com/Animator-Anon/animator_extension)
- [Prompt travel](https://github.com/Kahsolt/stable-diffusion-webui-prompt-travel)
- [KLMC2 animation](https://colab.research.google.com/github/dmarx/notebooks/blob/main/Stable_Diffusion_KLMC2_Animation.ipynb) 
- [BOAAB-limit animation](https://colab.research.google.com/drive/17kesyBVqubV_Zzchf2XoR-7MHk5jxTuo?usp=sharing) <https://www.ajayjain.net/journey>
- [Disco diffusion](https://colab.research.google.com/github/alembics/disco-diffusion/blob/main/Disco_Diffusion.ipynb)
- [Video killed the radio star](https://colab.research.google.com/github/dmarx/video-killed-the-radio-star/blob/main/Video_Killed_The_Radio_Star_Defusion.ipynb)
- [Seed travel](https://github.com/yownas/seed_travel)
- [Google frame interpolation](https://github.com/google-research/frame-interpolation)

## Experimental

Cool stuff that is not integrated anywhere...

- [TensorRT](https://www.photoroom.com/tech/stable-diffusion-25-percent-faster-and-save-seconds/)
- Bunch of stuff:<https://pharmapsychotic.com/tools.html>
- Prevalent colors to interrogate
- Auto-Sort inputs by face recognition

## Lee

- Impact of non-square image sizes
- Model vs embedding
- Monitor MEM when switching models
- Emulate model shoot using img2img
