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
- Settings params on updates install  

Tech that can be integrated as part of the core workflow...

- [Embedding mixing](https://github.com/tkalayci71/embedding-inspector)
- [Merge without distortion](https://github.com/ogkalu2/Merge-Stable-Diffusion-models-without-distortion)
- [Weighted merges](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui/tree/master)
- [Use scripts from API](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/6469)
- [Face swap](https://github.com/kex0/batch-face-swap)
- [Custom diffusion](https://github.com/guaneec/custom-diffusion-webui)
- [Null-text inversion](https://github.com/ouhenio/null-text-inversion-colab)
- [Custom diffusion](https://github.com/guaneec/custom-diffusion-webui)
  - <https://www.cs.cmu.edu/~custom-diffusion/>
- [Dream artist](https://github.com/7eu7d7/DreamArtist-sd-webui-extension)
- [ControlNet](https://github.com/lllyasviel/ControlNet)

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
- [Prompt fusion](https://github.com/ljleb/prompt-fusion-extension)

## Experimental

Cool stuff that is not integrated anywhere...

- [TensorRT](https://www.photoroom.com/tech/stable-diffusion-25-percent-faster-and-save-seconds/)
- [GIT](https://huggingface.co/microsoft/git-large-textcaps)
- Bunch of stuff:<https://pharmapsychotic.com/tools.html>
