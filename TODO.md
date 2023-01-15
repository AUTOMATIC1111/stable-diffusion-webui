# Merge repos

- Move sd-extensions/scripts/save-steps-animation script to automatic/extensions-builtin
- Move sd-extensions/api to automatic/cli
- [https://github.com/vladmandic/automatic]
- [https://github.com/vladmandic/sd-extensions]
- [https://github.com/vladmandic/generative-art]

# Investigating

Things I'm actively investigating...

## Embeddings

Need to study more to determine best out-of-the-box settings:
- Experiment with multiple init words: same as number of vectors
  - --init person,woman,girl,model
- Impact of Codeformer
- Impact of Hires fix:
  - e.g 25 steps and denoising strength 0.25-0.7
- Possible use of Gradient accumulation
  - num-images / batch-size
- Different learning-rates
  - monitor convergence in chart train.png

# Ideas

Things I'm looking into...

# Automatic

Stuff to be fixed...

- Torch 2.0 model compile
- Add DeepSpeed support
- Reconnect WebUI
- Model reload performance

Tech that can be integrated as part of the core workflow...

- Custom watermark
- [Embedding mixing](https://github.com/tkalayci71/embedding-inspector)
- [DAAM](https://github.com/kousw/stable-diffusion-webui-daam)
- [Merge without distortion](https://github.com/ogkalu2/Merge-Stable-Diffusion-models-without-distortion)
- [Weighted merges](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui/tree/master)
- [Prune models](https://github.com/Akegarasu/sd-webui-model-converter)
- [Use scripts from API](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/6469)
- [Aesthetic Gradients](https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients)
- [Latent blending](https://github.com/lunarring/latentblending/)
- [Face swap](https://github.com/kex0/batch-face-swap)
- [Animator extension](https://github.com/Animator-Anon/animator_extension)
- [LORA](https://github.com/cloneofsimo/lora)
  - <https://github.com/kohya-ss/sd-webui-additional-networks>
  - <https://github.com/kohya-ss/sd-scripts>
- [Hypernetworks](https://civitai.com/models/4086/luisap-tutorial-hypernetwork-monkeypatch-method)
  - <https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/2670#discussioncomment-4372336>
  - <https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/2670#discussioncomment-4582025>

# Experimental

Cool stuff that is not integrated anywhere...

- Bunch of stuff:<https://pharmapsychotic.com/tools.html>
- [TensorRT](https://www.photoroom.com/tech/stable-diffusion-25-percent-faster-and-save-seconds/)
- [KLMC2 Animation](https://colab.research.google.com/github/dmarx/notebooks/blob/main/Stable_Diffusion_KLMC2_Animation.ipynb)
- [Disco Diffusion](https://colab.research.google.com/github/alembics/disco-diffusion/blob/main/Disco_Diffusion.ipynb)
- [Video Killed the Radio Star](https://colab.research.google.com/github/dmarx/video-killed-the-radio-star/blob/main/Video_Killed_The_Radio_Star_Defusion.ipynb)

# SDAPI

- Support TLS
- Support auth

# External

Ideas that can be value-added to core tech...

- Prevalent colors to interrogate
- Auto-Sort inputs by face recognition
- Use semantic segmentation to remove background from inputs
- Auto-filter training inputs based on blur  
  <https://pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/>

/home/vlado/dev/sd-extensions/api/losschart.py:48: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`). Consider using `matplotlib.pyplot.close()`.