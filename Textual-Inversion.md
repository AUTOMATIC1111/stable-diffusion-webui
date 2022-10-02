# What is Textual Inversion?
Textual Inversion allows you to train a tiny part of the neural network on your own pictures, and use results when generating new ones.

The result of training is a .pt or a .bin file (former is the format used by original author, latter is by the diffusers library).

See original site for more details about what textual inversion is: https://textual-inversion.github.io/.

# Using pre-trained embeddings
Put the embedding into the `embeddings` directory and use its filename in the prompt. You don't have to restart the program for this to work.

As an example, here is an embedding of [Usada Pekora](https://drive.google.com/file/d/1MDSmzSbzkIcw5_aw_i79xfO3CRWQDl-8/view?usp=sharing) I trained on WD1.2 model, on 53 pictures (119 augmented) for 19500 steps, with 8 vectors per token setting.

Pictures it generates:
![grid-0037](https://user-images.githubusercontent.com/20920490/193285043-5d5d57d8-7b5e-4803-a211-5ca5220c35f4.png)
```
portrait of usada pekora
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 4077357776, Size: 512x512, Model hash: 45dee52b
```

You can combine multiple embeddings in one prompt:
![grid-0038](https://user-images.githubusercontent.com/20920490/193285265-a5224378-4ae2-48bf-ad7d-e79a9f998f9c.png)
```
portrait of usada pekora, mignon
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 4077357776, Size: 512x512, Model hash: 45dee52b
```

Be very careful about which model you are using with your embeddings: they work well with the model you used during training, and not so well on different models. For example, here is the above embedding and vanilla 1.4 stable diffusion model:
![grid-0036](https://user-images.githubusercontent.com/20920490/193285611-486373f2-35d0-437c-895a-71454564a7c4.png)
```
portrait of usada pekora
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 4077357776, Size: 512x512, Model hash: 7460a6fa
```

# Training embeddings
## Textual inversion tab
Experimental support for training embeddings in user interface.
- create a new empty embedding, select directory with images, train the embedding on it
- the feature is very raw, use at own risk
- i was able to reproduce results I got with other repos in training anime artists as styles, after few tens of thousands steps
- works with half precision floats, but needs experimentation to see if results will be just as good
- if you have enough memory, safer to run with `--no-half --precision full`
- no preprocessing is done for images (except for resizing to 512x512), not even flip
- planned: another button for UI to run preprocessing for images automatically.
- you can interrupt and resume training without any loss of data (except for AdamW optimization parameters, but it seems none of existing repos save those anyway so the general opinion is they are not important)
- no support for batch sizes or gradient accumulation
- it should not be possible to run this with `--lowvram` and `--medvram` flags.


## Third party repos
I successfully trained embeddings using those repositories:

 - [nicolai256](https://github.com/nicolai256/Stable-textual-inversion_win)
 - [lstein](https://github.com/invoke-ai/InvokeAI)

Other options are to train on colabs and/or using diffusers library, which I know nothing about.


# Finding embeddings online

- [huggingface concepts library](https://huggingface.co/sd-concepts-library) - a lot of different embeddings, but mostly useless.
- [16777216c](https://gitlab.com/16777216c/stable-diffusion-embeddings) - NSFW, anime artist styles by a mysterious stranger.
- [cattoroboto](https://gitlab.com/cattoroboto/waifu-diffusion-embeds) - some anime embeddings by anon.
- [viper1](https://gitgud.io/viper1/stable-diffusion-embeddings) - NSFW, furry girls.
- [anon's embeddings](https://mega.nz/folder/7k0R2arB#5_u6PYfdn-ZS7sRdoecD2A) - NSFW, anime artists.
- [rentry](https://rentry.org/embeddings) - a page with links to embeddings from many sources.
