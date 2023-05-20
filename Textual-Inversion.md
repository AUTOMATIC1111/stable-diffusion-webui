# What is Textual Inversion?

[Textual inversion](https://textual-inversion.github.io/): Teach the base model **new vocabulary about a particular concept** with a couple of images reflecting that concept.
* The concept can be: a pose, an artistic style, a texture, etc.
   * The concept doesn't have to actually exist in the real world. For example, you might have seen many generated images whose negative prompt (np) contained the tag "EasyNegative". That's an artificial concept trained on a bunch of images _someone_ thought of poor quality.
* It **doesn't enrich the model**. If your base model is trained solely on images of _apples_, and you tried to teach the model the word _"banana"_ with ~20 images of bananas, then -- at best -- your model will give you long, yellow apples when you ask for a banana. (Of course, you can get the model to approximate a banana with apples with 1,000+ images, but is it really worth it? ;) )

The result of the training is a `.pt` or a `.bin` file (former is the format used by original author, latter is by the [diffusers][dl] library) with the embedding in it. These files can be shared to other generative artists.

[dl]: https://huggingface.co/docs/diffusers/index

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
- Section for UI to run preprocessing for images automatically.
- you can interrupt and resume training without any loss of data (except for AdamW optimization parameters, but it seems none of existing repos save those anyway so the general opinion is they are not important)
- no support for batch sizes or gradient accumulation
- it should not be possible to run this with `--lowvram` and `--medvram` flags.

## Explanation for parameters

### Creating an embedding
- **Name**: filename for the created embedding. You will also use this text in prompts when referring to the embedding.
- **Initialization text**: the embedding you create will initially be filled with vectors of this text. If you create a one vector embedding named "zzzz1234" with "tree" as initialization text, and use it in prompt without training, then prompt "a zzzz1234 by monet" will produce same pictures as "a tree by monet".
- **Number of vectors per token**: the size of embedding. The larger this value, the more information about subject you can fit into the embedding, but also the more words it will take away from your prompt allowance. With stable diffusion, you have a limit of 75 tokens in the prompt. If you use an embedding with 16 vectors in a prompt, that will leave you with space for 75 - 16 = 59. Also from my experience, the larger the number of vectors, the more pictures you need to obtain good results.

### Preprocess
This takes images from a directory, processes them to be ready for textual inversion, and writes results to another directory. This is a convenience feature and you can preprocess pictures yourself if you wish.
- **Source directory**: directory with images
- **Destination directory**: directory where the results will be written
- **Create flipped copies**: for each image, also write its mirrored copy
- **Split oversized images into two**: if the image is too tall or wide, resize it to have the short side match the desired resolution, and create two, possibly intersecting pictures out of it.
- **Use BLIP caption as filename**: use BLIP model from the interrogator to add a caption to the filename.

### Training an embedding
- **Embedding**: select the embedding you want to train from this dropdown.
- **Learning rate**: how fast should the training go. The danger of setting this parameter to a high value is that you may break the embedding if you set it too high. If you see `Loss: nan` in the training info textbox, that means you failed and the embedding is dead. With the default value, this should not happen. It's possible to specify multiple learning rates in this setting using the following syntax: `0.005:100, 1e-3:1000, 1e-5` - this will train with lr of `0.005` for first 100 steps, then `1e-3` until 1000 steps, then `1e-5` until the end.
- **Dataset directory**: directory with images for training. They all must be square.
- **Log directory**: sample images and copies of partially trained embeddings will be written to this directory.
- **Prompt template file**: text file with prompts, one per line, for training the model on. See files in directory `textual_inversion_templates` for what you can do with those. Use `style.txt` when training styles, and `subject.txt` when training object embeddings. Following tags can be used in the file:
  - `[name]`: the name of embedding
  - `[filewords]`: words from the file name of the image from the dataset. See below for more info.
- **Max steps**: training will stop after this many steps have been completed. A step is when one picture (or one batch of pictures, but batches are currently not supported) is shown to the model and is used to improve embedding. if you interrupt training and resume it at a later date, the number of steps is preserved.
- **Save images with embedding in PNG chunks**: every time an image is generated it is combined with the most recently logged embedding and saved to image_embeddings in a format that can be both shared as an image, and placed into your embeddings folder and loaded.
- **Preview prompt**: if not empty, this prompt will be used to generate preview pictures. If empty, the prompt from training will be used.

### filewords
`[filewords]` is a tag for prompt template file that allows you to insert text from filename into the prompt. By default, file's extension is removed, as well as all numbers and dashes (`-`) at the start of filename. So this filename: `000001-1-a man in suit.png` will become this text for prompt: `a man in suit`. Formatting of the text in the filename is left as it is.

It's possible to use options `Filename word regex` and `Filename join string` to alter the text from filename: for example, with word regex = `\w+` and join string = `, `, the file from above will produce this text: `a, man, in, suit`. regex is used to extract words from text (and they are `['a', 'man', 'in', 'suit', ]`), and join string (', ') is placed between those words to create one text: `a, man, in, suit`.

It's also possible to make a text file with same filename as image (`000001-1-a man in suit.txt`) and just put the prompt text there. The filename and regex options will not be used.

## Third party repos
I successfully trained embeddings using those repositories:

 - [nicolai256](https://github.com/nicolai256/Stable-textual-inversion_win)
 - [lstein](https://github.com/invoke-ai/InvokeAI)

Other options are to train on colabs and/or using diffusers library, which I know nothing about.

# Finding embeddings online

- Github has kindly asked me to remove all the links here.

# Hypernetworks

Hypernetworks is a novel (get it?) concept for fine tuning a model without touching any of its weights.

The current way to train hypernets is in the textual inversion tab.

Training works the same way as with textual inversion.

The only requirement is to use a very, very low learning rate, something like 0.000005 or 0.0000005.

### Dum Dum Guide
An anonymous user has written a guide with pictures for using hypernetworks: https://rentry.org/hypernetwork4dumdums

### Unload VAE and CLIP from VRAM when training
This option on settings tab allows you to save some memoryat the cost of slower preview picture generation.