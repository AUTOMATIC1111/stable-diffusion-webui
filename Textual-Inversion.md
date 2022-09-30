# What is Textual Inversion?
Textual Inversion allows you to train a tiny part of the neural network on your own pictures, and use results when generating new ones.

The result of training is a .pt or a .bin file (former is the format used by original author, latter is by the diffusers library).

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
