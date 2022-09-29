Negative prompt is a way to use the Stable Diffusion in a way that allows the user to specify what he doesn't want to see, without any extra load or requirements for the model. As far as I know, I was the first to use this approach; the commit that adds it is [757bb7c4](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/757bb7c46b20651853ee23e3109ac4f9fb06a061). The feature has found extreme popularity among users who remove the usual deformities of Stable Diffusion like extra limbs with it. In addition to just being able to specify what you don't want to see, which sometimes is possible via usual prompt, and sometimes isn't, this allows you to do that without using any of your allowance of 75 tokens the prompt consists of.

The way negative prompt works is by using user-specified text instead of empty string for `unconditional_conditioning` when doing sampling.

Here's the (simplified) code from [txt2img.py](https://github.com/CompVis/stable-diffusion/blob/main/scripts/txt2img.py):

```python
# prompts = ["a castle in a forest"]
# batch_size = 1

c = model.get_learned_conditioning(prompts)
uc = model.get_learned_conditioning(batch_size * [""])

samples_ddim, _ = sampler.sample(conditioning=c, unconditional_conditioning=uc, [...])
```

This launches the sampler that repeatedly:
- de-noises the picture guiding it to look more like your prompt (conditioning)
- de-noises the picture guiding it to look more like an empty prompt (unconditional_conditioning)
- looks at difference between those and uses it to produce a set of changes for the noisy picture (different samplers do that part differently)

To use negative prompt, all that's needed is this:

```python
# prompts = ["a castle in a forest"]
# negative_prompts = ["grainy, fog"]

c = model.get_learned_conditioning(prompts)
uc = model.get_learned_conditioning(negative_prompts)

samples_ddim, _ = sampler.sample(conditioning=c, unconditional_conditioning=uc, [...])
```

The sampler then will look at differences between image de-noised to look like your prompt (a castle), and an image de-noised to look like your negative prompt (grainy, fog), and try to move the final results towards the former and away from latter.

### Examples:

```
a colorful photo of a castle in the middle of a forest with trees and (((bushes))), by Ismail Inceoglu, ((((shadows)))), ((((high contrast)))), dynamic shading, ((hdr)), detailed vegetation, digital painting, digital drawing, detailed painting, a detailed digital painting, gothic art, featured on deviantart
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 749109862, Size: 896x448, Model hash: 7460a6fa
```

| negative prompt     | image                                                                                                                     |
|---------------------|---------------------------------------------------------------------------------------------------------------------------|
| none                | ![01069-749109862](https://user-images.githubusercontent.com/20920490/192156368-18360487-0dcf-4b7d-b57e-b3fa80a81f1a.png) |
| fog                 | ![01070-749109862](https://user-images.githubusercontent.com/20920490/192156405-9c43ba8c-4eb8-415d-9f4d-902c8cf69b6d.png) |
| grainy              | ![01071-749109862](https://user-images.githubusercontent.com/20920490/192156421-17e53296-df5c-4e82-bf9a-f1ca562d3ad0.png) |
| fog, grainy         | ![01072-749109862](https://user-images.githubusercontent.com/20920490/192156430-3d05e5c4-2b86-409c-a357-a31178e0cb30.png) |
| fog, grainy, purple | ![01073-749109862](https://user-images.githubusercontent.com/20920490/192156440-ec59abe8-1a18-4372-a100-0da8bc1f8d13.png) |
