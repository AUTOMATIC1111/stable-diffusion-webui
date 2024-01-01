## 1.8.0 (dev: 1.7.0-225) [2024-01-01](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14145) - zero terminal SNR noise schedule option
Slightly changes all image generation. The PR changes alphas_cumprod to be never be fp16 unless the backwards compatibility option is enabled. Backwards compatibility option is "Downcast model alphas_cumprod to fp16 before sampling", and it's automatically enabled when restoring parameters from old pictures (as long as they have Version: ... in infotext).

## 1.6.0 [2023-08-24](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12457) - prompt editing timeline has separate range for first pass and hires-fix pass
Two changes:
1. Before the change, prompt editing instructions like `[red:green:0.25]` were the same for normal generation and for hires fix second pass. After: values in range 0.0 - 1.0 apply to first pass, and in range 1.0 - 2.0 - to second pass.
2. Before the change: numbers below 1 mean fraction of steps, number above - absolute number of steps. After the change: numbers with fractional point in them mean fraction of steps, without - absolute number of steps

There is a setting to enable old behavior on compatibility page.

| pattern            | old first pass                           | old second pass | new first pass                          | new second pass                          |
|--------------------|------------------------------------------|-----------------|-----------------------------------------|------------------------------------------|
| `[red:green:0.25]` | 25% of steps `red`, 75% of steps `green` | same            | 25% of steps `red`, 75% of steps `green` | `green`                                  |
| `[red:green:1.25]` | first step `red`, other steps `green`    | same            | `red`                                   | 25% of steps `red`, 75% of steps `green` |
| `[red:green:5]`    | first 5 steps `red`, other steps `green` | same            | first 5 steps `red`, other steps `green` | `green`                                  |
| `[red:green:5.0]`  | first 5 steps `red`, other steps `green` | same            | `red`                                   | `red`                                    |

## 1.6.0 [2023-07-30](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12177) - add support for whitespace after the number in attention and prompt editing
Before the PR, whitespace after the number in prompt editing (`[foo:0.5]`), and also before and after number in attention (`(foo:0.5)`) caused them to not work and just be treated as plain text. The PR changes this and now `[foo : 0.5 ]` and `(foo : 0.5 )` work. Prompts where the user erroneously written whitespaces where they are not allowed will generate different pictures.

## [2023-04-29](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/9669) - Fix prompt schedule for second order samplers
Second order samplers (Heun, DPM2/a, DPM++ 2S/a, DPM++ SDE / Karras) cause the prompt schedule to run twice as fast when prompting something like `[dog:cat:0.5]` (i.e. for 100 steps, prompt is `dog` until step 25, `cat` until 50, and remains `dog` until 100). This fixes that by checking if the sampler is any of these second order samplers and multiplies the step count by 2 for calculating the prompt schedule.

## [2023-03-26](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/80b26d2a69617b75d2d01c1e6b7d11445815ed4d) - Apply LoRA by altering layer's weights
TLDR: produces pictures are a little bit different. If using highres fix, those small differences can be amplified into big ones.

New method introduced in [80b26d2a](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/80b26d2a69617b75d2d01c1e6b7d11445815ed4d) allows to pre-calculate new model weights once and then not have to do anything when creating images. With this, adding many loras will incur small performance overhead the first time you apply those loras, and after that will be as fast as if you were making pictures without any loras enabled. Old method slows down generation by a lot with every new lora added.

Differences between produced images are tiny, but if that matters for you (or for some extension you are using), 1.2.0 adds an option to use old method.

## [2023-02-18](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/7730) - deterministic DPM++ SDE across different batch sizes
DPM++ SDE and DPM++ SDE Karras samplers used to produce different images in batches compared to single image with same parameters. PR https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/7730 fixes this. But the nature of the fix also changes what generates for single images. an option is added to compatibility settings to revert to old behavior: Do not make DPM++ SDE deterministic across different batch sizes.

## [2023-01-11](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/6628) - Alternating words syntax bugfix
If you used alternating words syntax bugfix with emphasis before [97ff69ef](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/97ff69eff338c6641f4abf430bf5ac112c1775e0), the program would incorrectly replace emphasized part with just `(`. So, `[a|(b:1.1)]`, rather than becoming a sequence of

`a` -> `(b:1.1)` -> `a` -> `(b:1.1)` -> ...

becomes

`a` -> `(` -> `a` -> `(` -> ...

The bug was fixed. If you need to reproduce old seeds, put the opening parenthesis into your prompt yourself (`[a|\(]`)

## [2023-01-05](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/6044) - Karras sigma min/max
Some of discussion is here: [PR](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4373)

To revert to old sigmas (0.1 to 10), use setting: `Use old karras scheduler sigmas`.

## [2023-01-02](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/ef27a18b6b7cb1a8eebdc9b2e88d25baf2c2414d) - Hires fix rework
Rather than using width/height to specify target resolution, width/height is used to specify first pass resolution, and resulting resolution is either set using "Scale by" multiplier (Hires upscale), or directly using "Resize width to" and/or "Resize height to" (Hires resize).

Here's how old and new settings correspond to each other:

| Old version                               | New version                                                                                     |
|-------------------------------------------|-------------------------------------------------------------------------------------------------|
| Size: 1024x1024                           | Size: 512x512, Hires upscale: 2.0                                                               |
| Size: 1280x1024, First pass size: 640x512 | Size: 640x512, Hires upscale: 2.0; Alternatively Size: 640x512, Hires resize: 1280x1024                                                               |
| Size: 1024x1280, First pass size: 0x0     | Size: 512x576 (auto-calcualted if you use old infotext - paste it into prompt and use ↙️ button), Hires upscale: 2.0                     |
| Size: 1024x512, First pass size: 512x512  | Size: 512x512, Hires resize: 1024x512 |

## [2022-09-29](https://github.com/AUTOMATIC1111/stable-diffusion-webui/commit/c1c27dad3ba371a5ae344b267c760aa51e77f193) - New emphasis implementation
New implementation supports escape characters and numerical weights. A downside of the new implementation is that the old one was not perfect and sometimes ate characters: "a (((farm))), daytime", for example, would become "a farm daytime" without the comma. This behavior is not shared by the new implementation which preserves all text correctly, and this means that your saved seeds may produce different pictures.

For now, there is an option in settings to use the old implementation: `Use old emphasis implementation`.

More info about the feature: [Attention/emphasis](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#attentionemphasis)
