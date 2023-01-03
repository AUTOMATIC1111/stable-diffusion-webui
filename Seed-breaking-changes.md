# 2023-01-03 - Hires fix rework
Rather than using width/height to specify target resolution, width/height is used to specify first pass resolution, and a slider is used to set multiplier for up scaling. New Hires fix does not crop image.

Here's how old and new settings correspond to each other:

| Old version                               | New version                                                                                     |
|-------------------------------------------|-------------------------------------------------------------------------------------------------|
| Size: 1024x1024                           | Size: 512x512, Hires upscale: 2.0                                                               |
| Size: 1280x1024, First pass size: 640x512 | Size: 640x512, Hires upscale: 2.0                                                               |
| Size: 1024x1280, First pass size: 0x0     | Size: 512x576 (auto-calcualted if you use old infotext), Hires upscale: 2.0                     |
| Size: 1024x512, First pass size: 512x512  | Impossible since this crops the first pass image to 256x512, and new hires fix does no cropping |

# 2023-01-01 - Karras sigma min/max
Some of discussion is here: [PR](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4373)

To revert to old sigmas (0.1 to 10), use setting: `Use old karras scheduler sigmas`.

# 2022-09-29 - New emphasis implementation
New implementation supports escape characters and numerical weights. A downside of the new implementation is that the old one was not perfect and sometimes ate characters: "a (((farm))), daytime", for example, would become "a farm daytime" without the comma. This behavior is not shared by the new implementation which preserves all text correctly, and this means that your saved seeds may produce different pictures.

For now, there is an option in settings to use the old implementation: `Use old emphasis implementation`.

More info about the feature: [Attention/emphasis](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#attentionemphasis)
