# 2023-01-01 - Karras sigma min/max
Some of discussion is here: [PR](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/4373)
To revert to old sigmas (0.1 to 10), use setting: `Use old karras scheduler sigmas`.

# 2022-09-29 - New emphasis implementation
New implementation supports escape characters and numerical weights. A downside of the new implementation is that the old one was not perfect and sometimes ate characters: "a (((farm))), daytime", for example, would become "a farm daytime" without the comma. This behavior is not shared by the new implementation which preserves all text correctly, and this means that your saved seeds may produce different pictures. For now, there is an option in settings to use the old implementation: `Use old emphasis implementation`.
More info about the feature: [Attention/emphasis](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#attentionemphasis)
