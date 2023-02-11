# Post-Processing Extensions

## Smart Process
https://github.com/d8ahazard/sd_smartprocess

Intelligent cropping, captioning, and image enhancement.

![image](https://user-images.githubusercontent.com/1633844/201435094-433d765c-56e8-4573-82d9-71af2b112159.png)

## Depth Maps
https://github.com/thygate/stable-diffusion-webui-depthmap-script

Creates depthmaps from the generated images. The result can be viewed on 3D or holographic devices like VR headsets or lookingglass display, used in Render or Game- Engines on a plane with a displacement modifier, and maybe even 3D printed.

![image](https://user-images.githubusercontent.com/98228077/208331747-9acba3f0-3039-485e-96ab-f0cf5619ec3b.png)

## ABG_extension
https://github.com/KutsuyaYuki/ABG_extension

Automatically remove backgrounds. Uses an onnx model fine-tuned for anime images. Runs on GPU.

| ![test](https://user-images.githubusercontent.com/98228077/209423352-e8ea64e7-8522-4c2c-9350-c7cd50c35c7c.png) |  ![00035-4190733039-cow](https://user-images.githubusercontent.com/98228077/209423400-720ddde9-258a-4e68-8a93-73b67dad714e.png) | ![00021-1317075604-samdoesarts portrait](https://user-images.githubusercontent.com/98228077/209423428-da7c68db-e7d1-45a1-b931-817de9233a67.png) | ![00025-2023077221-](https://user-images.githubusercontent.com/98228077/209423446-79e676ae-460e-4282-a591-b8b986bfd869.png) |
| :---: | :---: | :---: | :---: |
| ![img_-0002-3313071906-bust shot of person](https://user-images.githubusercontent.com/98228077/209423467-789c17ad-d7ed-41a9-a039-802cfcae324a.png) | ![img_-0022-4190733039-cow](https://user-images.githubusercontent.com/98228077/209423493-dcee7860-a09e-41e0-9397-715f52bdcaab.png) | ![img_-0008-1317075604-samdoesarts portrait](https://user-images.githubusercontent.com/98228077/209423521-736f33ca-aafb-4f8b-b067-14916ec6955f.png) | ![img_-0012-2023077221-](https://user-images.githubusercontent.com/98228077/209423546-31b2305a-3159-443f-8a98-4da22c29c415.png) |

## Ultimate SD Upscaler
https://github.com/Coyote-A/ultimate-upscale-for-automatic1111

More advanced options for SD Upscale, less artifacts than original using higher denoise ratio (0.3-0.5).

## Add image number to grid
https://github.com/AlUlkesh/sd_grid_add_image_number

Add the image's number to its picture in the grid.

## haku-img
https://github.com/KohakuBlueleaf/a1111-sd-webui-haku-img

Image utils extension. Allows blending, layering, hue and color adjustments, blurring and sketch effects, and basic pixelization.

## Pixelization
https://github.com/AUTOMATIC1111/stable-diffusion-webui-pixelization

Using pre-trained models, produce pixel art out of images in the extras tab.


## Txt2VectorGraphics
https://github.com/GeorgLegato/Txt2Vectorgraphics

Create custom, scaleable icons from your prompts as SVG or PDF.

<details><summary>Example: (Click to expand:)</summary>

| prompt  |PNG  |SVG |
| :--------  | :-----------------: | :---------------------: |
| Happy Einstein | <img src="https://user-images.githubusercontent.com/7210708/193370360-506eb6b5-4fa7-4b2a-9fec-6430f6d027f5.png" width="40%" /> | <img src="https://user-images.githubusercontent.com/7210708/193370379-2680aa2a-f460-44e7-9c4e-592cf096de71.svg" width=30%/> |
| Mountainbike Downhill | <img src="https://user-images.githubusercontent.com/7210708/193371353-f0f5ff6f-12f7-423b-a481-f9bd119631dd.png" width=40%/> | <img src="https://user-images.githubusercontent.com/7210708/193371585-68dea4ca-6c1a-4d31-965d-c1b5f145bb6f.svg" width=30%/> |
coffe mug in shape of a heart | <img src="https://user-images.githubusercontent.com/7210708/193374299-98379ca1-3106-4ceb-bcd3-fa129e30817a.png" width=40%/> | <img src="https://user-images.githubusercontent.com/7210708/193374525-460395af-9588-476e-bcf6-6a8ad426be8e.svg" width=30%/> |
| Headphones | <img src="https://user-images.githubusercontent.com/7210708/193376238-5c4d4a8f-1f06-4ba4-b780-d2fa2e794eda.png" width=40%/> | <img src="https://user-images.githubusercontent.com/7210708/193376255-80e25271-6313-4bff-a98e-ba3ae48538ca.svg" width=30%/> |

</details>

## Pixel Art
https://github.com/C10udburst/stable-diffusion-webui-scripts

Simple script which resizes images by a variable amount, also converts image to use a color palette of a given size.

<details><summary>Example: (Click to expand:)</summary>

| Disabled | Enabled x8, no resize back, no color palette | Enabled x8, no color palette | Enabled x8, 16 color palette |
| :---: | :---: | :---: | :---: |
|![preview](https://user-images.githubusercontent.com/18114966/201491785-e30cfa9d-c850-4853-98b8-11db8de78c8d.png) | ![preview](https://user-images.githubusercontent.com/18114966/201492204-f4303694-e98d-4ea3-8256-538a88ea26b6.png) | ![preview](https://user-images.githubusercontent.com/18114966/201491864-d0c0c9f1-e34f-4cb6-a68e-7043ec5ce74e.png) | ![preview](https://user-images.githubusercontent.com/18114966/201492175-c55fa260-a17d-47c9-a919-9116e1caa8fe.png) |

[model used](https://publicprompts.art/all-in-one-pixel-art-dreambooth-model/)
```text
japanese pagoda with blossoming cherry trees, full body game asset, in pixelsprite style
Steps: 20, Sampler: DDIM, CFG scale: 7, Seed: 4288895889, Size: 512x512, Model hash: 916ea38c, Batch size: 4
```

</details>
