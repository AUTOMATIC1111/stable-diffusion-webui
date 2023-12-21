import torch
import diffusers
from PIL import Image
from rich import print

model_id = "runwayml/stable-diffusion-v1-5"
print(f'torch=={torch.__version__} diffusers=={diffusers.__version__}')

adapters = [
    'TencentARC/t2iadapter_canny_sd15v2',
    # 'TencentARC/t2iadapter_depth_sd15v2',
    # 'TencentARC/t2iadapter_zoedepth_sd15v1',
    # 'TencentARC/t2iadapter_openpose_sd14v1',
    # 'TencentARC/t2iadapter_sketch_sd15v2',
]
seeds = [42]

print(f'loading: {model_id}')
base = diffusers.StableDiffusionPipeline.from_pretrained(model_id, variant="fp16", cache_dir='/mnt/d/Models/Diffusers').to('cuda')
image = Image.new('RGB', (512,512), 0) # input is irrelevant, so just creating blank image
print('loaded')

def callback(step: int, timestep: int, latents: torch.FloatTensor):
    print(f'callback: step={step} timestep={timestep} latents={latents.shape}')

for adapter_id in adapters:
    print(f'loading: {adapter_id}')
    adapter = diffusers.T2IAdapter.from_pretrained('TencentARC/t2iadapter_depth_sd15v2', cache_dir='/mnt/d/Models/Diffusers')
    pipe = diffusers.StableDiffusionAdapterPipeline(
        vae=base.vae,
        text_encoder=base.text_encoder,
        tokenizer=base.tokenizer,
        unet=base.unet,
        scheduler=base.scheduler,
        requires_safety_checker=False,
        safety_checker=None,
        feature_extractor=None,
        adapter=adapter,
    ).to('cuda')
    output = pipe(prompt=['test'], negative_prompt=['test'], num_inference_steps=20, image=image) # ok
    print(f'adapter: {adapter_id} {output}')
    pipe.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.config['num_train_timesteps'] = 1000
    pipe.scheduler.config['beta_start'] = 0.00085
    pipe.scheduler.config['beta_end'] = 0.012
    pipe.scheduler.config['beta_schedule'] = 'scaled_linear'
    pipe.scheduler.config['prediction_type'] = 'epsilon'
    pipe.scheduler.config['rescale_betas_zero_snr'] = False
    output = pipe(
        prompt=['test'],
        negative_prompt=['test'],
        num_inference_steps=20,
        image=image,
        callback=callback,
        callback_steps=1,
        output_type='latent',
        eta=1.0,
        clip_skip=1,
        guidance_scale=6,
        generator=[torch.Generator('cpu').manual_seed(seed) for seed in seeds],
    )
    print(f'adapter: {adapter_id} {output}')

"""
'callback_steps': 1,
'callback': <function process_diffusers.<locals>.diffusers_callback_legacy at 0x7f4569259a80>,

'guidance_scale': 6,
'generator': [<torch._C.Generator object at 0x7f4562c72370>],
'num_inference_steps': 20,

'eta': 1.0,
'clip_skip': 1,
'image': <PIL.Image.Image image mode=RGB size=512x512 at 0x7F456A271A50>}

Given groups=1, weight of size [320, 64, 3, 3], expected input[1, 192, 64, 64] to have 64 channels, but got 192 channels instead
"""
