import argparse
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image, LCMScheduler

parser = argparse.ArgumentParser("lcm_convert")
parser.add_argument("--name", help="Name of the new LCM model", type=str)
parser.add_argument("--model", help="A model to convert", type=str)
parser.add_argument("--huggingface", action="store_true", help="Use Hugging Face models instead of safetensors models")
parser.add_argument("--upload", action="store_true", help="Upload the new LCM model to Hugging Face")
parser.add_argument("--no_save", action="store_true", help="Don't save the new LCM model to local disk")
parser.add_argument("--sdxl", action="store_true", help="Use SDXL models")
parser.add_argument("--ssd_1b", action="store_true", help="Use SSD-1B models")

args = parser.parse_args()

if args.huggingface:
    pipeline = AutoPipelineForText2Image.from_pretrained(args.model, torch_dtype=torch.float16, variant="fp16")
else:
    if args.sdxl or args.ssd_1b:
        pipeline = StableDiffusionXLPipeline.from_single_file(args.model)
    else:
        pipeline = StableDiffusionPipeline.from_single_file(args.model)

pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
if args.sdxl:
    pipeline.load_lora_weights("latent-consistency/lcm-lora-sdxl")
elif args.ssd_1b:
    pipeline.load_lora_weights("latent-consistency/lcm-lora-ssd-1b")
else:
    pipeline.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipeline.fuse_lora()

#components = pipeline.components
#pipeline = LatentConsistencyModelPipeline(**components)

pipeline = pipeline.to(dtype=torch.float16)
print(pipeline)

if not args.no_save:
    pipeline.save_pretrained(args.name, variant="fp16")
if args.upload:
    pipeline.push_to_hub(args.name, variant="fp16")
