import gc
import hashlib
import html
import itertools
import json
import math
import os
import traceback
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, whoami
from torch import autocast
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import modules.sd_models
from modules import paths, sd_hijack, shared, sd_models
from modules.dreambooth import conversion

mem_record = {}


def printm(msg, reset=False):
    global mem_record
    if reset:
        mem_record = {}
    allocated = round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)
    cached = round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)
    mem_record[msg] = f"{allocated}/{cached}GB"
    print(f' {msg} \n Allocated: {allocated}GB \n Reserved: {cached}GB \n')


class DreamBooth:
    # TODO: Clean up notes below and make them a proper docstring
    def __init__(self,
                 model_name,
                 instance_prompt,
                 class_prompt,
                 learn_rate,
                 instance_data_dir,
                 class_data_dir,
                 steps,
                 create_image_every,
                 save_embedding_every,
                 num_class_images,
                 use_cpu,
                 train_text_encoder,
                 not_cache_latents,
                 use_adam,
                 center_crop,
                 gradient_checkpointing,
                 scale_lr,
                 mixed_precision,
                 scheduler,
                 resolution,
                 prior_loss_weight,
                 num_train_epochs,
                 adam_beta1,
                 adam_beta2,
                 adam_weight_decay,
                 adam_epsilon,
                 max_grad_norm,
                 batch_size,
                 class_batch_size,
                 seed,
                 grad_acc_steps,
                 warmup_steps,
                 total_steps
                 ):
        self.total_steps = total_steps
        self.instance_data_dir = instance_data_dir
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        if seed == -1:
            self.seed = None
        else:
            self.seed = seed
        self.train_batch_size = batch_size
        self.sample_batch_size = class_batch_size
        self.max_train_steps = steps
        self.num_class_images = num_class_images
        self.gradient_accumulation_steps = grad_acc_steps
        self.learning_rate = learn_rate
        self.lr_scheduler = scheduler
        self.lr_warmup_steps = warmup_steps
        self.use_8bit_adam = use_adam
        self.log_interval = 10
        self.save_img_every = create_image_every
        self.save_data_every = save_embedding_every
        self.mixed_precision = mixed_precision
        self.not_cache_latents = not_cache_latents

        name = "".join(x for x in model_name if x.isalnum())
        model_path = paths.models_path
        model_dir = os.path.join(model_path, "dreambooth", name)
        self.output_dir = os.path.join(model_dir, "working")
        # A folder containing the training data of instance images.
        if class_data_dir is not None and class_data_dir != "":
            self.class_data_dir = class_data_dir
        else:
            self.class_data_dir = os.path.join(self.output_dir, "classifiers")
            if not os.path.exists(self.class_data_dir):
                os.makedirs(self.class_data_dir)

        self.logging_dir = os.path.join(self.output_dir, "logging")
        self.pretrained_model_path = os.path.join(model_dir, "stable-diffusion-v1-4")
        self.with_prior_preservation = False

        if class_prompt != "*" and class_prompt != "" and num_class_images != 0:
            self.with_prior_preservation = True
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.train_text_encoder = train_text_encoder
        self.resolution = resolution
        self.use_cpu = use_cpu
        self.prior_loss_weight = prior_loss_weight
        self.center_crop = center_crop
        self.num_train_epochs = num_train_epochs
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_lr = scale_lr
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_weight_decay = adam_weight_decay
        self.adam_epsilon = adam_epsilon
        self.max_grad_norm = max_grad_norm

    def train(self):
        logging_dir = Path(self.output_dir, self.logging_dir)

        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            log_with="tensorboard",
            logging_dir=logging_dir,
            cpu=self.use_cpu
        )

        if self.train_text_encoder and self.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
            msg = "Gradient accumulation is not supported when training the text encoder in distributed training. " \
                  "Please set gradient_accumulation_steps to 1. This feature will be supported in the future. Text " \
                  "encoder training will be disabled."
            print(msg)
            shared.state.textinfo = msg
            self.train_text_encoder = False

        if self.seed is not None:
            set_seed(self.seed)

        if self.with_prior_preservation:
            class_images_dir = Path(self.class_data_dir)
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < self.num_class_images:
                shared.state.textinfo = f"Generating class images for training..."
                torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                pipeline = StableDiffusionPipeline.from_pretrained(
                    self.pretrained_model_path,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    revision=self.total_steps,
                )

                def foo(images, clip_input):
                    return images, False

                pipeline.safety_checker = foo
                pipeline.set_progress_bar_config(disable=True)

                num_new_images = self.num_class_images - cur_class_images
                print(f"Number of class images to sample: {num_new_images}.")
                shared.state.job_count = num_new_images
                shared.state.job_no = 0
                sample_dataset = PromptDataset(self.class_prompt, num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=self.sample_batch_size)
                sample_dataloader = accelerator.prepare(sample_dataloader)
                pipeline.to(accelerator.device)
                for example in tqdm(
                    sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
                ):
                    images = pipeline(example["prompt"]).images

                    for i, image in enumerate(images):
                        shared.state.job_no += 1
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        shared.state.current_image = image
                        image.save(image_filename)

                    if shared.state.interrupted:
                        break

                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if shared.state.interrupted:
                    shared.state.textinfo = "Training canceled..."
                    return self.output_dir, 0
        # Load existing training data if exist
        shared.state.textinfo = "Loading models..."
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
        ex_encoder = os.path.join(self.output_dir, "text_encoder", "pytorch_model.bin")
        ex_vae = os.path.join(self.output_dir, "vae", "diffusion_pytorch_model.bin")
        ex_unet = os.path.join(self.output_dir, "unet", "diffusion_pytorch_model.bin")
        ex_tokenizer = os.path.join(self.output_dir, "tokenizer")
        ex_model_path = self.pretrained_model_path
        if os.path.exists(ex_encoder) and os.path.exists(ex_vae) and os.path.exists(ex_unet) and os.path.exists(
                ex_tokenizer):
            ex_model_path = self.output_dir

        tokenizer = CLIPTokenizer.from_pretrained(os.path.join(ex_model_path, "tokenizer"))
        # Load models and create wrapper for stable diffusion
        text_encoder = CLIPTextModel.from_pretrained(os.path.join(ex_model_path, "text_encoder"))
        vae = AutoencoderKL.from_pretrained(os.path.join(ex_model_path, "vae"))
        unet = UNet2DConditionModel.from_pretrained(os.path.join(ex_model_path, "unet"))
        printm("Loaded model.")
        vae.requires_grad_(False)
        if not self.train_text_encoder:
            text_encoder.requires_grad_(False)
        if self.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            if self.train_text_encoder:
                text_encoder.gradient_checkpointing_enable()

        if self.scale_lr:
            self.learning_rate = (
                    self.learning_rate * self.gradient_accumulation_steps * self.train_batch_size * accelerator.num_processes
            )

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        use_adam = False
        optimizer_class = torch.optim.AdamW
        if self.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_class = bnb.optim.AdamW8bit
                use_adam = True
            except Exception as a:
                print(f"Exception importing 8bit adam: {a}")

        params_to_optimize = (
            itertools.chain(unet.parameters(),
                            text_encoder.parameters()) if self.train_text_encoder else unet.parameters()
        )
        optimizer = optimizer_class(
            params_to_optimize,
            lr=self.learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay,
            eps=self.adam_epsilon,
        )

        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
        )
        train_dataset = DreamBoothDataset(
            instance_data_root=self.instance_data_dir,
            instance_prompt=self.instance_prompt,
            class_data_root=self.class_data_dir if self.with_prior_preservation else None,
            class_prompt=self.class_prompt,
            tokenizer=tokenizer,
            size=self.resolution,
            center_crop=self.center_crop,
        )

        def collate_fn(examples):
            input_ids = [example["instance_prompt_ids"] for example in examples]
            pixel_values = [example["instance_images"] for example in examples]

            # Concat class and instance examples for prior preservation.
            # We do this to avoid doing two forward passes.
            if self.with_prior_preservation:
                input_ids += [example["class_prompt_ids"] for example in examples]
                pixel_values += [example["class_images"] for example in examples]

            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

            input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

            batch = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
            }
            return batch

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.train_batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.gradient_accumulation_steps)
        if self.max_train_steps is None:
            self.max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            self.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps * self.gradient_accumulation_steps,
            num_training_steps=self.max_train_steps * self.gradient_accumulation_steps,
        )
        printm("Scheduler Loaded")
        if self.train_text_encoder and text_encoder is not None:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )
        weight_dtype = torch.float32
        if self.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        vae.to(accelerator.device, dtype=weight_dtype)
        if not self.train_text_encoder:
            text_encoder.to(accelerator.device, dtype=weight_dtype)
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.max_train_steps = self.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.num_train_epochs = math.ceil(self.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("dreambooth", config=vars(self))

        # Train!
        total_batch_size = self.train_batch_size * accelerator.num_processes * self.gradient_accumulation_steps
        stats = f"CPU: {self.use_cpu} Adam: {use_adam}, Prec: {self.mixed_precision}, " \
                f"Prior: {self.with_prior_preservation}, Grad: {self.gradient_checkpointing}, " \
                f"TextTr: {self.train_text_encoder} "
        printm(stats)
        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num batches each epoch = {len(train_dataloader)}")
        print(f"  Num Epochs = {self.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {self.train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {self.max_train_steps}")
        print(f"  Total lifetime optimization steps = {self.max_train_steps + self.total_steps}")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        global_step = 0

        shared.state.job_count = self.max_train_steps
        shared.state.job_no = global_step
        shared.state.textinfo = f"Training: {global_step}/{self.max_train_steps} steps"
        training_failed = False

        try:
            for epoch in range(self.num_train_epochs):
                unet.train()
                if self.train_text_encoder:
                    text_encoder.train()
                for step, batch in enumerate(train_dataloader):
                    with accelerator.accumulate(unet):
                        # Convert images to latent space
                        latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                        latents = latents * 0.18215

                        # Sample noise that we'll add to the latents
                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]
                        # Sample a random timestep for each image
                        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                                  device=latents.device)
                        timesteps = timesteps.long()

                        # Add noise to the latents according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                        # Get the text embedding for conditioning
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                        # Predict the noise residual
                        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                        if self.with_prior_preservation:
                            # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                            noise, noise_prior = torch.chunk(noise, 2, dim=0)

                            # Compute instance loss
                            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean(
                                [1, 2, 3]).mean()

                            # Compute prior loss
                            prior_loss = F.mse_loss(noise_pred_prior.float(), noise_prior.float(), reduction="mean")

                            # Add the prior loss to the instance loss.
                            loss = loss + self.prior_loss_weight * prior_loss
                        else:
                            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            params_to_clip = (
                                itertools.chain(unet.parameters(), text_encoder.parameters())
                                if self.train_text_encoder
                                else unet.parameters()
                            )
                            accelerator.clip_grad_norm_(params_to_clip, self.max_grad_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                        if self.save_img_every:
                            if not global_step % self.save_img_every and global_step != 0:
                                prompt = self.instance_prompt
                                last_saved_image = os.path.join(self.output_dir,
                                                                f'{self.instance_prompt}_{global_step}.png')
                                if accelerator.is_main_process:
                                    pipeline = StableDiffusionPipeline.from_pretrained(
                                        self.pretrained_model_path,
                                        unet=accelerator.unwrap_model(unet),
                                        revision=self.total_steps + global_step
                                    )
                                    with autocast("cuda"):
                                        image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
                                        shared.state.current_image = image
                                        image.save(last_saved_image)
                                pass

                        if self.save_data_every:
                            # Check to make sure this doesn't throw OOM if training on CPU
                            if not global_step % self.save_data_every and global_step != 0:
                                if accelerator.is_main_process:
                                    print(f"Saving pretrained model data at step {global_step}.")
                                    pipeline = StableDiffusionPipeline.from_pretrained(
                                        self.pretrained_model_path,
                                        unet=accelerator.unwrap_model(unet),
                                        revision=self.total_steps + global_step
                                    )
                                    pipeline = pipeline.to("cuda")
                                    with autocast("cuda"):
                                        pipeline.text_encoder.resize_token_embeddings(49409)
                                        pipeline.save_pretrained(self.output_dir)
                                pass
                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1
                        shared.state.job_no = global_step
                        shared.state.textinfo = f"Training: {global_step}/{self.max_train_steps} steps"
                        if shared.state.interrupted:
                            shared.state.textinfo = f"Training canceled {global_step}/{self.max_train_steps}"
                            break
                        if global_step >= self.max_train_steps:
                            break
                    accelerator.wait_for_everyone()
                    if shared.state.interrupted:
                        shared.state.textinfo = f"Training canceled {global_step}/{self.max_train_steps}"
                        break
                    shared.state.job_no = global_step
                    shared.state.textinfo = f"Training: {global_step}/{self.max_train_steps} steps"
                # Create the pipeline using the trained modules and save it.
        except Exception as e:
            printm("Caught exception.")
            print(f"Exception training db: {e}")
            print(traceback.format_exc())
            training_failed = True

        if not training_failed:
            if accelerator.is_main_process:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    self.pretrained_model_path,
                    unet=accelerator.unwrap_model(unet),
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    revision=self.total_steps + global_step
                )
                pipeline.text_encoder.resize_token_embeddings(49409)
                pipeline.save_pretrained(self.output_dir)
                del pipeline

        def cleanup():
            gc.collect()  # Python thing
            torch.cuda.empty_cache()  # PyTorch thing

        # Free memory after OOM?
        try:
            print("CLEANUP: ")
            if unet:
                del unet
            if text_encoder:
                del text_encoder
            if tokenizer:
                del tokenizer
            if optimizer:
                del optimizer
            if train_dataloader:
                del train_dataloader
            if train_dataset:
                del train_dataset
            if lr_scheduler:
                del lr_scheduler
            if vae:
                del vae
        except:
            pass
        cleanup()
        printm("Cleanup Complete")
        try:
            accelerator.end_training()
        except Exception as f:
            print(f"Exception ending training: {f}")

        return self.output_dir, global_step


def start_training(model_name,
                   instance_prompt,
                   class_prompt,
                   learn_rate,
                   dataset_directory,
                   classification_directory,
                   steps,
                   create_image_every,
                   save_embedding_every,
                   num_class_images,
                   use_cpu,
                   train_text_encoder,
                   not_cache_latents,
                   use_adam,
                   center_crop,
                   gradient_checkpointing,
                   scale_lr,
                   mixed_precision,
                   scheduler,
                   resolution,
                   prior_loss_weight,
                   num_train_epochs,
                   adam_beta1,
                   adam_beta2,
                   adam_weight_decay,
                   adam_epsilon,
                   max_grad_norm,
                   batch_size,
                   class_batch_size,
                   seed,
                   grad_acc_steps,
                   warmup_steps
                   ):
    print("Starting Dreambooth training...")
    converted = ""
    sd_hijack.undo_optimizations()
    shared.sd_model.to('cpu')
    torch.cuda.empty_cache()
    gc.collect()
    printm("VRAM cleared, beginning training.", True)
    model_path = paths.models_path
    model_dir = os.path.join(model_path, "dreambooth", model_name, "working")
    config = None
    config_file = os.path.join(model_dir, "config.json")
    try:
        with open(config_file, 'r') as openfile:
            config = json.load(openfile)
    except:
        pass

    if config is None:
        print("Unable to load config?")
        return "Invalid source checkpoint", ""

    src_checkpoint = config["src"]
    total_steps = config["total_steps"]
    dream = DreamBooth(model_name,
                       instance_prompt,
                       class_prompt,
                       learn_rate,
                       dataset_directory,
                       classification_directory,
                       steps,
                       create_image_every,
                       save_embedding_every,
                       num_class_images,
                       use_cpu,
                       train_text_encoder,
                       not_cache_latents,
                       use_adam,
                       center_crop,
                       gradient_checkpointing,
                       scale_lr,
                       mixed_precision,
                       scheduler,
                       resolution,
                       prior_loss_weight,
                       num_train_epochs,
                       adam_beta1,
                       adam_beta2,
                       adam_weight_decay,
                       adam_epsilon,
                       max_grad_norm,
                       batch_size,
                       class_batch_size,
                       seed,
                       grad_acc_steps,
                       warmup_steps,
                       total_steps)
    if not os.path.exists(dream.instance_data_dir):
        print("Invalid training data dir!")
        shared.state.textinfo = "Invalid training data dir"
        return "", 0

    shared.state.textinfo = "Initializing dreambooth training..."
    out_dir, trained_steps = dream.train()
    total_steps += trained_steps
    config["total_steps"] = total_steps
    json_object = json.dumps(config, indent=4)
    if trained_steps > 0:
        with open(config_file, "w") as outfile:
            outfile.write(json_object)
        out_file = os.path.join(paths.models_path, "Stable-diffusion", f"{model_name}_{total_steps}.ckpt")
        if os.path.exists(os.path.join(model_dir, "model_index.json")):
            print(f"Successfully trained model for a total of {total_steps} steps, converting to ckpt.")
            src_path = modules.sd_models.get_closet_checkpoint_match(src_checkpoint)[0]
            converted = conversion.convert_diff_to_sd(model_dir, src_path, out_file, True)
            sd_models.list_models()
        embed_msg = f"Embedding saved to {html.escape(converted)}"
    else:
        print("Oops, something must have happened, unable to train model.")
        embed_msg = "Nothing to save."
    torch.cuda.empty_cache()
    gc.collect()
    printm("Training completed, reloading SD Model.")
    print(f'Memory output: {mem_record}')
    shared.sd_model.to(shared.device)
    # modules.sd_models.load_model()
    print("Re-applying optimizations...")
    sd_hijack.apply_optimizations()
    res = f"Training {'interrupted' if shared.state.interrupted else 'finished'}." \
          f"Total steps: {total_steps} \n {embed_msg}"
    print(f"Returning result: {res}")
    return res, ""


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            shared.state.textinfo = f"Invalid directory for training data: {self.instance_data_root}"

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def get_db_models():
    model_dir = paths.models_path
    out_dir = os.path.join(model_dir, "dreambooth")
    output = []
    if os.path.exists(out_dir):
        dirs = os.listdir(out_dir)
        for found in dirs:
            if os.path.isdir(os.path.join(out_dir, found)):
                output.append(found)
    return output
