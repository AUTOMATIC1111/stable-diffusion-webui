import datetime
import html
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import accelerate.utils.dataclasses
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, whoami
from torch import autocast, distributed
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import modules.sd_models
from modules import paths, sd_hijack, shared, sd_models
from modules.dreambooth import conversion


class DreamBooth:
    # TODO: Clean up notes below and make them a proper docstring
    def __init__(self, name, training_data: str, instance_prompt: str, class_prompt: str, learn_rate: float = 5e-6,
                 save_img_every=500, save_data_every=200, max_steps: int = 800, batch_size: int = 1,
                 grad_steps: int = 1, scheduler: str = "constant", warmup_steps: int = 0, use_adam=False,
                 class_data=None, seed=None, log_interval=10, mixed_precision="no", no_cache_latents=True, total_steps=0,
                 num_class_images = 200):
        self.tokenizer_name = None
        self.total_steps = total_steps
        self.resolution = 512
        # Pretrained tokenizer name or path if not the same as model_name
        self.instance_data_dir = training_data
        # A folder containing the training data of class images.
        self.instance_prompt = instance_prompt
        # The prompt with identifier specifing the instance
        self.class_prompt = class_prompt
        # The prompt to specify images in the same class as provided instance images.
        self.with_prior_preservation = False

        if class_prompt != "*" and class_prompt != "":
            self.with_prior_preservation = True
        # Flag to add prior perservation loss.
        self.prior_loss_weight = 1.0
        # "The weight of prior preservation loss."
        name = "".join(x for x in name if x.isalnum())
        model_path = paths.models_path
        model_dir = os.path.join(model_path, "dreambooth", name)
        # This is where all dreambooth trainings are saved
        self.output_dir = os.path.join(model_dir, "working")
        # A folder containing the training data of instance images.
        if class_data is not None and class_data != "":
            self.class_data_dir = class_data
        else:
            self.class_data_dir = os.path.join(self.output_dir, "classifiers")
            if not os.path.exists(self.class_data_dir):
                os.makedirs(self.class_data_dir)

        self.logging_dir = os.path.join(self.output_dir, "logging")
        self.pretrained_model_path = os.path.join(model_dir, "stable-diffusion-v1-4")
        # The output directory where the model predictions and checkpoints will be written.
        self.seed = seed
        # "A seed for reproducible training.")
        # "The resolution for input images, all the images in the train/validation dataset will be resized to this"
        # " resolution"
        self.center_crop = False
        # Whether to center crop images before resizing to resolution"
        self.train_batch_size = batch_size
        # Batch size (per device) for the training dataloader."
        self.sample_batch_size = batch_size
        # Batch size (per device) for sampling images."
        self.num_train_epochs = 1
        self.max_train_steps = max_steps
        self.num_class_images = num_class_images
        # Total number of training steps to perform.  If provided, overrides num_train_epochs.
        self.gradient_accumulation_steps = grad_steps
        # Number of updates steps to accumulate before performing a backward/update pass.
        self.gradient_checkpointing = False
        # Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
        self.learning_rate = learn_rate
        # Initial learning rate (after the potential warmup period) to use.
        self.scale_lr = False
        # Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
        self.lr_scheduler = scheduler
        # The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        #     ' "constant", "constant_with_warmup"]'
        self.lr_warmup_steps = warmup_steps
        # "Number of steps for the warmup in the lr scheduler."
        self.use_8bit_adam = use_adam
        # "Whether or not to use 8-bit Adam from bitsandbytes."
        self.adam_beta1 = 0.9
        # The beta1 parameter for the Adam optimizer.")
        self.adam_beta2 = 0.999
        # The beta2 parameter for the Adam optimizer.")
        self.adam_weight_decay = 1e-2
        # Weight decay to use.")
        self.adam_epsilon = 1e-08
        # Epsilon value for the Adam optimizer")
        self.max_grad_norm = 1.0
        # Max gradient norm.")
        self.log_interval = log_interval
        # Log every N steps.")
        self.save_img_every = save_img_every
        self.save_data_every = save_data_every
        # Save image/data every N steps.
        self.mixed_precision = mixed_precision
        # choices=["no", "fp16", "bf16"],
        # help=(
        #     "Whether to use mixed precision. Choose"
        #     "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        #     "and an Nvidia Ampere GPU."
        self.not_cache_latents = no_cache_latents
        #                 Do not precompute and cache latents from VAE.")

    def train(self):
        if not os.path.exists(self.instance_data_dir):
            print("Invalid training data dir!")
            shared.state.textinfo = "Invalid training data dir"
            return "", 0

        shared.state.textinfo = "Initializing dreambooth training..."
        logging_dir = Path(self.output_dir, self.logging_dir)
        use_cpu = False
        has_deepspeed = False
        ds = None
        #self.mixed_precision="fp16"
        if shared.cmd_opts.medvram or shared.cmd_opts.lowvram:
            use_cpu = True
            # self.mixed_precision = "fp16"
            self.use_8bit_adam = True
            try:
                import bitsandbytes
                use_cpu = False
            except Exception as f:
                print(f"Exception importing ADAM: {f}")
                self.use_8bit_adam = False
            if use_cpu:
                try:
                    torch.distributed.init_process_group("mpi", init_method=None,
                                                         timeout=datetime.timedelta(seconds=1800),
                                                         world_size=- 1, rank=- 1, store=None, group_name='',
                                                         pg_options=None)
                    import deepspeed
                    ds = accelerate.utils.dataclasses.DeepSpeedPlugin()
                    conf_dict = {
                        "zero_optimization.offload_optimizer.device": "cpu",
                        "zero_optimization.offload_param.device": "cpu",
                        "zero_optimization.stage": 2
                    }
                    ds.hf_ds_config = accelerate.utils.deepspeed.HfDeepSpeedConfig(conf_dict)
                    has_deepspeed = True
                    use_cpu = False
                except Exception as e:
                    print(f"Exception importing deepspeed: {e}")
                    pass
                print(f"Creating accelerator. Using cpu: {use_cpu}. Using deepspeed: {has_deepspeed}")
        if has_deepspeed:
            accelerator = Accelerator(
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                mixed_precision=self.mixed_precision,
                log_with="tensorboard",
                logging_dir=logging_dir,
                deepspeed_plugin=ds,
                cpu=use_cpu
            )
        else:
            accelerator = Accelerator(
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                mixed_precision=self.mixed_precision,
                log_with="tensorboard",
                logging_dir=logging_dir,
                cpu=use_cpu
            )

        if self.seed is not None:
            set_seed(self.seed)

        weight_dtype = torch.float32
        if self.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        if self.with_prior_preservation:
            class_images_dir = Path(self.class_data_dir)
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < self.num_class_images:
                shared.state.textinfo = f"Generating class images for training..."
                torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                pipeline = StableDiffusionPipeline.from_pretrained(
                    self.pretrained_model_path, use_auth_token=False, torch_dtype=torch_dtype
                )
                pipeline.set_progress_bar_config(disable=True)

                num_new_images = self.num_class_images - cur_class_images
                print(f"Number of class images to sample: {num_new_images}.")

                sample_dataset = PromptDataset(self.class_prompt, num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=self.sample_batch_size)

                sample_dataloader = accelerator.prepare(sample_dataloader)
                pipeline.to(accelerator.device)
                # Disabled in opt
                context = torch.autocast("cuda") if accelerator.device.type == "cuda" else nullcontext
                with context:
                    for example in tqdm(
                            sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
                    ):
                        images = pipeline(example["prompt"]).images

                        for i, image in enumerate(images):
                            image.save(class_images_dir / f"{example['index'][i] + cur_class_images}.jpg")

                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Handle the repository creation
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
            print(f"Loading existing pre-trained model data from {ex_model_path}")

        # Load the tokenizer
        if self.tokenizer_name:
            tokenizer = CLIPTokenizer.from_pretrained(self.tokenizer_name)
        else:
            tokenizer = CLIPTokenizer.from_pretrained(os.path.join(ex_model_path, "tokenizer"))

        # Load models and create wrapper for stable diffusion
        text_encoder = CLIPTextModel.from_pretrained(os.path.join(ex_model_path, "text_encoder"))
        vae = AutoencoderKL.from_pretrained(os.path.join(ex_model_path, "vae"))
        unet = UNet2DConditionModel.from_pretrained(os.path.join(ex_model_path, "unet"))

        if self.gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        if self.scale_lr:
            self.learning_rate = (
                    self.learning_rate * self.gradient_accumulation_steps * self.train_batch_size * accelerator.num_processes
            )

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if self.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                optimizer_class = bnb.optim.AdamW8bit
            except ImportError:
                print(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )
                optimizer_class = torch.optim.AdamW

        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            unet.parameters(),  # only optimize unet
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

        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
        # T this was moved in OPT from above
        # Move text_encode and vae to gpu.
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)

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
        shared.state.textinfo = f"Training: UseCPU: {use_cpu} 8BitAdam: {self.use_8bit_adam} Precision: {self.mixed_precision} Deepspeed: {has_deepspeed}"
        print("***** Running training *****")
        print(
            f"  UseCPU: {use_cpu} 8BitAdam: {self.use_8bit_adam} Precision: {self.mixed_precision} Deepspeed: {has_deepspeed}")
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
        loss_avg = AverageMeter()
        shared.state.job_count = self.max_train_steps
        shared.state.job_no = global_step
        for epoch in range(self.num_train_epochs):
            unet.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    with torch.autocast("cuda"), torch.no_grad():
                        latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                        latents = latents * 0.18215
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                              device=latents.device)
                    timesteps = timesteps.long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    with torch.autocast("cuda"), torch.no_grad():
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if self.with_prior_preservation:
                        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                        noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                        noise, noise_prior = torch.chunk(noise, 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                        # Compute prior loss
                        prior_loss = F.mse_loss(noise_pred_prior.float(), noise_prior.float(), reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + self.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), self.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Only save a preview image if we're not training with the CPU
                if not global_step % self.save_img_every and use_cpu is not True:
                    prompt = self.instance_prompt
                    last_saved_image = os.path.join(self.output_dir, f'{self.instance_prompt}_{global_step}.png')
                    if accelerator.is_main_process:
                        print(f"Saving pretrained model data at step {global_step}.")
                        pipeline = StableDiffusionPipeline.from_pretrained(
                            self.pretrained_model_path,
                            unet=accelerator.unwrap_model(unet),
                            use_auth_token=False
                        )
                        pipeline = pipeline.to("cuda")
                        pipeline.save_pretrained(self.output_dir)
                        with autocast("cuda"):
                            image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
                            shared.state.current_image = image
                            image.save(last_saved_image)
                    pass

                # Check to make sure this doesn't throw OOM if training on CPU
                if not global_step % self.save_data_every and global_step != 0:
                    if accelerator.is_main_process:
                        print(f"Saving pretrained model data at step {global_step}.")
                        pipeline = StableDiffusionPipeline.from_pretrained(
                            self.pretrained_model_path,
                            unet=accelerator.unwrap_model(unet),
                            use_auth_token=False,
                        )
                        if use_cpu:
                            pipeline.save_pretrained(self.output_dir)
                        else:
                            pipeline = pipeline.to("cuda")
                            with autocast("cuda"):
                                pipeline.save_pretrained(self.output_dir)
                    pass

                progress_bar.update(1)
                global_step += 1
                shared.state.job_no = global_step

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                if global_step >= self.max_train_steps or shared.state.interrupted:
                    break
            accelerator.wait_for_everyone()
            if shared.state.interrupted:
                shared.state.textinfo = f"Training canceled {global_step}/{self.max_train_steps}"
                break
            shared.state.job_no += 1
        # Create the pipeline using the trained modules and save it.
        if accelerator.is_main_process:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.pretrained_model_path,
                unet=accelerator.unwrap_model(unet),
                use_auth_token=False,
            )
            pipeline.save_pretrained(self.output_dir)

        accelerator.end_training()
        return self.output_dir, global_step

    def unpack_model(self):
        pass

    def pack_model(self):
        pass


def start_training(model_name, initialization_text, classification_text, learn_rate, dataset_directory,
                   classifier_directory, steps, create_image_every, save_embedding_every, db_num_class_images):
    print("Starting Dreambooth training...")
    converted = ""
    sd_hijack.undo_optimizations()
    shared.sd_model = None
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
    dream = DreamBooth(model_name, dataset_directory, initialization_text, classification_text, learn_rate,
                       create_image_every, save_embedding_every, steps, class_data=classifier_directory,
                       num_class_images=db_num_class_images)
    out_dir, trained_steps = dream.train()
    total_steps += trained_steps
    config["total_steps"] = total_steps
    json_object = json.dumps(config, indent=4)

    # Writing to sample.json
    with open(config_file, "w") as outfile:
        outfile.write(json_object)
    out_file = os.path.join(paths.models_path, "Stable-diffusion", f"{model_name}_{total_steps}.ckpt")
    if os.path.exists(os.path.join(model_dir, "model_index.json")):
        print(f"Successfully trained model for a total of {total_steps} steps, converting to ckpt.")
        src_path = modules.sd_models.get_closet_checkpoint_match(src_checkpoint)[0]
        converted = conversion.convert_diff_to_sd(model_dir, src_path, out_file, True)
        sd_models.list_models()

    modules.sd_models.load_model()
    print("Re-applying optimizations...")
    sd_hijack.apply_optimizations()
    res = f"Training {'interrupted' if shared.state.interrupted else 'finished'}." \
          f"Total steps: {total_steps}" \
          f"Embedding saved to {html.escape(converted)}"
    print(f"Returning result: {res}")
    return res, ""


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the promots for fine-tuning the model.
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

        self.instance_images_path = [x for x in Path(instance_data_root).iterdir() if x.is_file()]
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = [x for x in Path(class_data_root).iterdir() if x.is_file()]
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


class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        return self.latents_cache[index], self.text_encoder_cache[index]


class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
