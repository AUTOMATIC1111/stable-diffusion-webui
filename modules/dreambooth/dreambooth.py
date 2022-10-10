import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import accelerate
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, whoami
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from modules import paths, sd_hijack, shared


class DreamBooth:
    # TODO: Clean up notes below and make them a proper docstring
    def __init__(self, name, training_data: str, instance_prompt: str, class_prompt: str, learn_rate: float = 5e-6,
                 save_img_every=500, save_data_every=200, max_steps: int = 800, batch_size: int = 1,
                 grad_steps: int = 1, scheduler: str = "constant", warmup_steps: int = 0, use_adam=False,
                 class_data=None, seed=None, log_interval=10, mixed_precision="no", no_cache_latents=True
                 ):
        self.tokenizer_name = None
        self.resolution = 512
        # Pretrained tokenizer name or path if not the same as model_name
        self.instance_data_dir = training_data
        # A folder containing the training data of instance images.
        if class_data is not None:
            self.class_data_dir = class_data
        else:
            self.class_data_dir = ""
        # A folder containing the training data of class images.
        self.instance_prompt = instance_prompt
        # The prompt with identifier specifing the instance
        self.class_prompt = class_prompt
        # The prompt to specify images in the same class as provided intance images.
        self.with_prior_preservation = False
        # Flag to add prior perservation loss.
        self.prior_loss_weight = 1.0
        # "The weight of prior preservation loss."
        self.num_class_images = 100
        name = "".join(x for x in name if x.isalnum())
        model_path = paths.models_path
        model_dir = os.path.join(model_path, "dreambooth", name)
        # This is where all dreambooth trainings are saved
        self.output_dir = os.path.join(model_dir, "working")
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

        logging_dir = Path(self.output_dir, self.logging_dir)

        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            log_with="tensorboard",
            logging_dir=logging_dir,
        )

        if self.seed is not None:
            set_seed(self.seed)

        if self.with_prior_preservation:
            class_images_dir = Path(self.class_data_dir)
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < self.num_class_images:
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

                context = torch.autocast("cuda") if accelerator.device.type == "cuda" else nullcontext
                for example in tqdm(
                        sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
                ):
                    with context:
                        images = pipeline(example["prompt"]).images

                    for i, image in enumerate(images):
                        image.save(class_images_dir / f"{example['index'][i] + cur_class_images}.jpg")

                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Handle the repository creation
        if accelerator.is_main_process:
            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)

        # Load the tokenizer
        if self.tokenizer_name:
            tokenizer = CLIPTokenizer.from_pretrained(self.tokenizer_name)
        elif self.pretrained_model_path:
            tokenizer = CLIPTokenizer.from_pretrained(
                self.pretrained_model_path, subfolder="tokenizer", use_auth_token=False
            )

        # Load models and create wrapper for stable diffusion
        text_encoder = CLIPTextModel.from_pretrained(
            os.path.join(self.pretrained_model_path, "text_encoder"), use_auth_token=False
        )
        vae = AutoencoderKL.from_pretrained(
            os.path.join(self.pretrained_model_path, "vae"), use_auth_token=False
        )
        unet = UNet2DConditionModel.from_pretrained(
            os.path.join(self.pretrained_model_path, "unet"), use_auth_token=False
        )

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

        # Move text_encode and vae to gpu
        text_encoder.to(accelerator.device)
        vae.to(accelerator.device)

        if not self.not_cache_latents:
            latents_cache = []
            text_encoder_cache = []
            for batch in tqdm(train_dataloader, desc="Caching latents"):
                with torch.no_grad():
                    batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, non_blocking=True)
                    batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)
                    latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)
                    text_encoder_cache.append(text_encoder(batch["input_ids"])[0])
            train_dataset = LatentsDataset(latents_cache, text_encoder_cache)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x,
                                                           shuffle=True)

            del vae, text_encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num batches each epoch = {len(train_dataloader)}")
        print(f"  Num Epochs = {self.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {self.train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {self.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        global_step = 0
        loss_avg = AverageMeter()
        for epoch in range(self.num_train_epochs):
            unet.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    with torch.no_grad():
                        if not self.not_cache_latents:
                            latent_dist = batch[0][0]
                        else:
                            latent_dist = vae.encode(batch["pixel_values"]).latent_dist
                        latents = latent_dist.sample() * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn(latents.shape).to(latents.device)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                              device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    with torch.no_grad():
                        if not self.not_cache_latents:
                            encoder_hidden_states = batch[0][1]
                        else:
                            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if self.with_prior_preservation:
                        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                        noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                        noise, noise_prior = torch.chunk(noise, 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                        # Compute prior loss
                        prior_loss = F.mse_loss(noise_pred_prior, noise_prior, reduction="none").mean([1, 2, 3]).mean()

                        # Add the prior loss to the instance loss.
                        loss = loss + self.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), self.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    loss_avg.update(loss.detach_(), bsz)

                if not global_step % self.log_interval:
                    logs = {"loss": loss_avg.avg.item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                if not global_step % self.save_img_every:
                    # Todo: send an image to the UI
                    pass

                if not global_step % self.save_data_every:
                    # Todo: pray this doesn't break anything
                    if accelerator.is_main_process:
                        pipeline = StableDiffusionPipeline.from_pretrained(
                            self.pretrained_model_path,
                            unet=accelerator.unwrap_model(unet),
                            use_auth_token=False,
                        )
                        pipeline.save_pretrained(self.output_dir)
                    pass

                progress_bar.update(1)
                global_step += 1

                if global_step >= self.max_train_steps:
                    break

            accelerator.wait_for_everyone()

        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.pretrained_model_path,
                unet=accelerator.unwrap_model(unet),
                use_auth_token=False,
            )
            pipeline.save_pretrained(self.output_dir)

        accelerator.end_training()
        return self.output_dir

    def unpack_model(self):
        pass

    def pack_model(self):
        pass


def start_training(model_dir, initialization_text, classification_text, learn_rate, dataset_directory, steps,
                   create_image_every,
                   save_embedding_every):
    print("Starting Dreambooth training...")
    try:
        sd_hijack.undo_optimizations()
        dream = DreamBooth(model_dir, dataset_directory, initialization_text, classification_text, learn_rate,
                           create_image_every, save_embedding_every, steps)

        def is_available():
            return False

        if shared.cmd_opts.medvram or shared.cmd_opts.lowvram:
            accelerate.launchers.torch.cuda.is_available = is_available
        accelerate.launchers.notebook_launcher(dream.train, num_processes=1)
    except Exception:
        raise
    finally:
        sd_hijack.apply_optimizations()


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
            raise ValueError("Instance images root doesn't exists.")

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
