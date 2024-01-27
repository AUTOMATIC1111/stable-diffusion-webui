from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import PIL
import torch
import torch.nn as nn
from safetensors import safe_open
from huggingface_hub.utils import validate_hf_hub_args
from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.utils import _get_model_file
from transformers import CLIPImageProcessor
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
from transformers.models.clip.configuration_clip import CLIPVisionConfig


PipelineImageInput = Union[
    PIL.Image.Image,
    torch.FloatTensor,
    List[PIL.Image.Image],
    List[torch.FloatTensor],
]


VISION_CONFIG_DICT = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768
}


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class FuseModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def fuse_fn(self, prompt_embeds, id_embeds):
        stacked_id_embeds = torch.cat([prompt_embeds, id_embeds], dim=-1)
        stacked_id_embeds = self.mlp1(stacked_id_embeds) + prompt_embeds
        stacked_id_embeds = self.mlp2(stacked_id_embeds)
        stacked_id_embeds = self.layer_norm(stacked_id_embeds)
        return stacked_id_embeds

    def forward(
        self,
        prompt_embeds,
        id_embeds,
        class_tokens_mask,
    ) -> torch.Tensor:
        # id_embeds shape: [b, max_num_inputs, 1, 2048]
        id_embeds = id_embeds.to(prompt_embeds.dtype)
        num_inputs = class_tokens_mask.sum().unsqueeze(0)
        batch_size, max_num_inputs = id_embeds.shape[:2]
        # seq_length: 77
        seq_length = prompt_embeds.shape[1]
        # flat_id_embeds shape: [b*max_num_inputs, 1, 2048]
        flat_id_embeds = id_embeds.view(
            -1, id_embeds.shape[-2], id_embeds.shape[-1]
        )
        # valid_id_mask [b*max_num_inputs]
        valid_id_mask = (
            torch.arange(max_num_inputs, device=flat_id_embeds.device)[None, :]
            < num_inputs[:, None]
        )
        valid_id_embeds = flat_id_embeds[valid_id_mask.flatten()]

        prompt_embeds = prompt_embeds.view(-1, prompt_embeds.shape[-1])
        class_tokens_mask = class_tokens_mask.view(-1)
        valid_id_embeds = valid_id_embeds.view(-1, valid_id_embeds.shape[-1])
        # slice out the image token embeddings
        image_token_embeds = prompt_embeds[class_tokens_mask]
        stacked_id_embeds = self.fuse_fn(image_token_embeds, valid_id_embeds)
        assert class_tokens_mask.sum() == stacked_id_embeds.shape[0], f"{class_tokens_mask.sum()} != {stacked_id_embeds.shape[0]}"
        prompt_embeds.masked_scatter_(class_tokens_mask[:, None], stacked_id_embeds.to(prompt_embeds.dtype))
        updated_prompt_embeds = prompt_embeds.view(batch_size, seq_length, -1)
        return updated_prompt_embeds

class PhotoMakerIDEncoder(CLIPVisionModelWithProjection):
    def __init__(self):
        super().__init__(CLIPVisionConfig(**VISION_CONFIG_DICT))
        self.visual_projection_2 = nn.Linear(1024, 1280, bias=False)
        self.fuse_module = FuseModule(2048)

    def forward(self, id_pixel_values, prompt_embeds, class_tokens_mask):
        b, num_inputs, c, h, w = id_pixel_values.shape
        id_pixel_values = id_pixel_values.view(b * num_inputs, c, h, w)

        shared_id_embeds = self.vision_model(id_pixel_values)[1]
        id_embeds = self.visual_projection(shared_id_embeds)
        id_embeds_2 = self.visual_projection_2(shared_id_embeds)

        id_embeds = id_embeds.view(b, num_inputs, 1, -1)
        id_embeds_2 = id_embeds_2.view(b, num_inputs, 1, -1)

        id_embeds = torch.cat((id_embeds, id_embeds_2), dim=-1)
        updated_prompt_embeds = self.fuse_module(prompt_embeds, id_embeds, class_tokens_mask)

        return updated_prompt_embeds


class PhotoMakerStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    @validate_hf_hub_args
    def load_photomaker_adapter(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        weight_name: str,
        subfolder: str = '',
        trigger_word: str = 'img',
        **kwargs,
    ):
        # Load the main state dict first.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            if weight_name.endswith(".safetensors"):
                state_dict = {"id_encoder": {}, "lora_weights": {}}
                with safe_open(model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("id_encoder."):
                            state_dict["id_encoder"][key.replace("id_encoder.", "")] = f.get_tensor(key)
                        elif key.startswith("lora_weights."):
                            state_dict["lora_weights"][key.replace("lora_weights.", "")] = f.get_tensor(key)
            else:
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        keys = list(state_dict.keys())
        if keys != ["id_encoder", "lora_weights"]:
            raise ValueError("Required keys are (`id_encoder` and `lora_weights`) missing from the state dict.")

        self.trigger_word = trigger_word
        # load finetuned CLIP image encoder and fuse module here if it has not been registered to the pipeline yet
        id_encoder = PhotoMakerIDEncoder()
        id_encoder.load_state_dict(state_dict["id_encoder"], strict=True)
        id_encoder = id_encoder.to(self.device, dtype=self.unet.dtype)
        self.id_encoder = id_encoder
        self.id_image_processor = CLIPImageProcessor()

        # load lora into models
        self.load_lora_weights(state_dict["lora_weights"], adapter_name="photomaker")

        # Add trigger word token
        if self.tokenizer is not None:
            self.tokenizer.add_tokens([self.trigger_word], special_tokens=True)
        self.tokenizer_2.add_tokens([self.trigger_word], special_tokens=True)

    def encode_prompt_with_trigger_word(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        num_id_images: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        class_tokens_mask: Optional[torch.LongTensor] = None,
    ):
        device = device or self._execution_device

        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        """

        # Find the token id of the trigger word
        image_token_id = self.tokenizer_2.convert_tokens_to_ids(self.trigger_word)

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                input_ids = tokenizer.encode(prompt) # TODO: batch encode
                clean_index = 0
                clean_input_ids = []
                class_token_index = []
                # Find out the corrresponding class word token based on the newly added trigger word token
                for _i, token_id in enumerate(input_ids):
                    if token_id == image_token_id:
                        class_token_index.append(clean_index - 1)
                    else:
                        clean_input_ids.append(token_id)
                        clean_index += 1

                if len(class_token_index) != 1:
                    raise ValueError(
                        f"PhotoMaker currently does not support multiple trigger words in a single prompt.\
                            Trigger word: {self.trigger_word}, Prompt: {prompt}."
                    )
                class_token_index = class_token_index[0]

                # Expand the class word token and corresponding mask
                class_token = clean_input_ids[class_token_index]
                clean_input_ids = clean_input_ids[:class_token_index] + [class_token] * num_id_images + \
                    clean_input_ids[class_token_index+1:]

                # Truncation or padding
                max_len = tokenizer.model_max_length
                if len(clean_input_ids) > max_len:
                    clean_input_ids = clean_input_ids[:max_len]
                else:
                    clean_input_ids = clean_input_ids + [tokenizer.pad_token_id] * (
                        max_len - len(clean_input_ids)
                    )

                class_tokens_mask = [True if class_token_index <= i < class_token_index+num_id_images else False \
                     for i in range(len(clean_input_ids))]

                clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long).unsqueeze(0)
                class_tokens_mask = torch.tensor(class_tokens_mask, dtype=torch.bool).unsqueeze(0)

                prompt_embeds = text_encoder(
                    clean_input_ids.to(device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        class_tokens_mask = class_tokens_mask.to(device=device) # TODO: ignoring two-prompt case

        return prompt_embeds, pooled_prompt_embeds, class_tokens_mask


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        # Added parameters (for PhotoMaker)
        input_id_images: PipelineImageInput = None,
        start_merge_step: int = 0, # TODO: change to `style_strength_ratio` in the future
        class_tokens_mask: Optional[torch.LongTensor] = None,
        prompt_embeds_text_only: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds_text_only: Optional[torch.FloatTensor] = None,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )
        #
        if prompt_embeds is not None and class_tokens_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `class_tokens_mask` also have to be passed. Make sure to generate `class_tokens_mask` from the same tokenizer that was used to generate `prompt_embeds`."
            )
        # check the input id images
        if input_id_images is None:
            raise ValueError(
                "Provide `input_id_images`. Cannot leave `input_id_images` undefined for PhotoMaker pipeline."
            )
        if not isinstance(input_id_images, list):
            input_id_images = [input_id_images]

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        assert do_classifier_free_guidance

        # 3. Encode input prompt
        num_id_images = len(input_id_images)

        (
            prompt_embeds,
            pooled_prompt_embeds,
            class_tokens_mask,
        ) = self.encode_prompt_with_trigger_word(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_id_images=num_id_images,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            class_tokens_mask=class_tokens_mask,
        )

        # 4. Encode input prompt without the trigger word for delayed conditioning
        prompt_text_only = prompt.replace(" "+self.trigger_word, "") # sensitive to white space
        (
            prompt_embeds_text_only,
            negative_prompt_embeds,
            pooled_prompt_embeds_text_only, # TODO: replace the pooled_prompt_embeds with text only prompt
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt_text_only,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds_text_only,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds_text_only,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )

        # 5. Prepare the input ID images
        dtype = next(self.id_encoder.parameters()).dtype
        if not isinstance(input_id_images[0], torch.Tensor):
            id_pixel_values = self.id_image_processor(input_id_images, return_tensors="pt").pixel_values

        id_pixel_values = id_pixel_values.unsqueeze(0).to(device=device, dtype=dtype) # TODO: multiple prompts

        # 6. Get the update text embedding with the stacked ID embedding
        prompt_embeds = self.id_encoder(id_pixel_values, prompt_embeds, class_tokens_mask)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

        # 7. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 8. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 9. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 10. Prepare added time ids & embeddings
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 11. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                if i <= start_merge_step:
                    current_prompt_embeds = torch.cat(
                        [negative_prompt_embeds, prompt_embeds_text_only], dim=0
                    )
                    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds_text_only], dim=0)
                else:
                    current_prompt_embeds = torch.cat(
                        [negative_prompt_embeds, prompt_embeds], dim=0
                    )
                    add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=current_prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # make sure the VAE is in float32 mode, as it overflows in float16
        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        if output_type != "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
            return StableDiffusionXLPipelineOutput(images=image)

        # apply watermark if available
        # if self.watermark is not None:
        #     image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)
