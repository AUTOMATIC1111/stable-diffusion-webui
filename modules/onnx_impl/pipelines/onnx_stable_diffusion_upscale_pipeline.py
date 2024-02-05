import inspect
from typing import Union, Optional, Callable, Any, List
import torch
import numpy as np
import diffusers
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_onnx_stable_diffusion_upscale import preprocess
from diffusers.image_processor import PipelineImageInput
from modules.onnx_impl.pipelines import CallablePipelineBase
from modules.onnx_impl.pipelines.utils import prepare_latents, randn_tensor


class OnnxStableDiffusionUpscalePipeline(diffusers.OnnxStableDiffusionUpscalePipeline, CallablePipelineBase):
    __module__ = 'diffusers'
    __name__ = 'OnnxStableDiffusionUpscalePipeline'

    def __init__(
        self,
        vae_encoder: diffusers.OnnxRuntimeModel,
        vae_decoder: diffusers.OnnxRuntimeModel,
        text_encoder: diffusers.OnnxRuntimeModel,
        tokenizer: Any,
        unet: diffusers.OnnxRuntimeModel,
        scheduler: Any,
        safety_checker: diffusers.OnnxRuntimeModel,
        feature_extractor: Any,
        requires_safety_checker: bool = True
    ):
        super().__init__(vae_encoder, vae_decoder, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, requires_safety_checker)

    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: PipelineImageInput = None,
        num_inference_steps: int = 75,
        guidance_scale: float = 9.0,
        noise_level: int = 20,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[np.ndarray] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        # 1. Check inputs
        self.check_inputs(
            prompt,
            image,
            noise_level,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if generator is None:
            generator = torch.Generator("cpu")

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        latents_dtype = prompt_embeds.dtype
        image = preprocess(image).cpu().numpy()
        height, width = image.shape[2:]

        latents = prepare_latents(
            self.scheduler.init_noise_sigma,
            batch_size * num_images_per_prompt,
            height,
            width,
            latents_dtype,
            generator,
        )

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 5. Add noise to image
        noise_level = np.array([noise_level]).astype(np.int64)
        noise = randn_tensor(
            image.shape,
            latents_dtype,
            generator,
        )

        image = self.low_res_scheduler.add_noise(
            torch.from_numpy(image), torch.from_numpy(noise), torch.from_numpy(noise_level)
        )
        image = image.numpy()

        batch_multiplier = 2 if do_classifier_free_guidance else 1
        image = np.concatenate([image] * batch_multiplier * num_images_per_prompt)
        noise_level = np.concatenate([noise_level] * image.shape[0])

        # 7. Check that sizes of image and latents match
        num_channels_image = image.shape[1]
        if self.num_latent_channels + num_channels_image != self.num_unet_input_channels:
            raise ValueError(
                "Incorrect configuration settings! The config of `pipeline.unet` expects"
                f" {self.num_unet_input_channels} but received `num_channels_latents`: {self.num_latent_channels} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {self.num_latent_channels + num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        timestep_dtype = next(
            (input.type for input in self.unet.model.get_inputs() if input.name == "timestep"), "tensor(float)"
        )
        timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = np.concatenate([latent_model_input, image], axis=1)

                # timestep to tensor
                timestep = np.array([t], dtype=timestep_dtype)

                # predict the noise residual
                noise_pred = self.unet(
                    sample=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    class_labels=noise_level,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
                ).prev_sample
                latents = latents.numpy()

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        has_nsfw_concept = None

        if output_type != "latent":
            # 10. Post-processing
            image = self.decode_latents(latents)

            # image = self.vae_decoder(latent_sample=latents)[0]
            # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
            image = np.concatenate(
                [self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
            )

            image = np.clip(image / 2 + 0.5, 0, 1)
            image = image.transpose((0, 2, 3, 1))

            if self.safety_checker is not None:
                safety_checker_input = self.feature_extractor(
                    self.numpy_to_pil(image), return_tensors="np"
                ).pixel_values.astype(image.dtype)

                images, has_nsfw_concept = [], []
                for i in range(image.shape[0]):
                    image_i, has_nsfw_concept_i = self.safety_checker(
                        clip_input=safety_checker_input[i : i + 1], images=image[i : i + 1]
                    )
                    images.append(image_i)
                    has_nsfw_concept.append(has_nsfw_concept_i[0])
                image = np.concatenate(images)

            if output_type == "pil":
                image = self.numpy_to_pil(image)
        else:
            image = latents

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
