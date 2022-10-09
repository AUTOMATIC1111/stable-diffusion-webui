import json
from modules.sd_samplers import samplers
from modules.shared import State, opts
import modules.shared as shared


def create_infotext(p, all_prompts, all_seeds, all_subseeds, comments, iteration=0, position_in_batch=0):
    index = position_in_batch + iteration * p.batch_size

    clip_skip = getattr(p, 'clip_skip', opts.CLIP_ignore_last_layers)

    generation_params = {
        "Steps": p.steps,
        "Sampler": samplers[p.sampler_index].name,
        "CFG scale": p.cfg_scale,
        "Seed": all_seeds[index],
        "Face restoration": (opts.face_restoration_model if p.restore_faces else None),
        "Size": f"{p.width}x{p.height}",
        "Model hash": getattr(p, 'sd_model_hash', None if not opts.add_model_hash_to_info or not shared.sd_model.sd_model_hash else shared.sd_model.sd_model_hash),
        "Batch size": (None if p.batch_size < 2 else p.batch_size),
        "Batch pos": (None if p.batch_size < 2 else position_in_batch),
        "Variation seed": (None if p.subseed_strength == 0 else all_subseeds[index]),
        "Variation seed strength": (None if p.subseed_strength == 0 else p.subseed_strength),
        "Seed resize from": (None if p.seed_resize_from_w == 0 or p.seed_resize_from_h == 0 else f"{p.seed_resize_from_w}x{p.seed_resize_from_h}"),
        "Denoising strength": getattr(p, 'denoising_strength', None),
        "Eta": (None if p.sampler is None or p.sampler.eta == p.sampler.default_eta else p.sampler.eta),
        "Clip skip": None if clip_skip == 0 else clip_skip,
    }

    generation_params.update(p.extra_generation_params)

    generation_params_text = ", ".join(
        [k if k == v else f'{k}: {v}' for k, v in generation_params.items() if v is not None])

    negative_prompt_text = "\nNegative prompt: " + \
        p.negative_prompt if p.negative_prompt else ""

    return f"{all_prompts[index]}{negative_prompt_text}\n{generation_params_text}".strip()


class StableDiffusionProcessing:
    def __init__(self, sd_model=None, outpath_samples=None, outpath_grids=None, prompt="", styles=None, seed=-1, subseed=-1, subseed_strength=0, seed_resize_from_h=-1, seed_resize_from_w=-1, seed_enable_extras=True, sampler_index=0, batch_size=1, n_iter=1, steps=50, cfg_scale=7.0, width=512, height=512, restore_faces=False, tiling=False, do_not_save_samples=False, do_not_save_grid=False, extra_generation_params=None, overlay_images=None, negative_prompt=None, eta=None):
        self.sd_model = sd_model
        self.outpath_samples: str = outpath_samples
        self.outpath_grids: str = outpath_grids
        self.prompt: str = prompt
        self.prompt_for_display: str = None
        self.negative_prompt: str = (negative_prompt or "")
        self.styles: list = styles or []
        self.seed: int = seed
        self.subseed: int = subseed
        self.subseed_strength: float = subseed_strength
        self.seed_resize_from_h: int = seed_resize_from_h
        self.seed_resize_from_w: int = seed_resize_from_w
        self.sampler_index: int = sampler_index
        self.batch_size: int = batch_size
        self.n_iter: int = n_iter
        self.steps: int = steps
        self.cfg_scale: float = cfg_scale
        self.width: int = width
        self.height: int = height
        self.restore_faces: bool = restore_faces
        self.tiling: bool = tiling
        self.do_not_save_samples: bool = do_not_save_samples
        self.do_not_save_grid: bool = do_not_save_grid
        self.extra_generation_params: dict = extra_generation_params or {}
        self.overlay_images = overlay_images
        self.eta = eta
        self.paste_to = None
        self.color_corrections = None
        self.denoising_strength: float = 0
        self.sampler_noise_scheduler_override = None
        self.ddim_discretize = opts.ddim_discretize
        self.s_churn = opts.s_churn
        self.s_tmin = opts.s_tmin
        self.s_tmax = float('inf')  # not representable as a standard ui option
        self.s_noise = opts.s_noise

        if not seed_enable_extras:
            self.subseed = -1
            self.subseed_strength = 0
            self.seed_resize_from_h = 0
            self.seed_resize_from_w = 0

    def init(self, all_prompts, all_seeds, all_subseeds):
        pass

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
        raise NotImplementedError()
        
class Processed:
    def __init__(self, p: StableDiffusionProcessing, images_list, seed=-1, info="", subseed=None, all_prompts=None, all_seeds=None, all_subseeds=None, index_of_first_image=0, infotexts=None):
        self.images = images_list
        self.prompt = p.prompt
        self.negative_prompt = p.negative_prompt
        self.seed = seed
        self.subseed = subseed
        self.subseed_strength = p.subseed_strength
        self.info = info
        self.width = p.width
        self.height = p.height
        self.sampler_index = p.sampler_index
        self.sampler = samplers[p.sampler_index].name
        self.cfg_scale = p.cfg_scale
        self.steps = p.steps
        self.batch_size = p.batch_size
        self.restore_faces = p.restore_faces
        self.face_restoration_model = opts.face_restoration_model if p.restore_faces else None
        self.sd_model_hash = shared.sd_model.sd_model_hash
        self.seed_resize_from_w = p.seed_resize_from_w
        self.seed_resize_from_h = p.seed_resize_from_h
        self.denoising_strength = getattr(p, 'denoising_strength', None)
        self.extra_generation_params = p.extra_generation_params
        self.index_of_first_image = index_of_first_image
        self.styles = p.styles
        self.job_timestamp = shared.state.job_timestamp
        self.clip_skip = opts.CLIP_ignore_last_layers

        self.eta = p.eta
        self.ddim_discretize = p.ddim_discretize
        self.s_churn = p.s_churn
        self.s_tmin = p.s_tmin
        self.s_tmax = p.s_tmax
        self.s_noise = p.s_noise
        self.sampler_noise_scheduler_override = p.sampler_noise_scheduler_override
        self.prompt = self.prompt if type(
            self.prompt) != list else self.prompt[0]
        self.negative_prompt = self.negative_prompt if type(
            self.negative_prompt) != list else self.negative_prompt[0]
        self.seed = int(self.seed if type(self.seed) != list else self.seed[0])
        self.subseed = int(self.subseed if type(
            self.subseed) != list else self.subseed[0]) if self.subseed is not None else -1

        self.all_prompts = all_prompts or [self.prompt]
        self.all_seeds = all_seeds or [self.seed]
        self.all_subseeds = all_subseeds or [self.subseed]
        self.infotexts = infotexts or [info]

    def js(self):
        obj = {
            "prompt": self.prompt,
            "all_prompts": self.all_prompts,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "all_seeds": self.all_seeds,
            "subseed": self.subseed,
            "all_subseeds": self.all_subseeds,
            "subseed_strength": self.subseed_strength,
            "width": self.width,
            "height": self.height,
            "sampler_index": self.sampler_index,
            "sampler": self.sampler,
            "cfg_scale": self.cfg_scale,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "restore_faces": self.restore_faces,
            "face_restoration_model": self.face_restoration_model,
            "sd_model_hash": self.sd_model_hash,
            "seed_resize_from_w": self.seed_resize_from_w,
            "seed_resize_from_h": self.seed_resize_from_h,
            "denoising_strength": self.denoising_strength,
            "extra_generation_params": self.extra_generation_params,
            "index_of_first_image": self.index_of_first_image,
            "infotexts": self.infotexts,
            "styles": self.styles,
            "job_timestamp": self.job_timestamp,
            "clip_skip": self.clip_skip,
        }

        return json.dumps(obj)

    def infotext(self,  p: StableDiffusionProcessing, index):
        return create_infotext(p, self.all_prompts, self.all_seeds, self.all_subseeds, comments=[], position_in_batch=index % self.batch_size, iteration=index // self.batch_size)
