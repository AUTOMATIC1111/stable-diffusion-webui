import json
from typing import Callable
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from .config import Config
from .defaults import GET_CONFIG_TIMEOUT, POST_TIMEOUT, STATE_URLERROR
from .utils import fix_prompt, img_to_b64


class Client:
    def __init__(self, cfg: Config, error_callback: Callable):
        """It is highly dependent on config's structure to the point it writes directly to it. :/"""
        self.cfg = cfg
        self.cb = error_callback

    def handle_api_error(self, exc: Exception):
        """Handle exceptions that can occur while interacting with the backend."""
        try:
            # conveniently allows error to bubble back up if not handled by here
            raise exc
        except URLError as e:
            self.cb(f"{STATE_URLERROR}: {e.reason}")
        except json.JSONDecodeError:
            self.cb(f"{STATE_URLERROR}: invalid JSON response")
        except ValueError:
            self.cb(f"{STATE_URLERROR}: Invalid backend URL")

    def post(self, route, body, base_url=...):
        base_url = self.cfg("base_url", str) if base_url is ... else base_url
        # FastAPI doesn't support urlencoded data transparently
        body = json.dumps(body).encode("utf-8")
        req = Request(urljoin(base_url, route))
        req.add_header("Content-Type", "application/json")
        req.add_header("Content-Length", str(len(body)))
        try:
            # TODO: how to cancel this? might as well refactor the API to be async...
            with urlopen(req, body, POST_TIMEOUT) as res:
                return json.loads(res.read())
        except Exception as e:
            self.handle_api_error(e)

    def get_common_params(self, has_selection):
        """Parameters nearly all the post routes share."""
        tiling = self.cfg("sd_tiling", bool) and not (
            self.cfg("only_full_img_tiling", bool) and has_selection
        )

        # its fine to stuff extra stuff here; pydantic will shave off irrelevant params
        params = dict(
            sd_model=self.cfg("sd_model", str),
            batch_count=self.cfg("sd_batch_count", int),
            batch_size=self.cfg("sd_batch_size", int),
            base_size=self.cfg("sd_base_size", int),
            max_size=self.cfg("sd_max_size", int),
            tiling=tiling,
            upscaler_name=self.cfg("upscaler_name", str),
            restore_faces=self.cfg("face_restorer_model", str) != "None",
            face_restorer=self.cfg("face_restorer_model", str),
            codeformer_weight=self.cfg("codeformer_weight", float),
            filter_nsfw=self.cfg("filter_nsfw", bool),
            do_exact_steps=self.cfg("do_exact_steps", bool),
            include_grid=self.cfg("include_grid", bool),
        )
        return params

    def get_config(self) -> bool:
        obj = None
        try:
            with urlopen(
                urljoin(self.cfg("base_url", str), "config"),
                None,
                GET_CONFIG_TIMEOUT,
            ) as res:
                obj = json.loads(res.read())
        except Exception as e:
            self.handle_api_error(e)
            return False

        try:
            assert "new_img" in obj
            assert "new_img_mask" in obj
            assert len(obj["upscalers"]) > 0
            assert len(obj["samplers"]) > 0
            assert len(obj["samplers_img2img"]) > 0
            assert len(obj["face_restorers"]) > 0
            assert len(obj["sd_models"]) > 0
        except:
            self.cb(
                f"{STATE_URLERROR}: incompatible response, are you running the right API?"
            )
            return False

        # replace only after verifying
        self.cfg.set("new_img_path", obj["new_img"])
        self.cfg.set("new_img_mask_path", obj["new_img_mask"])
        self.cfg.set("upscaler_list", obj["upscalers"])
        self.cfg.set("txt2img_sampler_list", obj["samplers"])
        self.cfg.set("img2img_sampler_list", obj["samplers_img2img"])
        self.cfg.set("inpaint_sampler_list", obj["samplers_img2img"])
        self.cfg.set("face_restorer_model_list", obj["face_restorers"])
        self.cfg.set("sd_model_list", obj["sd_models"])
        return True

    def post_txt2img(self, width, height, has_selection):
        params = dict(orig_width=width, orig_height=height)
        if not self.cfg("just_use_yaml", bool):
            seed = (
                self.cfg("txt2img_seed", int)
                if not self.cfg("txt2img_seed", str).strip() == ""
                else -1
            )
            params.update(self.get_common_params(has_selection))
            params.update(
                prompt=fix_prompt(self.cfg("txt2img_prompt", str)),
                negative_prompt=fix_prompt(self.cfg("txt2img_negative_prompt", str)),
                sampler_name=self.cfg("txt2img_sampler", str),
                steps=self.cfg("txt2img_steps", int),
                cfg_scale=self.cfg("txt2img_cfg_scale", float),
                seed=seed,
                highres_fix=self.cfg("txt2img_highres", bool),
                denoising_strength=self.cfg("txt2img_denoising_strength", float),
            )

        return self.post("/txt2img", params)

    def post_img2img(self, src_img, mask_img, has_selection):
        params = dict(
            mode=0, src_img=img_to_b64(src_img), mask_img=img_to_b64(mask_img)
        )
        if not self.cfg("just_use_yaml", bool):
            seed = (
                self.cfg("img2img_seed", int)
                if not self.cfg("img2img_seed", str).strip() == ""
                else -1
            )
            params.update(self.get_common_params(has_selection))
            params.update(
                prompt=fix_prompt(self.cfg("img2img_prompt", str)),
                negative_prompt=fix_prompt(self.cfg("img2img_negative_prompt", str)),
                sampler_name=self.cfg("img2img_sampler", str),
                steps=self.cfg("img2img_steps", int),
                cfg_scale=self.cfg("img2img_cfg_scale", float),
                denoising_strength=self.cfg("img2img_denoising_strength", float),
                color_correct=self.cfg("img2img_color_correct", bool),
                seed=seed,
            )

        return self.post("/img2img", params)

    def post_inpaint(self, src_img, mask_img, has_selection):
        params = dict(
            mode=1, src_img=img_to_b64(src_img), mask_img=img_to_b64(mask_img)
        )
        if not self.cfg("just_use_yaml", bool):
            seed = (
                self.cfg("inpaint_seed", int)
                if not self.cfg("inpaint_seed", str).strip() == ""
                else -1
            )
            fill = self.cfg("inpaint_fill_list", "QStringList").index(
                self.cfg("inpaint_fill", str)
            )
            params.update(self.get_common_params(has_selection))
            params.update(
                prompt=fix_prompt(self.cfg("inpaint_prompt", str)),
                negative_prompt=fix_prompt(self.cfg("inpaint_negative_prompt", str)),
                sampler_name=self.cfg("inpaint_sampler", str),
                steps=self.cfg("inpaint_steps", int),
                cfg_scale=self.cfg("inpaint_cfg_scale", float),
                denoising_strength=self.cfg("inpaint_denoising_strength", float),
                color_correct=self.cfg("inpaint_color_correct", bool),
                seed=seed,
                invert_mask=self.cfg("inpaint_invert_mask", bool),
                mask_blur=self.cfg("inpaint_mask_blur", int),
                inpainting_fill=fill,
                inpaint_full_res=self.cfg("inpaint_full_res", bool),
                inpaint_full_res_padding=self.cfg("inpaint_full_res_padding", int),
                include_grid=False,  # it is never useful for inpaint mode
            )

        return self.post("/img2img", params)

    def post_upscale(self, src_img):
        params = (
            {
                "src_img": img_to_b64(src_img),
                "upscaler_name": self.cfg("upscale_upscaler_name", str),
                "downscale_first": self.cfg("upscale_downscale_first", bool),
            }
            if not self.cfg("just_use_yaml", bool)
            else {"src_img": img_to_b64(src_img)}
        )
        return self.post("/upscale", params)
