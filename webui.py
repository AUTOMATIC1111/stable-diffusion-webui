import os
import threading

from modules.paths import script_path

import signal

from modules.shared import opts, cmd_opts, state
import modules.shared as shared
import modules.ui
import modules.scripts
import modules.sd_hijack
import modules.codeformer_model
import modules.gfpgan_model
import modules.face_restoration
import modules.realesrgan_model as realesrgan
import modules.esrgan_model as esrgan
import modules.extras
import modules.lowvram
import modules.txt2img
import modules.img2img
import modules.sd_models


modules.codeformer_model.setup_codeformer()
modules.gfpgan_model.setup_gfpgan()
shared.face_restorers.append(modules.face_restoration.FaceRestoration())

esrgan.load_models(cmd_opts.esrgan_models_path)
realesrgan.setup_realesrgan()

queue_lock = threading.Lock()


def wrap_queued_call(func):
    def f(*args, **kwargs):
        with queue_lock:
            res = func(*args, **kwargs)

        return res

    return f


def wrap_gradio_gpu_call(func):
    def f(*args, **kwargs):
        shared.state.sampling_step = 0
        shared.state.job_count = -1
        shared.state.job_no = 0
        shared.state.current_latent = None
        shared.state.current_image = None
        shared.state.current_image_sampling_step = 0

        with queue_lock:
            res = func(*args, **kwargs)

        shared.state.job = ""
        shared.state.job_count = 0

        return res

    return modules.ui.wrap_gradio_call(f)


modules.scripts.load_scripts(os.path.join(script_path, "scripts"))

shared.sd_model = modules.sd_models.load_model()
shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights(shared.sd_model)))


def webui():
    if not cmd_opts.api:
        # make the program just exit at ctrl+c without waiting for anything
        def sigint_handler(sig, frame):
            print(f'Interrupted with signal {sig} in {frame}')
            os._exit(0)

        signal.signal(signal.SIGINT, sigint_handler)

        demo = modules.ui.create_ui(
            txt2img=wrap_gradio_gpu_call(modules.txt2img.txt2img),
            img2img=wrap_gradio_gpu_call(modules.img2img.img2img),
            run_extras=wrap_gradio_gpu_call(modules.extras.run_extras),
            run_pnginfo=modules.extras.run_pnginfo
        )

        demo.launch(
            share=cmd_opts.share,
            server_name="0.0.0.0" if cmd_opts.listen else None,
            server_port=cmd_opts.port,
            debug=cmd_opts.gradio_debug,
            auth=[tuple(cred.split(':')) for cred in cmd_opts.gradio_auth.strip('"').split(',')] if cmd_opts.gradio_auth else None,
            inbrowser=cmd_opts.autolaunch,
        )
    else:
        import uvicorn
        from fastapi import FastAPI, Body
        from pydantic import BaseModel, Field
        import json
        import io
        import base64

        app = FastAPI()

        class TextToImage(BaseModel):
            prompt: str = Field(..., title="Prompt Text", description="The text to generate an image from.")
            negative_prompt: str = Field(default="", title="Negative Prompt Text")
            prompt_style: str = Field(default="None", title="Prompt Style")
            prompt_style2: str = Field(default="None", title="Prompt Style 2")
            steps: int = Field(default=20, title="Steps")
            sampler_index: int = Field(0, title="Sampler Index")
            restore_faces: bool = Field(default=False, title="Restore Faces")
            tiling: bool = Field(default=False, title="Tiling")
            n_iter: int = Field(default=1, title="N Iter")
            batch_size: int = Field(default=1, title="Batch Size")
            cfg_scale: float = Field(default=7, title="Config Scale")
            seed: int = Field(default=-1.0, title="Seed")
            subseed: int = Field(default=-1.0, title="Subseed")
            subseed_strength: float = Field(default=0, title="Subseed Strength")
            seed_resize_from_h: int = Field(default=0, title="Seed Resize From Height")
            seed_resize_from_w: int = Field(default=0, title="Seed Resize From Width")
            height: int = Field(default=512, title="Height")
            width: int = Field(default=512, title="Width")
            enable_hr: bool = Field(default=False, title="Enable HR")
            scale_latent: bool = Field(default=True, title="Scale Latent")
            denoising_strength: float = Field(default=0.7, title="Denoising Strength")

        class TextToImageResponse(BaseModel):
            images: list[str] = Field(default=None, title="Image", description="The generated image in base64 format.")
            all_prompts: list[str] = Field(default=None, title="All Prompts", description="The prompt text.")
            negative_prompt: str = Field(default=None, title="Negative Prompt Text")
            seed: int = Field(default=None, title="Seed")
            all_seeds: list[int] = Field(default=None, title="All Seeds")
            subseed: int = Field(default=None, title="Subseed")
            all_subseeds: list[int] = Field(default=None, title="All Subseeds")
            subseed_strength: float = Field(default=None, title="Subseed Strength")
            width: int = Field(default=None, title="Width")
            height: int = Field(default=None, title="Height")
            sampler_index: int = Field(default=None, title="Sampler Index")
            sampler: str = Field(default=None, title="Sampler")
            cfg_scale: float = Field(default=None, title="Config Scale")
            steps: int = Field(default=None, title="Steps")
            batch_size: int = Field(default=None, title="Batch Size")
            restore_faces: bool = Field(default=None, title="Restore Faces")
            face_restoration_model: str = Field(default=None, title="Face Restoration Model")
            sd_model_hash: str = Field(default=None, title="SD Model Hash")
            seed_resize_from_w: int = Field(default=None, title="Seed Resize From Width")
            seed_resize_from_h: int = Field(default=None, title="Seed Resize From Height")
            denoising_strength: float = Field(default=None, title="Denoising Strength")
            extra_generation_params: dict = Field(default={}, title="Extra Generation Params")
            index_of_first_image: int = Field(default=None, title="Index of First Image")
            html: str = Field(default=None, title="HTML")

        @app.get("/txt2img", response_model=TextToImageResponse)
        def text2img(txt2img: TextToImage = Body(embed=True)):
            resp = modules.txt2img.txt2img(*[v for v in txt2img.dict().values()], 0, False, None, '', False, 1, '', 4, '', True)
            images = []
            for i in resp[0]:
                buffer = io.BytesIO()
                i.save(buffer, format="png")
                images.append(base64.b64encode(buffer.getvalue()))
            resp_params = json.loads(resp[1])

            return TextToImageResponse(images=images, **resp_params, html=resp[2])

        print("Starting server...")
        uvicorn.run(app,
                    host="0.0.0.0" if cmd_opts.listen else "localhost",
                    port=cmd_opts.port if cmd_opts.port else 7861)


if __name__ == "__main__":
    webui()
