import os
import base64
from io import BytesIO
import gradio as gr
import torch
from PIL import Image
from pydantic import BaseModel, Field # pylint: disable=no-name-in-module
from fastapi.exceptions import HTTPException
import modules.generation_parameters_copypaste as parameters_copypaste
from modules import devices, lowvram, shared, paths, ui_common


ci = None
low_vram = False


class BatchWriter:
    def __init__(self, folder):
        self.folder = folder
        self.csv, self.file = None, None

    def add(self, file, prompt):
        txt_file = os.path.splitext(file)[0] + ".txt"
        with open(os.path.join(self.folder, txt_file), 'w', encoding='utf-8') as f:
            f.write(prompt)

    def close(self):
        if self.file is not None:
            self.file.close()


def load(clip_model_name):
    from clip_interrogator import Config, Interrogator
    global ci # pylint: disable=global-statement
    if ci is None:
        config = Config(device=devices.get_optimal_device(), cache_path=os.path.join(paths.models_path, 'Interrogator'), clip_model_name=clip_model_name, quiet=True)
        if low_vram:
            config.apply_low_vram_defaults()
        shared.log.info(f'Interrogate load: config={config}')
        ci = Interrogator(config)
    elif clip_model_name != ci.config.clip_model_name:
        ci.config.clip_model_name = clip_model_name
        shared.log.info(f'Interrogate load: config={ci.config}')
        ci.load_clip_model()


def unload():
    if ci is not None:
        shared.log.debug('Interrogate offload')
        ci.caption_model = ci.caption_model.to(devices.cpu)
        ci.clip_model = ci.clip_model.to(devices.cpu)
        ci.caption_offloaded = True
        ci.clip_offloaded = True
        devices.torch_gc()


def image_analysis(image, clip_model_name):
    load(clip_model_name)
    image = image.convert('RGB')
    image_features = ci.image_to_features(image)
    top_mediums = ci.mediums.rank(image_features, 5)
    top_artists = ci.artists.rank(image_features, 5)
    top_movements = ci.movements.rank(image_features, 5)
    top_trendings = ci.trendings.rank(image_features, 5)
    top_flavors = ci.flavors.rank(image_features, 5)
    medium_ranks = dict(zip(top_mediums, ci.similarities(image_features, top_mediums)))
    artist_ranks = dict(zip(top_artists, ci.similarities(image_features, top_artists)))
    movement_ranks = dict(zip(top_movements, ci.similarities(image_features, top_movements)))
    trending_ranks = dict(zip(top_trendings, ci.similarities(image_features, top_trendings)))
    flavor_ranks = dict(zip(top_flavors, ci.similarities(image_features, top_flavors)))
    return medium_ranks, artist_ranks, movement_ranks, trending_ranks, flavor_ranks


def interrogate(image, mode, caption=None):
    shared.log.info(f'Interrogate: image={image} mode={mode} config={ci.config}')
    if mode == 'best':
        prompt = ci.interrogate(image, caption=caption)
    elif mode == 'caption':
        prompt = ci.generate_caption(image) if caption is None else caption
    elif mode == 'classic':
        prompt = ci.interrogate_classic(image, caption=caption)
    elif mode == 'fast':
        prompt = ci.interrogate_fast(image, caption=caption)
    elif mode == 'negative':
        prompt = ci.interrogate_negative(image)
    else:
        raise RuntimeError(f"Unknown mode {mode}")
    return prompt


def image_to_prompt(image, mode, clip_model_name):
    shared.state.begin()
    shared.state.job = 'interrogate'
    try:
        if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
            lowvram.send_everything_to_cpu()
            devices.torch_gc()
        load(clip_model_name)
        image = image.convert('RGB')
        shared.log.info(f'Interrogate: image={image} mode={mode} config={ci.config}')
        prompt = interrogate(image, mode)
    except Exception as e:
        prompt = f"Exception {type(e)}"
        shared.log.error(f'Interrogate: {e}')
    shared.state.end()
    return prompt


def get_models():
    import open_clip
    return ['/'.join(x) for x in open_clip.list_pretrained()]


def batch_process(batch_files, batch_folder, batch_str, mode, clip_model, write):
    files = []
    if batch_files is not None:
        files += [f.name for f in batch_files]
    if batch_folder is not None:
        files += [f.name for f in batch_folder]
    if batch_str is not None and len(batch_str) > 0 and os.path.exists(batch_str) and os.path.isdir(batch_str):
        files += [os.path.join(batch_str, f) for f in os.listdir(batch_str) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    if len(files) == 0:
        shared.log.error('Interrogate batch no images')
        return ''
    shared.log.info(f'Interrogate batch: images={len(files)} mode={mode} config={ci.config}')
    shared.state.begin()
    shared.state.job = 'batch interrogate'
    prompts = []
    try:
        if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
            lowvram.send_everything_to_cpu()
            devices.torch_gc()
        load(clip_model)
        captions = []
        # first pass: generate captions
        for file in files:
            caption = ""
            try:
                if shared.state.interrupted:
                    break
                image = Image.open(file).convert('RGB')
                caption = ci.generate_caption(image)
            except Exception as e:
                shared.log.error(f'Interrogate caption: {e}')
            finally:
                captions.append(caption)
        # second pass: interrogate
        if write:
            writer = BatchWriter(os.path.dirname(files[0]))
        for idx, file in enumerate(files):
            try:
                if shared.state.interrupted:
                    break
                image = Image.open(file).convert('RGB')
                prompt = interrogate(image, mode, caption=captions[idx])
                prompts.append(prompt)
                if write:
                    writer.add(file, prompt)
            except OSError as e:
                shared.log.error(f'Interrogate batch: {e}')
        if write:
            writer.close()
        ci.config.quiet = False
        unload()
    except Exception as e:
        shared.log.error(f'Interrogate batch: {e}')
    shared.state.end()
    return '\n\n'.join(prompts)


def create_ui():
    global low_vram # pylint: disable=global-statement
    low_vram = shared.cmd_opts.lowvram or shared.cmd_opts.medvram
    if not low_vram and torch.cuda.is_available():
        device = devices.get_optimal_device()
        vram_total = torch.cuda.get_device_properties(device).total_memory
        if vram_total <= 12*1024*1024*1024:
            low_vram = True
    with gr.Row(elem_id="interrogate_tab"):
        with gr.Column():
            with gr.Tab("Image"):
                with gr.Row():
                    image = gr.Image(type='pil', label="Image")
                with gr.Row():
                    prompt = gr.Textbox(label="Prompt", lines=3)
                with gr.Row():
                    medium = gr.Label(label="Medium", num_top_classes=5)
                    artist = gr.Label(label="Artist", num_top_classes=5)
                    movement = gr.Label(label="Movement", num_top_classes=5)
                    trending = gr.Label(label="Trending", num_top_classes=5)
                    flavor = gr.Label(label="Flavor", num_top_classes=5)
                with gr.Row():
                    interrogate_btn = gr.Button("Interrogate", variant='primary')
                    analyze_btn = gr.Button("Analyze", variant='primary')
                    unload_btn = gr.Button("Unload")
                with gr.Row():
                    buttons = parameters_copypaste.create_buttons(["txt2img", "img2img", "extras", "control"])
                for tabname, button in buttons.items():
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(paste_button=button, tabname=tabname, source_text_component=prompt, source_image_component=image,))
            with gr.Tab("Batch"):
                with gr.Row():
                    batch_files = gr.File(label="Files", show_label=True, file_count='multiple', file_types=['image'], type='file', interactive=True, height=100)
                with gr.Row():
                    batch_folder = gr.File(label="Folder", show_label=True, file_count='directory', file_types=['image'], type='file', interactive=True, height=100)
                with gr.Row():
                    batch_str = gr.Text(label="Folder", value="", interactive=True)
                with gr.Row():
                    batch = gr.Text(label="Prompts", lines=10)
                with gr.Row():
                    write = gr.Checkbox(label='Write prompts to files', value=False)
                with gr.Row():
                    batch_btn = gr.Button("Interrogate", variant='primary')
        with gr.Column():
            with gr.Row():
                # clip_model = gr.Dropdown(get_models(), value='ViT-L-14/openai', label='CLIP Model')
                clip_model = gr.Dropdown([], value='ViT-L-14/openai', label='CLIP Model')
                ui_common.create_refresh_button(clip_model, get_models, lambda: {"choices": get_models()}, 'refresh_interrogate_models')
            with gr.Row():
                mode = gr.Radio(['best', 'fast', 'classic', 'caption', 'negative'], label='Mode', value='best')
        interrogate_btn.click(image_to_prompt, inputs=[image, mode, clip_model], outputs=prompt)
        analyze_btn.click(image_analysis, inputs=[image, clip_model], outputs=[medium, artist, movement, trending, flavor])
        unload_btn.click(unload)
        batch_btn.click(batch_process, inputs=[batch_files, batch_folder, batch_str, mode, clip_model, write], outputs=[batch])


def decode_base64_to_image(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as e:
        raise HTTPException(status_code=500, detail="Invalid encoded image") from e


def mount_interrogator_api(_: gr.Blocks, app): # TODO redesign interrogator api

    class InterrogatorAnalyzeRequest(BaseModel):
        image: str = Field(default="", title="Image", description="Image to work on, must be a Base64 string containing the image's data.")
        clip_model_name: str = Field(default="ViT-L-14/openai", title="Model", description="The interrogate model used. See the models endpoint for a list of available models.")

    class InterrogatorPromptRequest(InterrogatorAnalyzeRequest):
        mode: str = Field(default="fast", title="Mode", description="The mode used to generate the prompt. Can be one of: best, fast, classic, negative.")

    @app.get("/interrogator/models")
    async def api_get_models():
        import open_clip
        return ["/".join(x) for x in open_clip.list_pretrained()]

    @app.post("/interrogator/prompt")
    async def api_get_prompt(analyzereq: InterrogatorPromptRequest):
        image_b64 = analyzereq.image
        if image_b64 is None:
            raise HTTPException(status_code=404, detail="Image not found")
        img = decode_base64_to_image(image_b64)
        prompt = image_to_prompt(img, analyzereq.mode, analyzereq.clip_model_name)
        return {"prompt": prompt}

    @app.post("/interrogator/analyze")
    async def api_analyze(analyzereq: InterrogatorAnalyzeRequest):
        image_b64 = analyzereq.image
        if image_b64 is None:
            raise HTTPException(status_code=404, detail="Image not found")

        img = decode_base64_to_image(image_b64)
        (medium_ranks, artist_ranks, movement_ranks, trending_ranks, flavor_ranks) = image_analysis(img, analyzereq.clip_model_name)
        return {"medium": medium_ranks, "artist": artist_ranks, "movement": movement_ranks, "trending": trending_ranks, "flavor": flavor_ranks}

# script_callbacks.on_app_started(mount_interrogator_api)
