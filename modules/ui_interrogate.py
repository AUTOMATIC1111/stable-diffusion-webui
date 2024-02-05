import os
import gradio as gr
import torch
from PIL import Image
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


def get_models():
    import open_clip
    return ['/'.join(x) for x in open_clip.list_pretrained()]


def load_interrogator(clip_model_name):
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


def interrogate_image(image, model, mode):
    shared.state.begin()
    shared.state.job = 'interrogate'
    try:
        if shared.backend == shared.Backend.ORIGINAL and (shared.cmd_opts.lowvram or shared.cmd_opts.medvram):
            lowvram.send_everything_to_cpu()
            devices.torch_gc()
        load_interrogator(model)
        image = image.convert('RGB')
        shared.log.info(f'Interrogate: image={image} mode={mode} config={ci.config}')
        prompt = interrogate(image, mode)
    except Exception as e:
        prompt = f"Exception {type(e)}"
        shared.log.error(f'Interrogate: {e}')
    shared.state.end()
    return prompt


def interrogate_batch(batch_files, batch_folder, batch_str, model, mode, write):
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
    shared.state.begin()
    shared.state.job = 'batch interrogate'
    prompts = []
    try:
        if shared.backend == shared.Backend.ORIGINAL and (shared.cmd_opts.lowvram or shared.cmd_opts.medvram):
            lowvram.send_everything_to_cpu()
            devices.torch_gc()
        load_interrogator(model)
        shared.log.info(f'Interrogate batch: images={len(files)} mode={mode} config={ci.config}')
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


def analyze_image(image, model):
    load_interrogator(model)
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
                    btn_interrogate_img = gr.Button("Interrogate", variant='primary')
                    btn_analyze_img = gr.Button("Analyze", variant='primary')
                    btn_unload = gr.Button("Unload")
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
                    btn_interrogate_batch = gr.Button("Interrogate", variant='primary')
        with gr.Column():
            with gr.Row():
                # clip_model = gr.Dropdown(get_models(), value='ViT-L-14/openai', label='CLIP Model')
                clip_model = gr.Dropdown([], value='ViT-L-14/openai', label='CLIP Model')
                ui_common.create_refresh_button(clip_model, get_models, lambda: {"choices": get_models()}, 'refresh_interrogate_models')
            with gr.Row():
                mode = gr.Radio(['best', 'fast', 'classic', 'caption', 'negative'], label='Mode', value='best')
        btn_interrogate_img.click(interrogate_image, inputs=[image, clip_model, mode], outputs=prompt)
        btn_analyze_img.click(analyze_image, inputs=[image, clip_model], outputs=[medium, artist, movement, trending, flavor])
        btn_interrogate_batch.click(interrogate_batch, inputs=[batch_files, batch_folder, batch_str, clip_model, mode, write], outputs=[batch])
        btn_unload.click(unload)
