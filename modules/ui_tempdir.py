import os
import tempfile
from collections import namedtuple
from pathlib import Path
import gradio as gr
from PIL import PngImagePlugin
from modules import shared


Savedfile = namedtuple("Savedfile", ["name"])


def register_tmp_file(gradio, filename):
    if hasattr(gradio, 'temp_file_sets'):
        gradio.temp_file_sets[0] = gradio.temp_file_sets[0] | {os.path.abspath(filename)}


def check_tmp_file(gradio, filename):
    ok = False
    if hasattr(gradio, 'temp_file_sets'):
        ok = ok or any([filename in fileset for fileset in gradio.temp_file_sets])
    if shared.opts.outdir_samples != '':
        ok = ok or Path(shared.opts.outdir_samples).resolve() in Path(filename).resolve().parents
    else:
        ok = ok or Path(shared.opts.outdir_txt2img_samples).resolve() in Path(filename).resolve().parents
        ok = ok or Path(shared.opts.outdir_img2img_samples).resolve() in Path(filename).resolve().parents
        ok = ok or Path(shared.opts.outdir_extras_samples).resolve() in Path(filename).resolve().parents
    if shared.opts.outdir_grids != '':
        ok = ok or Path(shared.opts.outdir_grids).resolve() in Path(filename).resolve().parents
    else:
        ok = ok or Path(shared.opts.outdir_txt2img_grids).resolve() in Path(filename).resolve().parents
        ok = ok or Path(shared.opts.outdir_img2img_grids).resolve() in Path(filename).resolve().parents
    ok = ok or Path(shared.opts.outdir_save).resolve() in Path(filename).resolve().parents
    ok = ok or Path(shared.opts.outdir_init_images).resolve() in Path(filename).resolve().parents
    return ok


def pil_to_temp_file(self, img, dir: str, format="png") -> str: # pylint: disable=redefined-builtin,unused-argument
    """
    # original gradio implementation
    bytes_data = gr.processing_utils.encode_pil_to_bytes(img, format)
    temp_dir = Path(dir) / self.hash_bytes(bytes_data)
    temp_dir.mkdir(exist_ok=True, parents=True)
    filename = str(temp_dir / f"image.{format}")
    img.save(filename, pnginfo=gr.processing_utils.get_pil_metadata(img))
    """
    already_saved_as = getattr(img, 'already_saved_as', None)
    if already_saved_as and os.path.isfile(already_saved_as):
        register_tmp_file(shared.demo, already_saved_as)
        file_obj = Savedfile(already_saved_as)
        name = file_obj.name
        return name
    if shared.opts.temp_dir != "":
        dir = shared.opts.temp_dir
    use_metadata = False
    metadata = PngImagePlugin.PngInfo()
    for key, value in img.info.items():
        if isinstance(key, str) and isinstance(value, str):
            metadata.add_text(key, value)
            use_metadata = True
    file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=dir)
    img.save(file_obj, pnginfo=(metadata if use_metadata else None))
    name = file_obj.name
    shared.log.debug(f'Saving temp image: {name}')
    return name


# override save to file function so that it also writes PNG info
# gr.processing_utils.save_pil_to_file = save_pil_to_file # gradio <=3.31.0
gr.components.IOComponent.pil_to_temp_file = pil_to_temp_file      # gradio >=3.32.0

def on_tmpdir_changed():
    if shared.opts.temp_dir == "":
        return
    register_tmp_file(shared.demo, os.path.join(shared.opts.temp_dir, "x"))


def cleanup_tmpdr():
    temp_dir = shared.opts.temp_dir
    if temp_dir == "" or not os.path.isdir(temp_dir):
        return
    for root, _dirs, files in os.walk(temp_dir, topdown=False):
        for name in files:
            _, extension = os.path.splitext(name)
            if extension != ".png" and extension != ".jpg" and extension != ".webp":
                continue
            filename = os.path.join(root, name)
            os.remove(filename)
