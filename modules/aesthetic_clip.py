import itertools
import os
from pathlib import Path
import html
import gc

import gradio as gr
import torch
from PIL import Image
from modules import shared
from modules.shared import device
from transformers import CLIPModel, CLIPProcessor

from tqdm.auto import tqdm


def get_all_images_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if
            os.path.isfile(os.path.join(folder, f)) and check_is_valid_image_file(f)]


def check_is_valid_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', ".gif", ".tiff", ".webp"))


def batched(dataset, total, n=1):
    for ndx in range(0, total, n):
        yield [dataset.__getitem__(i) for i in range(ndx, min(ndx + n, total))]


def iter_to_batched(iterable, n=1):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def generate_imgs_embd(name, folder, batch_size):
    # clipModel = CLIPModel.from_pretrained(
    #     shared.sd_model.cond_stage_model.clipModel.name_or_path
    # )
    model = CLIPModel.from_pretrained(shared.sd_model.cond_stage_model.clipModel.name_or_path).to(device)
    processor = CLIPProcessor.from_pretrained(shared.sd_model.cond_stage_model.clipModel.name_or_path)

    with torch.no_grad():
        embs = []
        for paths in tqdm(iter_to_batched(get_all_images_in_folder(folder), batch_size),
                          desc=f"Generating embeddings for {name}"):
            if shared.state.interrupted:
                break
            inputs = processor(images=[Image.open(path) for path in paths], return_tensors="pt").to(device)
            outputs = model.get_image_features(**inputs).cpu()
            embs.append(torch.clone(outputs))
            inputs.to("cpu")
            del inputs, outputs

        embs = torch.cat(embs, dim=0).mean(dim=0, keepdim=True)

        # The generated embedding will be located here
        path = str(Path(shared.cmd_opts.aesthetic_embeddings_dir) / f"{name}.pt")
        torch.save(embs, path)

        model = model.cpu()
        del model
        del processor
        del embs
        gc.collect()
        torch.cuda.empty_cache()
        res = f"""
        Done generating embedding for {name}!
        Aesthetic embedding saved to {html.escape(path)}
        """
        shared.update_aesthetic_embeddings()
        return gr.Dropdown.update(choices=sorted(shared.aesthetic_embeddings.keys()), label="Imgs embedding",
                           value=sorted(shared.aesthetic_embeddings.keys())[0] if len(
                               shared.aesthetic_embeddings) > 0 else None), res, ""
