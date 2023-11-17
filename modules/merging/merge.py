import gc
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Tuple

import safetensors.torch
import torch
from tqdm import tqdm

import modules.memstats
from modules.shared import log
from modules.merging import merge_methods
from modules.merging.merge_utils import WeightClass
from modules.merging.merge_model import SDModel
from modules.merging.merge_rebasin import (
    apply_permutation,
    sdunet_permutation_spec,
    update_model_a,
    weight_matching,
)

MAX_TOKENS = 77


KEY_POSITION_IDS = ".".join(
    [
        "cond_stage_model",
        "transformer",
        "text_model",
        "embeddings",
        "position_ids",
    ]
)


NAI_KEYS = {
    "cond_stage_model.transformer.embeddings.": "cond_stage_model.transformer.text_model.embeddings.",
    "cond_stage_model.transformer.encoder.": "cond_stage_model.transformer.text_model.encoder.",
    "cond_stage_model.transformer.final_layer_norm.": "cond_stage_model.transformer.text_model.final_layer_norm.",
}


def fix_clip(model: Dict) -> Dict:
    if KEY_POSITION_IDS in model.keys():
        model[KEY_POSITION_IDS] = torch.tensor(
            [list(range(MAX_TOKENS))],
            dtype=torch.int64,
            device=model[KEY_POSITION_IDS].device,
        )

    return model


def fix_key(model: Dict, key: str) -> Dict:
    for nk in NAI_KEYS:
        if key.startswith(nk):
            model[key.replace(nk, NAI_KEYS[nk])] = model[key]
            del model[key]

    return model


# https://github.com/j4ded/sdweb-merge-block-weighted-gui/blob/master/scripts/mbw/merge_block_weighted.py#L115
def fix_model(model: Dict) -> Dict:
    for k in model.keys():
        model = fix_key(model, k)
    return fix_clip(model)


def load_sd_model(model: os.PathLike | str, device: str = "cpu") -> Dict:
    if isinstance(model, str):
        model = Path(model)

    return SDModel(model, device).load_model()


def prune_sd_model(model: Dict) -> Dict:
    keys = list(model.keys())
    for k in keys:
        if (
            not k.startswith("model.diffusion_model.")
            and not k.startswith("first_stage_model.")
            and not k.startswith("cond_stage_model.")
        ):
            del model[k]
    return model


def restore_sd_model(original_model: Dict, merged_model: Dict) -> Dict:
    for k in original_model:
        if k not in merged_model:
            merged_model[k] = original_model[k]
    return merged_model


def log_vram(txt=""):
    log.info(f"{txt} VRAM: {modules.memstats.memory_stats()}")


def load_thetas(
    models: Dict[str, os.PathLike | str],
    prune: bool,
    device: str,
    precision: str,
) -> Dict:
    log_vram("before loading models")
    if prune:
        thetas = {k: prune_sd_model(load_sd_model(m, "cpu")) for k, m in models.items()}
    else:
        thetas = {k: load_sd_model(m, device) for k, m in models.items()}

    if device == "cuda":
        for model_key, model in thetas.items():
            for key, block in model.items():
                if precision == "fp16":
                    thetas[model_key].update({key: block.to(device).half()})
                else:
                    thetas[model_key].update({key: block.to(device)})

    log_vram("models loaded")
    return thetas


def merge_models(
    models: Dict[str, os.PathLike | str],
    merge_mode: str,
    precision: str = "full",
    weights_clip: bool = False,
    re_basin: bool = False,
    device: str = "cpu",
    work_device: Optional[str] = None,
    prune: bool = False,
    threads: int = 1,
    **kwargs,
) -> Dict:
    iterations = kwargs.get("re_basin_iterations", 1)
    thetas = load_thetas(models, prune, device, precision)

    log.info(f"start merging with {merge_mode} method")
    weight_matcher = WeightClass(thetas["model_a"], **kwargs)
    if re_basin:
        merged = rebasin_merge(
            thetas,
            weight_matcher,
            merge_mode,
            precision=precision,
            weights_clip=weights_clip,
            iterations=iterations,
            device=device,
            work_device=work_device,
            threads=threads,
        )
    else:
        merged = simple_merge(
            thetas,
            weight_matcher,
            merge_mode,
            precision=precision,
            weights_clip=weights_clip,
            device=device,
            work_device=work_device,
            threads=threads,
        )

    return un_prune_model(merged, thetas, models, device, prune, precision)


def un_prune_model(
    merged: Dict,
    thetas: Dict,
    models: Dict,
    device: str,
    prune: bool,
    precision: str,
) -> Dict:
    if prune:
        log.info("Un-pruning merged model")
        del thetas
        gc.collect()
        log_vram("remove thetas")
        original_a = load_sd_model(models["model_a"], device)
        for key in tqdm(original_a.keys(), desc="un-prune model a"):
            if KEY_POSITION_IDS in key:
                continue
            if "model" in key and key not in merged.keys():
                merged.update({key: original_a[key]})
                if precision == "fp16":
                    merged.update({key: merged[key].half()})
        del original_a
        gc.collect()
        # log_vram("remove original_a")
        original_b = load_sd_model(models["model_b"], device)
        for key in tqdm(original_b.keys(), desc="un-prune model b"):
            if KEY_POSITION_IDS in key:
                continue
            if "model" in key and key not in merged.keys():
                merged.update({key: original_b[key]})
                if precision == "fp16":
                    merged.update({key: merged[key].half()})
        del original_b

    return fix_model(merged)


def simple_merge(
    thetas: Dict[str, Dict],
    weight_matcher: WeightClass,
    merge_mode: str,
    precision: str = "fp16",
    weights_clip: bool = False,
    device: str = "cpu",
    work_device: Optional[str] = None,
    threads: int = 1,
) -> Dict:
    futures = []
    with tqdm(thetas["model_a"].keys(), desc="stage 1") as progress:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            for key in thetas["model_a"].keys():
                future = executor.submit(
                    simple_merge_key,
                    progress,
                    key,
                    thetas,
                    weight_matcher,
                    merge_mode,
                    precision,
                    weights_clip,
                    device,
                    work_device,
                )
                futures.append(future)

        for res in futures:
            res.result()

    log_vram("after stage 1")

    for key in tqdm(thetas["model_b"].keys(), desc="stage 2"):
        if KEY_POSITION_IDS in key:
            continue
        if "model" in key and key not in thetas["model_a"].keys():
            thetas["model_a"].update({key: thetas["model_b"][key]})
            if precision == 16:
                thetas["model_a"].update({key: thetas["model_a"][key].half()})

    log_vram("after stage 2")

    return fix_model(thetas["model_a"])


def rebasin_merge(
    thetas: Dict[str, os.PathLike | str],
    weight_matcher: WeightClass,
    merge_mode: str,
    precision: str = "fp16",
    weights_clip: bool = False,
    iterations: int = 1,
    device="cpu",
    work_device=None,
    threads: int = 1,
):
    # WARNING: not sure how this does when 3 models are involved...

    model_a = thetas["model_a"].clone()
    perm_spec = sdunet_permutation_spec()

    for it in range(iterations):
        log_vram(f"Rebasin iteration {it}")
        weight_matcher.set_it(it)

        # normal block merge we already know and love
        thetas["model_a"] = simple_merge(
            thetas,
            weight_matcher,
            merge_mode,
            precision,
            False,
            device,
            work_device,
            threads,
        )

        log_vram("simple merge done")

        # find permutations
        perm_1, y = weight_matching(
            perm_spec,
            model_a,
            thetas["model_a"],
            max_iter=it,
            init_perm=None,
            usefp16=precision == 16,
            device=device,
        )

        log_vram("weight matching #1 done")

        thetas["model_a"] = apply_permutation(perm_spec, perm_1, thetas["model_a"])

        log_vram("apply perm 1 done")

        perm_2, z = weight_matching(
            perm_spec,
            thetas["model_b"],
            thetas["model_a"],
            max_iter=it,
            init_perm=None,
            usefp16=precision == 16,
            device=device,
        )

        log_vram("weight matching #2 done")

        new_alpha = torch.nn.functional.normalize(
            torch.sigmoid(torch.Tensor([y, z])), p=1, dim=0
        ).tolist()[0]
        thetas["model_a"] = update_model_a(
            perm_spec, perm_2, thetas["model_a"], new_alpha
        )

        log_vram("model a updated")

    if weights_clip:
        clip_thetas = thetas.copy()
        clip_thetas["model_a"] = model_a
        thetas["model_a"] = clip_weights(thetas, thetas["model_a"])

    return thetas["model_a"]


def simple_merge_key(progress, key, thetas, *args, **kwargs):
    with merge_key_context(key, thetas, *args, **kwargs) as result:
        if result is not None:
            thetas["model_a"].update({key: result.detach().clone()})

        progress.update()


def merge_key(
    key: str,
    thetas: Dict,
    weight_matcher: WeightClass,
    merge_mode: str,
    precision: int = 16,
    weights_clip: bool = False,
    device: str = "cpu",
    work_device: Optional[str] = None,
) -> Optional[Tuple[str, Dict]]:
    if work_device is None:
        work_device = device

    if KEY_POSITION_IDS in key:
        return

    for theta in thetas.values():
        if key not in theta.keys():
            return

        current_bases = weight_matcher(key)
        try:
            merge_method = getattr(merge_methods, merge_mode)
        except AttributeError as e:
            raise ValueError(f"{merge_mode} not implemented, aborting merge!") from e

        merge_args = get_merge_method_args(current_bases, thetas, key, work_device)

        # dealing with pix2pix and inpainting models
        if (a_size := merge_args["a"].size()) != (b_size := merge_args["b"].size()):
            if a_size[1] > b_size[1]:
                merged_key = merge_args["a"]
            else:
                merged_key = merge_args["b"]
        else:
            merged_key = merge_method(**merge_args).to(device)

        if weights_clip:
            merged_key = clip_weights_key(thetas, merged_key, key)

        if precision == 16:
            merged_key = merged_key.half()

        return merged_key


def clip_weights(thetas, merged):
    for k in thetas["model_a"].keys():
        if k in thetas["model_b"].keys():
            merged.update({k: clip_weights_key(thetas, merged[k], k)})
    return merged


def clip_weights_key(thetas, merged_weights, key):
    t0 = thetas["model_a"][key]
    t1 = thetas["model_b"][key]
    maximums = torch.maximum(t0, t1)
    minimums = torch.minimum(t0, t1)
    return torch.minimum(torch.maximum(merged_weights, minimums), maximums)


@contextmanager
def merge_key_context(*args, **kwargs):
    result = merge_key(*args, **kwargs)
    try:
        yield result
    finally:
        if result is not None:
            del result


def get_merge_method_args(
    current_bases: Dict,
    thetas: Dict,
    key: str,
    work_device: str,
) -> Dict:
    merge_method_args = {
        "a": thetas["model_a"][key].to(work_device),
        "b": thetas["model_b"][key].to(work_device),
        **current_bases,
    }

    if "model_c" in thetas:
        merge_method_args["c"] = thetas["model_c"][key].to(work_device)

    return merge_method_args


def save_model(model, output_file, file_format) -> None:
    log.info(f"Saving {output_file}")
    if file_format == "safetensors":
        safetensors.torch.save_file(
            model if type(model) == dict else model.to_dict(),
            f"{output_file}.safetensors",
            metadata={"format": "pt"},
        )
    else:
        torch.save({"state_dict": model}, f"{output_file}.ckpt")
