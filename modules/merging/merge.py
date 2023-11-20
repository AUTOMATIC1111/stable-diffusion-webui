import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Dict, Optional, Tuple
import safetensors.torch
import torch
from tensordict import TensorDict
import modules.memstats
import modules.devices as devices
from modules.shared import log, console
from modules.sd_models import read_state_dict
from modules.merging import merge_methods
from modules.merging.merge_utils import WeightClass
from modules.merging.merge_rebasin import (
    apply_permutation,
    sdunet_permutation_spec,
    update_model_a,
    weight_matching,
)
##########################################################
# Files in modules.merging are heavily modified
# versions of sd-meh by @s1dxl used with his blessing
# orginal code can be found @ https://github.com/s1dlx/meh
##########################################################

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


def fix_clip(model: Dict) -> Dict:
    if KEY_POSITION_IDS in model.keys():
        model[KEY_POSITION_IDS] = torch.tensor(
            [list(range(MAX_TOKENS))],
            dtype=torch.int64,
            device=model[KEY_POSITION_IDS].device,
        )

    return model


def prune_sd_model(model: Dict) -> Dict:
    keys = list(model.keys())
    for k in keys:
        if (
            not k.startswith("model.diffusion_model.")
            # and not k.startswith("first_stage_model.")
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
    log.debug(f"Merge {txt}: {modules.memstats.memory_stats()}")


def load_thetas(
    models: Dict[str, os.PathLike | str],
    prune: bool,
    device: torch.device,
    precision: str,
) -> Dict:
    if prune:
        thetas = {k: prune_sd_model(TensorDict.from_dict(read_state_dict(m, "cpu"))) for k, m in models.items()}
    else:
        thetas = {k: TensorDict.from_dict(read_state_dict(m, device)) for k, m in models.items()}

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
    precision: str = "fp16",
    weights_clip: bool = False,
    re_basin: bool = False,
    device: torch.device = None,
    work_device: torch.device = None,
    prune: bool = False,
    threads: int = 4,
    **kwargs,
) -> Dict:
    thetas = load_thetas(models, prune, device, precision)
    # log.info(f'Merge start: models={models.values()} precision={precision} clip={weights_clip} rebasin={re_basin} prune={prune} threads={threads}')
    weight_matcher = WeightClass(thetas["model_a"], **kwargs)
    if re_basin:
        merged = rebasin_merge(
            thetas,
            weight_matcher,
            merge_mode,
            precision=precision,
            weights_clip=weights_clip,
            iterations=kwargs.get("re_basin_iterations", 1),
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
    device: torch.device,
    prune: bool,
    precision: str,
) -> Dict:
    if prune:
        log.info("Merge restoring pruned keys")
        del thetas
        devices.torch_gc(force=False)
        original_a = TensorDict.from_dict(read_state_dict(models["model_a"], device))
        unpruned = 0
        for key in original_a.keys():
            if KEY_POSITION_IDS in key:
                continue
            if "model" in key and key not in merged.keys():
                merged.update({key: original_a[key]})
                unpruned += 1
                if precision == "fp16":
                    merged.update({key: merged[key].half()})
        if unpruned > 248:  # VAE has 248 keys, and we are purposely restoring it here
            log.debug(f"Merge restored from primary model: keys={unpruned - 248}")
        unpruned = 0
        del original_a
        original_b = TensorDict.from_dict(read_state_dict(models["model_b"], device))
        for key in original_b.keys():
            if KEY_POSITION_IDS in key:
                continue
            if "model" in key and key not in merged.keys():
                merged.update({key: original_b[key]})
                unpruned += 1
                if precision == "fp16":
                    merged.update({key: merged[key].half()})
        if unpruned != 0:
            log.debug(f"Merge restored from secondary model: keys={unpruned}")
        del original_b
        devices.torch_gc(force=False)

    return fix_clip(merged)


def simple_merge(
    thetas: Dict[str, Dict],
    weight_matcher: WeightClass,
    merge_mode: str,
    precision: str = "fp16",
    weights_clip: bool = False,
    device: torch.device = None,
    work_device: torch.device = None,
    threads: int = 4,
) -> Dict:
    futures = []
    # with tqdm(thetas["model_a"].keys(), desc="Merge") as progress:
    import rich.progress as p
    with p.Progress(p.TextColumn('[cyan]{task.description}'), p.BarColumn(), p.TaskProgressColumn(), p.TimeRemainingColumn(), p.TimeElapsedColumn(), p.TextColumn('[cyan]keys={task.fields[keys]}'), console=console) as progress:
        task = progress.add_task(description="Merging", total=len(thetas["model_a"].keys()), keys=len(thetas["model_a"].keys()))
        with ThreadPoolExecutor(max_workers=threads) as executor:
            for key in thetas["model_a"].keys():
                future = executor.submit(
                    simple_merge_key,
                    progress,
                    task,
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

    if len(thetas["model_b"]) > 0:
        log.debug(f'Merge update thetas: keys={len(thetas["model_b"])}')
        for key in thetas["model_b"].keys():
            if KEY_POSITION_IDS in key:
                continue
            if "model" in key and key not in thetas["model_a"].keys():
                thetas["model_a"].update({key: thetas["model_b"][key]})
                if precision == "fp16":
                    thetas["model_a"].update({key: thetas["model_a"][key].half()})

    return fix_clip(thetas["model_a"])


def rebasin_merge(
    thetas: Dict[str, os.PathLike | str],
    weight_matcher: WeightClass,
    merge_mode: str,
    precision: str = "fp16",
    weights_clip: bool = False,
    iterations: int = 1,
    device: torch.device = None,
    work_device: torch.device = None,
    threads: int = 1,
):
    # not sure how this does when 3 models are involved...
    model_a = thetas["model_a"].clone()
    perm_spec = sdunet_permutation_spec()

    for it in range(iterations):
        log_vram(f"rebasin: iteration={it}")
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

        # find permutations
        perm_1, y = weight_matching(
            perm_spec,
            model_a,
            thetas["model_a"],
            max_iter=it,
            init_perm=None,
            usefp16=precision == "fp16",
            device=device,
        )
        thetas["model_a"] = apply_permutation(perm_spec, perm_1, thetas["model_a"])

        perm_2, z = weight_matching(
            perm_spec,
            thetas["model_b"],
            thetas["model_a"],
            max_iter=it,
            init_perm=None,
            usefp16=precision == "fp16",
            device=device,
        )

        new_alpha = torch.nn.functional.normalize(
            torch.sigmoid(torch.Tensor([y, z])), p=1, dim=0
        ).tolist()[0]
        thetas["model_a"] = update_model_a(
            perm_spec, perm_2, thetas["model_a"], new_alpha
        )

    if weights_clip:
        clip_thetas = thetas.copy()
        clip_thetas["model_a"] = model_a
        thetas["model_a"] = clip_weights(thetas, thetas["model_a"])

    return thetas["model_a"]


def simple_merge_key(progress, task, key, thetas, *args, **kwargs):
    with merge_key_context(key, thetas, *args, **kwargs) as result:
        if result is not None:
            thetas["model_a"].update({key: result.detach().clone()})
    progress.update(task, advance=1)


def merge_key(  # pylint: disable=inconsistent-return-statements
    key: str,
    thetas: Dict,
    weight_matcher: WeightClass,
    merge_mode: str,
    precision: str = "fp16",
    weights_clip: bool = False,
    device: torch.device = None,
    work_device: torch.device = None,
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

        if precision == "fp16":
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
    work_device: torch.device,
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
    log.info(f"Merge saving: model='{output_file}'")
    if file_format == "safetensors":
        safetensors.torch.save_file(
            model if type(model) == dict else model.to_dict(),
            f"{output_file}.safetensors",
            metadata={"format": "pt"},
        )
    else:
        torch.save({"state_dict": model}, f"{output_file}.ckpt")
