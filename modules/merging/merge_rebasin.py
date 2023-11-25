# https://github.com/ogkalu2/Merge-Stable-Diffusion-models-without-distortion
from collections import defaultdict
from random import shuffle
from typing import NamedTuple
import torch
from scipy.optimize import linear_sum_assignment
from modules.shared import log


SPECIAL_KEYS = [
    "first_stage_model.decoder.norm_out.weight",
    "first_stage_model.decoder.norm_out.bias",
    "first_stage_model.encoder.norm_out.weight",
    "first_stage_model.encoder.norm_out.bias",
    "model.diffusion_model.out.0.weight",
    "model.diffusion_model.out.0.bias",
]


class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):
        # Skip the axis we're trying to permute.
        if axis == except_axis:
            continue

        # None indicates that there is no permutation relevant to that axis.
        if p:
            w = torch.index_select(w, axis, perm[p].int())

    return w


def apply_permutation(ps: PermutationSpec, perm, params):
    """Apply a `perm` to `params`."""
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def update_model_a(ps: PermutationSpec, perm, model_a, new_alpha):
    for k in model_a:
        try:
            perm_params = get_permuted_param(
                ps, perm, k, model_a
            )
            model_a[k] = model_a[k] * (1 - new_alpha) + new_alpha * perm_params
        except RuntimeError:  # dealing with pix2pix and inpainting models
            continue
    return model_a


def inner_matching(
    n,
    ps,
    p,
    params_a,
    params_b,
    usefp16,
    progress,
    number,
    linear_sum,
    perm,
    device,
):
    A = torch.zeros((n, n), dtype=torch.float16) if usefp16 else torch.zeros((n, n))
    A = A.to(device)

    for wk, axis in ps.perm_to_axes[p]:
        w_a = params_a[wk]
        w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
        w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1)).to(device)
        w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1)).T.to(device)

        if usefp16:
            w_a = w_a.half().to(device)
            w_b = w_b.half().to(device)

        try:
            A += torch.matmul(w_a, w_b)
        except RuntimeError:
            A += torch.matmul(torch.dequantize(w_a), torch.dequantize(w_b))

    A = A.cpu()
    ri, ci = linear_sum_assignment(A.detach().numpy(), maximize=True)
    A = A.to(device)

    assert (torch.tensor(ri) == torch.arange(len(ri))).all()

    eye_tensor = torch.eye(n).to(device)

    oldL = torch.vdot(
        torch.flatten(A).float(), torch.flatten(eye_tensor[perm[p].long()])
    )
    newL = torch.vdot(torch.flatten(A).float(), torch.flatten(eye_tensor[ci, :]))

    if usefp16:
        oldL = oldL.half()
        newL = newL.half()

    if newL - oldL != 0:
        linear_sum += abs((newL - oldL).item())
        number += 1
        log.debug(f"Merge Rebasin permutation: {p}={newL-oldL}")

    progress = progress or newL > oldL + 1e-12

    perm[p] = torch.Tensor(ci).to(device)

    return linear_sum, number, perm, progress


def weight_matching(
    ps: PermutationSpec,
    params_a,
    params_b,
    max_iter=1,
    init_perm=None,
    usefp16=False,
    device="cpu",
):
    perm_sizes = {
        p: params_a[axes[0][0]].shape[axes[0][1]]
        for p, axes in ps.perm_to_axes.items()
        if axes[0][0] in params_a.keys()
    }
    perm = {}
    perm = (
        {p: torch.arange(n).to(device) for p, n in perm_sizes.items()}
        if init_perm is None
        else init_perm
    )

    linear_sum = 0
    number = 0

    special_layers = ["P_bg324"]
    for _i in range(max_iter):
        progress = False
        shuffle(special_layers)
        for p in special_layers:
            n = perm_sizes[p]
            linear_sum, number, perm, progress = inner_matching(
                n,
                ps,
                p,
                params_a,
                params_b,
                usefp16,
                progress,
                number,
                linear_sum,
                perm,
                device,
            )
        progress = True
        if not progress:
            break

    average = linear_sum / number if number > 0 else 0
    return perm, average
