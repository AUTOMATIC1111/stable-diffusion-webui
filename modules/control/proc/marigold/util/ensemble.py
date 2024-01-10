# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import numpy as np
import torch

from scipy.optimize import minimize


def inter_distances(tensors: torch.Tensor):
    """
    To calculate the distance between each two depth maps.
    """
    distances = []
    for i, j in torch.combinations(torch.arange(tensors.shape[0])):
        arr1 = tensors[i : i + 1]
        arr2 = tensors[j : j + 1]
        distances.append(arr1 - arr2)
    dist = torch.concatenate(distances, dim=0)
    return dist


def ensemble_depths(
    input_images: torch.Tensor,
    regularizer_strength: float = 0.02,
    max_iter: int = 2,
    tol: float = 1e-3,
    reduction: str = "median",
    max_res: int = None,
):
    """
    To ensemble multiple affine-invariant depth images (up to scale and shift),
        by aligning estimating the scale and shift
    """
    device = input_images.device
    dtype = input_images.dtype
    np_dtype = np.float32

    original_input = input_images.clone()
    n_img = input_images.shape[0]
    ori_shape = input_images.shape

    if max_res is not None:
        scale_factor = torch.min(max_res / torch.tensor(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            input_images = downscaler(torch.from_numpy(input_images)).numpy()

    # init guess
    _min = np.min(input_images.reshape((n_img, -1)).cpu().numpy(), axis=1)
    _max = np.max(input_images.reshape((n_img, -1)).cpu().numpy(), axis=1)
    s_init = 1.0 / (_max - _min).reshape((-1, 1, 1))
    t_init = (-1 * s_init.flatten() * _min.flatten()).reshape((-1, 1, 1))
    x = np.concatenate([s_init, t_init]).reshape(-1).astype(np_dtype)

    input_images = input_images.to(device)

    # objective function
    def closure(x):
        l = len(x) # noqa
        s = x[: int(l / 2)]
        t = x[int(l / 2) :]
        s = torch.from_numpy(s).to(dtype=dtype).to(device)
        t = torch.from_numpy(t).to(dtype=dtype).to(device)

        transformed_arrays = input_images * s.view((-1, 1, 1)) + t.view((-1, 1, 1))
        dists = inter_distances(transformed_arrays)
        sqrt_dist = torch.sqrt(torch.mean(dists**2))

        if "mean" == reduction:
            pred = torch.mean(transformed_arrays, dim=0)
        elif "median" == reduction:
            pred = torch.median(transformed_arrays, dim=0).values
        else:
            raise ValueError

        near_err = torch.sqrt((0 - torch.min(pred)) ** 2)
        far_err = torch.sqrt((1 - torch.max(pred)) ** 2)

        err = sqrt_dist + (near_err + far_err) * regularizer_strength
        err = err.detach().cpu().numpy().astype(np_dtype)
        return err

    res = minimize(
        closure, x, method="BFGS", tol=tol, options={"maxiter": max_iter, "disp": False}
    )
    x = res.x
    l = len(x) # noqa
    s = x[: int(l / 2)]
    t = x[int(l / 2) :]

    # Prediction
    s = torch.from_numpy(s).to(dtype=dtype).to(device)
    t = torch.from_numpy(t).to(dtype=dtype).to(device)
    transformed_arrays = original_input * s.view(-1, 1, 1) + t.view(-1, 1, 1)
    if "mean" == reduction:
        aligned_images = torch.mean(transformed_arrays, dim=0)
        std = torch.std(transformed_arrays, dim=0)
        uncertainty = std
    elif "median" == reduction:
        aligned_images = torch.median(transformed_arrays, dim=0).values
        # MAD (median absolute deviation) as uncertainty indicator
        abs_dev = torch.abs(transformed_arrays - aligned_images)
        mad = torch.median(abs_dev, dim=0).values
        uncertainty = mad
    else:
        raise ValueError(f"Unknown reduction method: {reduction}")

    # Scale and shift to [0, 1]
    _min = torch.min(aligned_images)
    _max = torch.max(aligned_images)
    aligned_images = (aligned_images - _min) / (_max - _min)
    uncertainty /= _max - _min

    return aligned_images, uncertainty
