# Modified from https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend  # noqa

from os import path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.onnx.operators import shape_as_tensor


def bilinear_grid_sample(im, grid, align_corners=False):
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid. Supported only bilinear interpolation
    method to sample the input pixels.

    Args:
        im (torch.Tensor): Input feature map, shape (N, C, H, W)
        grid (torch.Tensor): Point coordinates, shape (N, Hg, Wg, 2)
        align_corners {bool}: If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.
    Returns:
        torch.Tensor: A tensor with sampled points, shape (N, C, Hg, Wg)
    """
    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.view(n, -1)
    y = y.view(n, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding
    im_padded = F.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)
    padded_h = h + 2
    padded_w = w + 2
    # save points positions after padding
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size
    x0 = torch.where(x0 < 0, torch.tensor(0), x0)
    x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1), x0)
    x1 = torch.where(x1 < 0, torch.tensor(0), x1)
    x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1), x1)
    y0 = torch.where(y0 < 0, torch.tensor(0), y0)
    y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1), y0)
    y1 = torch.where(y1 < 0, torch.tensor(0), y1)
    y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1), y1)

    im_padded = im_padded.view(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    Ia = torch.gather(im_padded, 2, x0_y0)
    Ib = torch.gather(im_padded, 2, x0_y1)
    Ic = torch.gather(im_padded, 2, x1_y0)
    Id = torch.gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)


def is_in_onnx_export_without_custom_ops():
    from annotator.mmpkg.mmcv.ops import get_onnxruntime_op_path
    ort_custom_op_path = get_onnxruntime_op_path()
    return torch.onnx.is_in_onnx_export(
    ) and not osp.exists(ort_custom_op_path)


def normalize(grid):
    """Normalize input grid from [-1, 1] to [0, 1]
    Args:
        grid (Tensor): The grid to be normalize, range [-1, 1].
    Returns:
        Tensor: Normalized grid, range [0, 1].
    """

    return (grid + 1.0) / 2.0


def denormalize(grid):
    """Denormalize input grid from range [0, 1] to [-1, 1]
    Args:
        grid (Tensor): The grid to be denormalize, range [0, 1].
    Returns:
        Tensor: Denormalized grid, range [-1, 1].
    """

    return grid * 2.0 - 1.0


def generate_grid(num_grid, size, device):
    """Generate regular square grid of points in [0, 1] x [0, 1] coordinate
    space.

    Args:
        num_grid (int): The number of grids to sample, one for each region.
        size (tuple(int, int)): The side size of the regular grid.
        device (torch.device): Desired device of returned tensor.

    Returns:
        (torch.Tensor): A tensor of shape (num_grid, size[0]*size[1], 2) that
            contains coordinates for the regular grids.
    """

    affine_trans = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]], device=device)
    grid = F.affine_grid(
        affine_trans, torch.Size((1, 1, *size)), align_corners=False)
    grid = normalize(grid)
    return grid.view(1, -1, 2).expand(num_grid, -1, -1)


def rel_roi_point_to_abs_img_point(rois, rel_roi_points):
    """Convert roi based relative point coordinates to image based absolute
    point coordinates.

    Args:
        rois (Tensor): RoIs or BBoxes, shape (N, 4) or (N, 5)
        rel_roi_points (Tensor): Point coordinates inside RoI, relative to
            RoI, location, range (0, 1), shape (N, P, 2)
    Returns:
        Tensor: Image based absolute point coordinates, shape (N, P, 2)
    """

    with torch.no_grad():
        assert rel_roi_points.size(0) == rois.size(0)
        assert rois.dim() == 2
        assert rel_roi_points.dim() == 3
        assert rel_roi_points.size(2) == 2
        # remove batch idx
        if rois.size(1) == 5:
            rois = rois[:, 1:]
        abs_img_points = rel_roi_points.clone()
        # To avoid an error during exporting to onnx use independent
        # variables instead inplace computation
        xs = abs_img_points[:, :, 0] * (rois[:, None, 2] - rois[:, None, 0])
        ys = abs_img_points[:, :, 1] * (rois[:, None, 3] - rois[:, None, 1])
        xs += rois[:, None, 0]
        ys += rois[:, None, 1]
        abs_img_points = torch.stack([xs, ys], dim=2)
    return abs_img_points


def get_shape_from_feature_map(x):
    """Get spatial resolution of input feature map considering exporting to
    onnx mode.

    Args:
        x (torch.Tensor): Input tensor, shape (N, C, H, W)
    Returns:
        torch.Tensor: Spatial resolution (width, height), shape (1, 1, 2)
    """
    if torch.onnx.is_in_onnx_export():
        img_shape = shape_as_tensor(x)[2:].flip(0).view(1, 1, 2).to(
            x.device).float()
    else:
        img_shape = torch.tensor(x.shape[2:]).flip(0).view(1, 1, 2).to(
            x.device).float()
    return img_shape


def abs_img_point_to_rel_img_point(abs_img_points, img, spatial_scale=1.):
    """Convert image based absolute point coordinates to image based relative
    coordinates for sampling.

    Args:
        abs_img_points (Tensor): Image based absolute point coordinates,
            shape (N, P, 2)
        img (tuple/Tensor): (height, width) of image or feature map.
        spatial_scale (float): Scale points by this factor. Default: 1.

    Returns:
        Tensor: Image based relative point coordinates for sampling,
            shape (N, P, 2)
    """

    assert (isinstance(img, tuple) and len(img) == 2) or \
           (isinstance(img, torch.Tensor) and len(img.shape) == 4)

    if isinstance(img, tuple):
        h, w = img
        scale = torch.tensor([w, h],
                             dtype=torch.float,
                             device=abs_img_points.device)
        scale = scale.view(1, 1, 2)
    else:
        scale = get_shape_from_feature_map(img)

    return abs_img_points / scale * spatial_scale


def rel_roi_point_to_rel_img_point(rois,
                                   rel_roi_points,
                                   img,
                                   spatial_scale=1.):
    """Convert roi based relative point coordinates to image based absolute
    point coordinates.

    Args:
        rois (Tensor): RoIs or BBoxes, shape (N, 4) or (N, 5)
        rel_roi_points (Tensor): Point coordinates inside RoI, relative to
            RoI, location, range (0, 1), shape (N, P, 2)
        img (tuple/Tensor): (height, width) of image or feature map.
        spatial_scale (float): Scale points by this factor. Default: 1.

    Returns:
        Tensor: Image based relative point coordinates for sampling,
            shape (N, P, 2)
    """

    abs_img_point = rel_roi_point_to_abs_img_point(rois, rel_roi_points)
    rel_img_point = abs_img_point_to_rel_img_point(abs_img_point, img,
                                                   spatial_scale)

    return rel_img_point


def point_sample(input, points, align_corners=False, **kwargs):
    """A wrapper around :func:`grid_sample` to support 3D point_coords tensors
    Unlike :func:`torch.nn.functional.grid_sample` it assumes point_coords to
    lie inside ``[0, 1] x [0, 1]`` square.

    Args:
        input (Tensor): Feature map, shape (N, C, H, W).
        points (Tensor): Image based absolute point coordinates (normalized),
            range [0, 1] x [0, 1], shape (N, P, 2) or (N, Hgrid, Wgrid, 2).
        align_corners (bool): Whether align_corners. Default: False

    Returns:
        Tensor: Features of `point` on `input`, shape (N, C, P) or
            (N, C, Hgrid, Wgrid).
    """

    add_dim = False
    if points.dim() == 3:
        add_dim = True
        points = points.unsqueeze(2)
    if is_in_onnx_export_without_custom_ops():
        # If custom ops for onnx runtime not compiled use python
        # implementation of grid_sample function to make onnx graph
        # with supported nodes
        output = bilinear_grid_sample(
            input, denormalize(points), align_corners=align_corners)
    else:
        output = F.grid_sample(
            input, denormalize(points), align_corners=align_corners, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


class SimpleRoIAlign(nn.Module):

    def __init__(self, output_size, spatial_scale, aligned=True):
        """Simple RoI align in PointRend, faster than standard RoIAlign.

        Args:
            output_size (tuple[int]): h, w
            spatial_scale (float): scale the input boxes by this number
            aligned (bool): if False, use the legacy implementation in
                MMDetection, align_corners=True will be used in F.grid_sample.
                If True, align the results more perfectly.
        """

        super(SimpleRoIAlign, self).__init__()
        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        # to be consistent with other RoI ops
        self.use_torchvision = False
        self.aligned = aligned

    def forward(self, features, rois):
        num_imgs = features.size(0)
        num_rois = rois.size(0)
        rel_roi_points = generate_grid(
            num_rois, self.output_size, device=rois.device)

        if torch.onnx.is_in_onnx_export():
            rel_img_points = rel_roi_point_to_rel_img_point(
                rois, rel_roi_points, features, self.spatial_scale)
            rel_img_points = rel_img_points.reshape(num_imgs, -1,
                                                    *rel_img_points.shape[1:])
            point_feats = point_sample(
                features, rel_img_points, align_corners=not self.aligned)
            point_feats = point_feats.transpose(1, 2)
        else:
            point_feats = []
            for batch_ind in range(num_imgs):
                # unravel batch dim
                feat = features[batch_ind].unsqueeze(0)
                inds = (rois[:, 0].long() == batch_ind)
                if inds.any():
                    rel_img_points = rel_roi_point_to_rel_img_point(
                        rois[inds], rel_roi_points[inds], feat,
                        self.spatial_scale).unsqueeze(0)
                    point_feat = point_sample(
                        feat, rel_img_points, align_corners=not self.aligned)
                    point_feat = point_feat.squeeze(0).transpose(0, 1)
                    point_feats.append(point_feat)

            point_feats = torch.cat(point_feats, dim=0)

        channels = features.size(1)
        roi_feats = point_feats.reshape(num_rois, channels, *self.output_size)

        return roi_feats

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += '(output_size={}, spatial_scale={}'.format(
            self.output_size, self.spatial_scale)
        return format_str
