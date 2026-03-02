"""
Patched forward_timestep_embed function to support the following:
@source https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/ldm/modules/diffusionmodules/openaimodel.py
"""
from ldm.modules.attention import SpatialTransformer
try:
    from ldm.modules.attention import SpatialVideoTransformer
except (ImportError, ModuleNotFoundError):
    SpatialVideoTransformer = None
from ldm.modules.diffusionmodules.openaimodel import TimestepBlock, TimestepEmbedSequential, Upsample
try:
    from ldm.modules.diffusionmodules.openaimodel import VideoResBlock
except (ImportError, ModuleNotFoundError):
    VideoResBlock = None

# SD XL modules from generative-models repo
from sgm.modules.attention import SpatialTransformer as SpatialTransformerSGM
try:
    from sgm.modules.attention import SpatialVideoTransformer as SpatialVideoTransformerSGM
except (ImportError, ModuleNotFoundError):
    SpatialVideoTransformerSGM = None
from sgm.modules.diffusionmodules.openaimodel import TimestepBlock as TimestepBlockSGM, Upsample as UpsampleSGM
try:
    from sgm.modules.diffusionmodules.openaimodel import VideoResBlock as VideoResBlockSGM
except (ImportError, ModuleNotFoundError):
    VideoResBlockSGM = None

import torch.nn.functional as F

def forward_timestep_embed(ts:TimestepEmbedSequential, x, emb, context=None, output_shape=None, time_context=None, num_video_frames=None, image_only_indicator=None):
    for layer in ts:
        if VideoResBlock and isinstance(layer, (VideoResBlock, VideoResBlockSGM)):
            x = layer(x, emb, num_video_frames, image_only_indicator)
        elif isinstance(layer, (TimestepBlock, TimestepBlockSGM)):
            x = layer(x, emb)
        elif SpatialVideoTransformer and isinstance(layer, (SpatialVideoTransformer, SpatialVideoTransformerSGM)):
            x = layer(x, context, time_context, num_video_frames, image_only_indicator)
        elif isinstance(layer, (SpatialTransformer, SpatialTransformerSGM)):
            x = layer(x, context)
        elif isinstance(layer, (Upsample, UpsampleSGM)):
            x = forward_upsample(layer, x, output_shape=output_shape)
        else:
            x = layer(x)
    return x

def forward_upsample(self:Upsample, x, output_shape=None):
    assert x.shape[1] == self.channels
    if self.dims == 3:
        shape = [x.shape[2], x.shape[3] * 2, x.shape[4] * 2]
        if output_shape is not None:
            shape[1] = output_shape[3]
            shape[2] = output_shape[4]
    else:
        shape = [x.shape[2] * 2, x.shape[3] * 2]
        if output_shape is not None:
            shape[0] = output_shape[2]
            shape[1] = output_shape[3]

    x = F.interpolate(x, size=shape, mode="nearest")
    if self.use_conv:
        x = self.conv(x)
    return x
