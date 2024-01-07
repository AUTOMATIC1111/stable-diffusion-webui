from typing import Union
import time
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from modules.control.proc.reference_sd15 import StableDiffusionReferencePipeline
from modules.control.proc.reference_sdxl import StableDiffusionXLReferencePipeline
from modules.shared import log, opts
from modules.control.units import detect


what = 'Reference'


def list_models():
    return ['Reference']


class ReferencePipeline():
    def __init__(self, pipeline: Union[StableDiffusionXLPipeline, StableDiffusionPipeline], dtype = None):
        t0 = time.time()
        self.orig_pipeline = pipeline
        self.pipeline = None
        if pipeline is None:
            log.error(f'Control {what} model pipeline: model not loaded')
            return
        if opts.diffusers_fuse_projections and hasattr(pipeline, 'unfuse_qkv_projections'):
            pipeline.unfuse_qkv_projections()
        if detect.is_sdxl(pipeline):
            self.pipeline = StableDiffusionXLReferencePipeline(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                text_encoder_2=pipeline.text_encoder_2,
                tokenizer=pipeline.tokenizer,
                tokenizer_2=pipeline.tokenizer_2,
                unet=pipeline.unet,
                scheduler=pipeline.scheduler,
                feature_extractor=getattr(pipeline, 'feature_extractor', None),
            ).to(pipeline.device)
        elif detect.is_sd15(pipeline):
            self.pipeline = StableDiffusionReferencePipeline(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                tokenizer=pipeline.tokenizer,
                unet=pipeline.unet,
                scheduler=pipeline.scheduler,
                feature_extractor=getattr(pipeline, 'feature_extractor', None),
                requires_safety_checker=False,
                safety_checker=None,
            ).to(pipeline.device)
        else:
            log.error(f'Control {what} pipeline: class={pipeline.__class__.__name__} unsupported model type')
            return
        if dtype is not None and self.pipeline is not None:
            self.pipeline = self.pipeline.to(dtype)
        t1 = time.time()
        if self.pipeline is not None:
            log.debug(f'Control {what} pipeline: class={self.pipeline.__class__.__name__} time={t1-t0:.2f}')
        else:
            log.error(f'Control {what} pipeline: not initialized')

    def restore(self):
        self.pipeline = None
        return self.orig_pipeline
