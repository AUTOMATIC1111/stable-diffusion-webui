import torch
import numpy as np
from .processors import Processor_id


class ControlNetConfigUnit:
    def __init__(self, processor_id: Processor_id, model_path, scale=1.0):
        self.processor_id = processor_id
        self.model_path = model_path
        self.scale = scale


class ControlNetUnit:
    def __init__(self, processor, model, scale=1.0):
        self.processor = processor
        self.model = model
        self.scale = scale


class MultiControlNetManager:
    def __init__(self, controlnet_units=[]):
        self.processors = [unit.processor for unit in controlnet_units]
        self.models = [unit.model for unit in controlnet_units]
        self.scales = [unit.scale for unit in controlnet_units]

    def process_image(self, image, return_image=False):
        processed_image = [
            processor(image)
            for processor in self.processors
        ]
        if return_image:
            return processed_image
        processed_image = torch.concat([
            torch.Tensor(np.array(image_, dtype=np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
            for image_ in processed_image
        ], dim=0)
        return processed_image
    
    def __call__(
        self,
        sample, timestep, encoder_hidden_states, conditionings,
        tiled=False, tile_size=64, tile_stride=32
    ):
        res_stack = None
        for conditioning, model, scale in zip(conditionings, self.models, self.scales):
            res_stack_ = model(
                sample, timestep, encoder_hidden_states, conditioning,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
            )
            res_stack_ = [res * scale for res in res_stack_]
            if res_stack is None:
                res_stack = res_stack_
            else:
                res_stack = [i + j for i, j in zip(res_stack, res_stack_)]
        return res_stack
