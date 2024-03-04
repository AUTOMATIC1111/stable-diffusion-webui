from .runners.fast import TableManager, PyramidPatchMatcher
from PIL import Image
import numpy as np
import cupy as cp


class FastBlendSmoother:
    def __init__(self):
        self.batch_size = 8
        self.window_size = 32
        self.ebsynth_config = {
            "minimum_patch_size": 5,
            "threads_per_block": 8,
            "num_iter": 5,
            "gpu_id": 0,
            "guide_weight": 10.0,
            "initialize": "identity",
            "tracking_window_size": 0,
        }

    @staticmethod
    def from_model_manager(model_manager):
        # TODO: fetch GPU ID from model_manager
        return FastBlendSmoother()

    def run(self, frames_guide, frames_style, batch_size, window_size, ebsynth_config):
        frames_guide = [np.array(frame) for frame in frames_guide]
        frames_style = [np.array(frame) for frame in frames_style]
        table_manager = TableManager()
        patch_match_engine = PyramidPatchMatcher(
            image_height=frames_style[0].shape[0],
            image_width=frames_style[0].shape[1],
            channel=3,
            **ebsynth_config
        )
        # left part
        table_l = table_manager.build_remapping_table(frames_guide, frames_style, patch_match_engine, batch_size, desc="FastBlend Step 1/4")
        table_l = table_manager.remapping_table_to_blending_table(table_l)
        table_l = table_manager.process_window_sum(frames_guide, table_l, patch_match_engine, window_size, batch_size, desc="FastBlend Step 2/4")
        # right part
        table_r = table_manager.build_remapping_table(frames_guide[::-1], frames_style[::-1], patch_match_engine, batch_size, desc="FastBlend Step 3/4")
        table_r = table_manager.remapping_table_to_blending_table(table_r)
        table_r = table_manager.process_window_sum(frames_guide[::-1], table_r, patch_match_engine, window_size, batch_size, desc="FastBlend Step 4/4")[::-1]
        # merge
        frames = []
        for (frame_l, weight_l), frame_m, (frame_r, weight_r) in zip(table_l, frames_style, table_r):
            weight_m = -1
            weight = weight_l + weight_m + weight_r
            frame = frame_l * (weight_l / weight) + frame_m * (weight_m / weight) + frame_r * (weight_r / weight)
            frames.append(frame)
        frames = [Image.fromarray(frame.clip(0, 255).astype("uint8")) for frame in frames]
        return frames
    
    def __call__(self, rendered_frames, original_frames=None, **kwargs):
        frames = self.run(
            original_frames, rendered_frames,
            self.batch_size, self.window_size, self.ebsynth_config
        )
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return frames