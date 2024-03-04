from ..patch_match import PyramidPatchMatcher
import os
import numpy as np
from PIL import Image
from tqdm import tqdm


class InterpolationModeRunner:
    def __init__(self):
        pass

    def get_index_dict(self, index_style):
        index_dict = {}
        for i, index in enumerate(index_style):
            index_dict[index] = i
        return index_dict

    def get_weight(self, l, m, r):
        weight_l, weight_r = abs(m - r), abs(m - l)
        if weight_l + weight_r == 0:
            weight_l, weight_r = 0.5, 0.5
        else:
            weight_l, weight_r = weight_l / (weight_l + weight_r), weight_r / (weight_l + weight_r)
        return weight_l, weight_r

    def get_task_group(self, index_style, n):
        task_group = []
        index_style = sorted(index_style)
        # first frame
        if index_style[0]>0:
            tasks = []
            for m in range(index_style[0]):
                tasks.append((index_style[0], m, index_style[0]))
            task_group.append(tasks)
        # middle frames
        for l, r in zip(index_style[:-1], index_style[1:]):
            tasks = []
            for m in range(l, r):
                tasks.append((l, m, r))
            task_group.append(tasks)
        # last frame
        tasks = []
        for m in range(index_style[-1], n):
            tasks.append((index_style[-1], m, index_style[-1]))
        task_group.append(tasks)
        return task_group

    def run(self, frames_guide, frames_style, index_style, batch_size, ebsynth_config, save_path=None):
        patch_match_engine = PyramidPatchMatcher(
            image_height=frames_style[0].shape[0],
            image_width=frames_style[0].shape[1],
            channel=3,
            use_mean_target_style=False,
            use_pairwise_patch_error=True,
            **ebsynth_config
        )
        # task
        index_dict = self.get_index_dict(index_style)
        task_group = self.get_task_group(index_style, len(frames_guide))
        # run
        for tasks in task_group:
            index_start, index_end = min([i[1] for i in tasks]), max([i[1] for i in tasks])
            for batch_id in tqdm(range(0, len(tasks), batch_size), desc=f"Rendering frames {index_start}...{index_end}"):
                tasks_batch = tasks[batch_id: min(batch_id+batch_size, len(tasks))]
                source_guide, target_guide, source_style = [], [], []
                for l, m, r in tasks_batch:
                    # l -> m
                    source_guide.append(frames_guide[l])
                    target_guide.append(frames_guide[m])
                    source_style.append(frames_style[index_dict[l]])
                    # r -> m
                    source_guide.append(frames_guide[r])
                    target_guide.append(frames_guide[m])
                    source_style.append(frames_style[index_dict[r]])
                source_guide = np.stack(source_guide)
                target_guide = np.stack(target_guide)
                source_style = np.stack(source_style)
                _, target_style = patch_match_engine.estimate_nnf(source_guide, target_guide, source_style)
                if save_path is not None:
                    for frame_l, frame_r, (l, m, r) in zip(target_style[0::2], target_style[1::2], tasks_batch):
                        weight_l, weight_r = self.get_weight(l, m, r)
                        frame = frame_l * weight_l + frame_r * weight_r
                        frame = frame.clip(0, 255).astype("uint8")
                        Image.fromarray(frame).save(os.path.join(save_path, "%05d.png" % m))


class InterpolationModeSingleFrameRunner:
    def __init__(self):
        pass

    def run(self, frames_guide, frames_style, index_style, batch_size, ebsynth_config, save_path=None):
        # check input
        tracking_window_size = ebsynth_config["tracking_window_size"]
        if tracking_window_size * 2 >= batch_size:
            raise ValueError("batch_size should be larger than track_window_size * 2")
        frame_style = frames_style[0]
        frame_guide = frames_guide[index_style[0]]
        patch_match_engine = PyramidPatchMatcher(
            image_height=frame_style.shape[0],
            image_width=frame_style.shape[1],
            channel=3,
            **ebsynth_config
        )
        # run
        frame_id, n = 0, len(frames_guide)
        for i in tqdm(range(0, n, batch_size - tracking_window_size * 2), desc=f"Rendering frames 0...{n}"):
            if i + batch_size > n:
                l, r = max(n - batch_size, 0), n
            else:
                l, r = i, i + batch_size
            source_guide = np.stack([frame_guide] * (r-l))
            target_guide = np.stack([frames_guide[i] for i in range(l, r)])
            source_style = np.stack([frame_style] * (r-l))
            _, target_style = patch_match_engine.estimate_nnf(source_guide, target_guide, source_style)
            for i, frame in zip(range(l, r), target_style):
                if i==frame_id:
                    frame = frame.clip(0, 255).astype("uint8")
                    Image.fromarray(frame).save(os.path.join(save_path, "%05d.png" % frame_id))
                    frame_id += 1
                if r < n and r-frame_id <= tracking_window_size:
                    break
