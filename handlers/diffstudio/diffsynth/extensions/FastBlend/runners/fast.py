from ..patch_match import PyramidPatchMatcher
import functools, os
import numpy as np
from PIL import Image
from tqdm import tqdm


class TableManager:
    def __init__(self):
        pass

    def task_list(self, n):
        tasks = []
        max_level = 1
        while (1<<max_level)<=n:
            max_level += 1
        for i in range(n):
            j = i
            for level in range(max_level):
                if i&(1<<level):
                    continue
                j |= 1<<level
                if j>=n:
                    break
                meta_data = {
                    "source": i,
                    "target": j,
                    "level": level + 1
                }
                tasks.append(meta_data)
        tasks.sort(key=functools.cmp_to_key(lambda u, v: u["level"]-v["level"]))
        return tasks
    
    def build_remapping_table(self, frames_guide, frames_style, patch_match_engine, batch_size, desc=""):
        n = len(frames_guide)
        tasks = self.task_list(n)
        remapping_table = [[(frames_style[i], 1)] for i in range(n)]
        for batch_id in tqdm(range(0, len(tasks), batch_size), desc=desc):
            tasks_batch = tasks[batch_id: min(batch_id+batch_size, len(tasks))]
            source_guide = np.stack([frames_guide[task["source"]] for task in tasks_batch])
            target_guide = np.stack([frames_guide[task["target"]] for task in tasks_batch])
            source_style = np.stack([frames_style[task["source"]] for task in tasks_batch])
            _, target_style = patch_match_engine.estimate_nnf(source_guide, target_guide, source_style)
            for task, result in zip(tasks_batch, target_style):
                target, level = task["target"], task["level"]
                if len(remapping_table[target])==level:
                    remapping_table[target].append((result, 1))
                else:
                    frame, weight = remapping_table[target][level]
                    remapping_table[target][level] = (
                        frame * (weight / (weight + 1)) + result / (weight + 1),
                        weight + 1
                    )
        return remapping_table

    def remapping_table_to_blending_table(self, table):
        for i in range(len(table)):
            for j in range(1, len(table[i])):
                frame_1, weight_1 = table[i][j-1]
                frame_2, weight_2 = table[i][j]
                frame = (frame_1 + frame_2) / 2
                weight = weight_1 + weight_2
                table[i][j] = (frame, weight)
        return table

    def tree_query(self, leftbound, rightbound):
        node_list = []
        node_index = rightbound
        while node_index>=leftbound:
            node_level = 0
            while (1<<node_level)&node_index and node_index-(1<<node_level+1)+1>=leftbound:
                node_level += 1
            node_list.append((node_index, node_level))
            node_index -= 1<<node_level
        return node_list

    def process_window_sum(self, frames_guide, blending_table, patch_match_engine, window_size, batch_size, desc=""):
        n = len(blending_table)
        tasks = []
        frames_result = []
        for target in range(n):
            node_list = self.tree_query(max(target-window_size, 0), target)
            for source, level in node_list:
                if source!=target:
                    meta_data = {
                        "source": source,
                        "target": target,
                        "level": level
                    }
                    tasks.append(meta_data)
                else:
                    frames_result.append(blending_table[target][level])
        for batch_id in tqdm(range(0, len(tasks), batch_size), desc=desc):
            tasks_batch = tasks[batch_id: min(batch_id+batch_size, len(tasks))]
            source_guide = np.stack([frames_guide[task["source"]] for task in tasks_batch])
            target_guide = np.stack([frames_guide[task["target"]] for task in tasks_batch])
            source_style = np.stack([blending_table[task["source"]][task["level"]][0] for task in tasks_batch])
            _, target_style = patch_match_engine.estimate_nnf(source_guide, target_guide, source_style)
            for task, frame_2 in zip(tasks_batch, target_style):
                source, target, level = task["source"], task["target"], task["level"]
                frame_1, weight_1 = frames_result[target]
                weight_2 = blending_table[source][level][1]
                weight = weight_1 + weight_2
                frame = frame_1 * (weight_1 / weight) + frame_2 * (weight_2 / weight)
                frames_result[target] = (frame, weight)
        return frames_result


class FastModeRunner:
    def __init__(self):
        pass

    def run(self, frames_guide, frames_style, batch_size, window_size, ebsynth_config, save_path=None):
        frames_guide = frames_guide.raw_data()
        frames_style = frames_style.raw_data()
        table_manager = TableManager()
        patch_match_engine = PyramidPatchMatcher(
            image_height=frames_style[0].shape[0],
            image_width=frames_style[0].shape[1],
            channel=3,
            **ebsynth_config
        )
        # left part
        table_l = table_manager.build_remapping_table(frames_guide, frames_style, patch_match_engine, batch_size, desc="Fast Mode Step 1/4")
        table_l = table_manager.remapping_table_to_blending_table(table_l)
        table_l = table_manager.process_window_sum(frames_guide, table_l, patch_match_engine, window_size, batch_size, desc="Fast Mode Step 2/4")
        # right part
        table_r = table_manager.build_remapping_table(frames_guide[::-1], frames_style[::-1], patch_match_engine, batch_size, desc="Fast Mode Step 3/4")
        table_r = table_manager.remapping_table_to_blending_table(table_r)
        table_r = table_manager.process_window_sum(frames_guide[::-1], table_r, patch_match_engine, window_size, batch_size, desc="Fast Mode Step 4/4")[::-1]
        # merge
        frames = []
        for (frame_l, weight_l), frame_m, (frame_r, weight_r) in zip(table_l, frames_style, table_r):
            weight_m = -1
            weight = weight_l + weight_m + weight_r
            frame = frame_l * (weight_l / weight) + frame_m * (weight_m / weight) + frame_r * (weight_r / weight)
            frames.append(frame)
        frames = [frame.clip(0, 255).astype("uint8") for frame in frames]
        if save_path is not None:
            for target, frame in enumerate(frames):
                Image.fromarray(frame).save(os.path.join(save_path, "%05d.png" % target))
