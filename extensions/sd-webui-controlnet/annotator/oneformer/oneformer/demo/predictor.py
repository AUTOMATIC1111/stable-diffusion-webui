# Copyright (c) Facebook, Inc. and its affiliates.
# Copied from: https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py
import atexit
import bisect
import multiprocessing as mp
from collections import deque

import cv2
import torch

from annotator.oneformer.detectron2.data import MetadataCatalog
from defaults import DefaultPredictor
from annotator.oneformer.detectron2.utils.video_visualizer import VideoVisualizer
from visualizer import ColorMode, Visualizer


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        if 'cityscapes_fine_sem_seg_val' in cfg.DATASETS.TEST[0]:
            from cityscapesscripts.helpers.labels import labels
            stuff_colors = [k.color for k in labels if k.trainId != 255]
            self.metadata = self.metadata.set(stuff_colors=stuff_colors)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image, task, sem_gt, pan_gt, ins_gt, box_gt):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        vis_output = {}
        
        if task == 'panoptic':
            visualizer = Visualizer(image, metadata=self.metadata, instance_mode=0)
            predictions = self.predictor(image, "panoptic")
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output['panoptic'] = visualizer.draw_panoptic_seg_predictions(
            panoptic_seg.to(self.cpu_device), segments_info, alpha=1
        )

            # visualizer = Visualizer(image, metadata=self.metadata, instance_mode=0)
            # vis_output['pan_gt'] = visualizer.draw_panoptic_seg(
            #     pan_gt[0].to(self.cpu_device), pan_gt[1], alpha=1
            # )

        if task == 'panoptic' or task == 'semantic':
            visualizer = Visualizer(image, metadata=self.metadata, instance_mode=1)
            predictions = self.predictor(image, "semantic")
            vis_output['semantic'] = visualizer.draw_sem_seg(
                predictions["sem_seg"].argmax(dim=0).to(self.cpu_device), alpha=1
            )
            
            # visualizer = Visualizer(image, metadata=self.metadata, instance_mode=1)
            # vis_output['gt_sem'] = visualizer.draw_sem_seg(
            #     sem_gt.to(self.cpu_device), alpha=1
            # )

        if task == 'panoptic' or task == 'instance':
            visualizer = Visualizer(image, metadata=self.metadata, instance_mode=2)
            predictions = self.predictor(image, "instance")
            instances = predictions["instances"].to(self.cpu_device)
            vis_output['instance'] = visualizer.draw_instance_predictions(predictions=instances, alpha=1)

            if 'boxes' in predictions:
                boxes, labels, scores  = predictions["boxes"]
                visualizer = Visualizer(image, False, metadata=self.metadata, instance_mode=0)
                vis_output['boxes'] = visualizer.draw_box_predictions(
                        boxes.to(self.cpu_device), labels.to(self.cpu_device), scores.to(self.cpu_device))
            
            
            # visualizer = Visualizer(image, metadata=self.metadata, instance_mode=2)
            # vis_output['ins_gt'] = visualizer.draw_instance_predictions(predictions=ins_gt.to(self.cpu_device), alpha=1)
        # vis_output['input'] = visualizer.get_image(image)

        return predictions, vis_output


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
