from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image
import torch
import numpy as np
import random
import decord
from decord import VideoReader
import json
import os
from data.utils import pre_caption

decord.bridge.set_bridge("torch")

class ImageNorm(object):
    """Apply Normalization to Image Pixels on GPU
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        
    def __call__(self, img):

        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.)
        return img.sub_(self.mean).div_(self.std)

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]
    
    
class VideoDataset(Dataset):

    def __init__(self, video_root, ann_root, num_frm=4, frm_sampling_strategy="rand", max_img_size=384, video_fmt='.mp4'):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''        
        url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/msrvtt_test.jsonl'
        filename = 'msrvtt_test.jsonl'

        download_url(url,ann_root)
        self.annotation = load_jsonl(os.path.join(ann_root,filename))
        
        self.num_frm = num_frm
        self.frm_sampling_strategy = frm_sampling_strategy
        self.max_img_size = max_img_size
        self.video_root = video_root
        self.video_fmt = video_fmt
        self.img_norm = ImageNorm(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

        self.text = [pre_caption(ann['caption'],40) for ann in self.annotation]
        self.txt2video = [i for i in range(len(self.annotation))]
        self.video2txt = self.txt2video               
            
            
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]  

        video_path = os.path.join(self.video_root, ann['clip_name'] + self.video_fmt) 

        vid_frm_array = self._load_video_from_path_decord(video_path, height=self.max_img_size, width=self.max_img_size)

        video = self.img_norm(vid_frm_array.float())
        
        return video, ann['clip_name']

        

    def _load_video_from_path_decord(self, video_path, height=None, width=None, start_time=None, end_time=None, fps=-1):
        try:
            if not height or not width:
                vr = VideoReader(video_path)
            else:
                vr = VideoReader(video_path, width=width, height=height)

            vlen = len(vr)

            if start_time or end_time:
                assert fps > 0, 'must provide video fps if specifying start and end time.'

                start_idx = min(int(start_time * fps), vlen)
                end_idx = min(int(end_time * fps), vlen)
            else:
                start_idx, end_idx = 0, vlen

            if self.frm_sampling_strategy == 'uniform':
                frame_indices = np.arange(start_idx, end_idx, vlen / self.num_frm, dtype=int)
            elif self.frm_sampling_strategy == 'rand':
                frame_indices = sorted(random.sample(range(vlen), self.num_frm))
            elif self.frm_sampling_strategy == 'headtail':
                frame_indices_head = sorted(random.sample(range(vlen // 2), self.num_frm // 2))
                frame_indices_tail = sorted(random.sample(range(vlen // 2, vlen), self.num_frm // 2))
                frame_indices = frame_indices_head + frame_indices_tail
            else:
                raise NotImplementedError('Invalid sampling strategy {} '.format(self.frm_sampling_strategy))

            raw_sample_frms = vr.get_batch(frame_indices)
        except Exception as e:
            return None

        raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2)

        return raw_sample_frms
