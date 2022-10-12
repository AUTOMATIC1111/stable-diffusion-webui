import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random
import tqdm
from modules import devices, shared
import re

re_tag = re.compile(r"[a-zA-Z][_\w\d()]+")


class PersonalizedBase(Dataset):
    def __init__(self, data_root, width, height, repeats, flip_p=0.5, placeholder_token="*", model=None, device=None, template_file=None, include_cond=False):

        self.placeholder_token = placeholder_token

        self.width = width
        self.height = height
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        self.dataset = []

        with open(template_file, "r") as file:
            lines = [x.strip() for x in file.readlines()]

        self.lines = lines

        assert data_root, 'dataset directory not specified'

        cond_model = shared.sd_model.cond_stage_model

        self.image_paths = [os.path.join(data_root, file_path) for file_path in os.listdir(data_root)]
        print("Preparing dataset...")
        for path in tqdm.tqdm(self.image_paths):
            try:
                image = Image.open(path).convert('RGB').resize((self.width, self.height), PIL.Image.BICUBIC)
            except Exception:
                continue

            filename = os.path.basename(path)
            filename_tokens = os.path.splitext(filename)[0]
            filename_tokens = re_tag.findall(filename_tokens)

            npimage = np.array(image).astype(np.uint8)
            npimage = (npimage / 127.5 - 1.0).astype(np.float32)

            torchdata = torch.from_numpy(npimage).to(device=device, dtype=torch.float32)
            torchdata = torch.moveaxis(torchdata, 2, 0)

            init_latent = model.get_first_stage_encoding(model.encode_first_stage(torchdata.unsqueeze(dim=0))).squeeze()
            init_latent = init_latent.to(devices.cpu)

            if include_cond:
                text = self.create_text(filename_tokens)
                cond = cond_model([text]).to(devices.cpu)
            else:
                cond = None

            self.dataset.append((init_latent, filename_tokens, cond))

        self.length = len(self.dataset) * repeats

        self.initial_indexes = np.arange(self.length) % len(self.dataset)
        self.indexes = None
        self.shuffle()

    def shuffle(self):
        self.indexes = self.initial_indexes[torch.randperm(self.initial_indexes.shape[0])]

    def create_text(self, filename_tokens):
        text = random.choice(self.lines)
        text = text.replace("[name]", self.placeholder_token)
        text = text.replace("[filewords]", ' '.join(filename_tokens))
        return text

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i % len(self.dataset) == 0:
            self.shuffle()

        index = self.indexes[i % len(self.indexes)]
        x, filename_tokens, cond = self.dataset[index]

        text = self.create_text(filename_tokens)
        return x, text, cond
