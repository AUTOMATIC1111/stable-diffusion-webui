import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random
import tqdm
from modules import devices


class PersonalizedBase(Dataset):
    def __init__(self, data_root, size=None, repeats=100, flip_p=0.5, placeholder_token="*", width=512, height=512, model=None, device=None, template_file=None):

        self.placeholder_token = placeholder_token

        self.size = size
        self.width = width
        self.height = height
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        self.dataset = []

        with open(template_file, "r") as file:
            lines = [x.strip() for x in file.readlines()]

        self.lines = lines

        assert data_root, 'dataset directory not specified'

        self.image_paths = [os.path.join(data_root, file_path) for file_path in os.listdir(data_root)]
        print("Preparing dataset...")
        for path in tqdm.tqdm(self.image_paths):
            image = Image.open(path)
            image = image.convert('RGB')
            image = image.resize((self.width, self.height), PIL.Image.BICUBIC)

            filename = os.path.basename(path)
            filename_tokens = os.path.splitext(filename)[0].replace('_', '-').replace(' ', '-').split('-')
            filename_tokens = [token for token in filename_tokens if token.isalpha()]

            npimage = np.array(image).astype(np.uint8)
            npimage = (npimage / 127.5 - 1.0).astype(np.float32)

            torchdata = torch.from_numpy(npimage).to(device=device, dtype=torch.float32)
            torchdata = torch.moveaxis(torchdata, 2, 0)

            init_latent = model.get_first_stage_encoding(model.encode_first_stage(torchdata.unsqueeze(dim=0))).squeeze()
            init_latent = init_latent.to(devices.cpu)

            self.dataset.append((init_latent, filename_tokens))

        self.length = len(self.dataset) * repeats

        self.initial_indexes = np.arange(self.length) % len(self.dataset)
        self.indexes = None
        self.shuffle()

    def shuffle(self):
        self.indexes = self.initial_indexes[torch.randperm(self.initial_indexes.shape[0])]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if i % len(self.dataset) == 0:
            self.shuffle()

        index = self.indexes[i % len(self.indexes)]
        x, filename_tokens = self.dataset[index]

        text = random.choice(self.lines)
        text = text.replace("[name]", self.placeholder_token)
        text = text.replace("[filewords]", ' '.join(filename_tokens))

        return x, text
