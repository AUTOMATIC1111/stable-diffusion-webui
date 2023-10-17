import torch

from ldm.modules.midas.api import load_midas_transform


class AddMiDaS(object):
    def __init__(self, model_type):
        super().__init__()
        self.transform = load_midas_transform(model_type)

    def pt2np(self, x):
        x = ((x + 1.0) * .5).detach().cpu().numpy()
        return x

    def np2pt(self, x):
        x = torch.from_numpy(x) * 2 - 1.
        return x

    def __call__(self, sample):
        # sample['jpg'] is tensor hwc in [-1, 1] at this point
        x = self.pt2np(sample['jpg'])
        x = self.transform({"image": x})["image"]
        sample['midas_in'] = x
        return sample