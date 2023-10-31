import os
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset

from taming.data.sflckr import SegmentationBase # for examples included in repo


class Examples(SegmentationBase):
    def __init__(self, size=256, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv="data/ade20k_examples.txt",
                         data_root="data/ade20k_images",
                         segmentation_root="data/ade20k_segmentations",
                         size=size, random_crop=random_crop,
                         interpolation=interpolation,
                         n_labels=151, shift_segmentation=False)


# With semantic map and scene label
class ADE20kBase(Dataset):
    def __init__(self, config=None, size=None, random_crop=False, interpolation="bicubic", crop_size=None):
        self.split = self.get_split()
        self.n_labels = 151 # unknown + 150
        self.data_csv = {"train": "data/ade20k_train.txt",
                         "validation": "data/ade20k_test.txt"}[self.split]
        self.data_root = "data/ade20k_root"
        with open(os.path.join(self.data_root, "sceneCategories.txt"), "r") as f:
            self.scene_categories = f.read().splitlines()
        self.scene_categories = dict(line.split() for line in self.scene_categories)
        with open(self.data_csv, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, "images", l)
                           for l in self.image_paths],
            "relative_segmentation_path_": [l.replace(".jpg", ".png")
                                            for l in self.image_paths],
            "segmentation_path_": [os.path.join(self.data_root, "annotations",
                                                l.replace(".jpg", ".png"))
                                   for l in self.image_paths],
            "scene_category": [self.scene_categories[l.split("/")[1].replace(".jpg", "")]
                               for l in self.image_paths],
        }

        size = None if size is not None and size<=0 else size
        self.size = size
        if crop_size is None:
            self.crop_size = size if size is not None else None
        else:
            self.crop_size = crop_size
        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                 interpolation=self.interpolation)
            self.segmentation_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                        interpolation=cv2.INTER_NEAREST)

        if crop_size is not None:
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.crop_size, width=self.crop_size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.crop_size, width=self.crop_size)
            self.preprocessor = self.cropper

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]
        segmentation = Image.open(example["segmentation_path_"])
        segmentation = np.array(segmentation).astype(np.uint8)
        if self.size is not None:
            segmentation = self.segmentation_rescaler(image=segmentation)["image"]
        if self.size is not None:
            processed = self.preprocessor(image=image, mask=segmentation)
        else:
            processed = {"image": image, "mask": segmentation}
        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
        segmentation = processed["mask"]
        onehot = np.eye(self.n_labels)[segmentation]
        example["segmentation"] = onehot
        return example


class ADE20kTrain(ADE20kBase):
    # default to random_crop=True
    def __init__(self, config=None, size=None, random_crop=True, interpolation="bicubic", crop_size=None):
        super().__init__(config=config, size=size, random_crop=random_crop,
                          interpolation=interpolation, crop_size=crop_size)

    def get_split(self):
        return "train"


class ADE20kValidation(ADE20kBase):
    def get_split(self):
        return "validation"


if __name__ == "__main__":
    dset = ADE20kValidation()
    ex = dset[0]
    for k in ["image", "scene_category", "segmentation"]:
        print(type(ex[k]))
        try:
            print(ex[k].shape)
        except:
            print(ex[k])
