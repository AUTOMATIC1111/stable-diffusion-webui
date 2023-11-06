import os
import json
import albumentations
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from taming.data.sflckr import SegmentationBase # for examples included in repo


class Examples(SegmentationBase):
    def __init__(self, size=256, random_crop=False, interpolation="bicubic"):
        super().__init__(data_csv="data/coco_examples.txt",
                         data_root="data/coco_images",
                         segmentation_root="data/coco_segmentations",
                         size=size, random_crop=random_crop,
                         interpolation=interpolation,
                         n_labels=183, shift_segmentation=True)


class CocoBase(Dataset):
    """needed for (image, caption, segmentation) pairs"""
    def __init__(self, size=None, dataroot="", datajson="", onehot_segmentation=False, use_stuffthing=False,
                 crop_size=None, force_no_crop=False, given_files=None):
        self.split = self.get_split()
        self.size = size
        if crop_size is None:
            self.crop_size = size
        else:
            self.crop_size = crop_size

        self.onehot = onehot_segmentation       # return segmentation as rgb or one hot
        self.stuffthing = use_stuffthing        # include thing in segmentation
        if self.onehot and not self.stuffthing:
            raise NotImplemented("One hot mode is only supported for the "
                                 "stuffthings version because labels are stored "
                                 "a bit different.")

        data_json = datajson
        with open(data_json) as json_file:
            self.json_data = json.load(json_file)
            self.img_id_to_captions = dict()
            self.img_id_to_filepath = dict()
            self.img_id_to_segmentation_filepath = dict()

        assert data_json.split("/")[-1] in ["captions_train2017.json",
                                            "captions_val2017.json"]
        if self.stuffthing:
            self.segmentation_prefix = (
                "data/cocostuffthings/val2017" if
                data_json.endswith("captions_val2017.json") else
                "data/cocostuffthings/train2017")
        else:
            self.segmentation_prefix = (
                "data/coco/annotations/stuff_val2017_pixelmaps" if
                data_json.endswith("captions_val2017.json") else
                "data/coco/annotations/stuff_train2017_pixelmaps")

        imagedirs = self.json_data["images"]
        self.labels = {"image_ids": list()}
        for imgdir in tqdm(imagedirs, desc="ImgToPath"):
            self.img_id_to_filepath[imgdir["id"]] = os.path.join(dataroot, imgdir["file_name"])
            self.img_id_to_captions[imgdir["id"]] = list()
            pngfilename = imgdir["file_name"].replace("jpg", "png")
            self.img_id_to_segmentation_filepath[imgdir["id"]] = os.path.join(
                self.segmentation_prefix, pngfilename)
            if given_files is not None:
                if pngfilename in given_files:
                    self.labels["image_ids"].append(imgdir["id"])
            else:
                self.labels["image_ids"].append(imgdir["id"])

        capdirs = self.json_data["annotations"]
        for capdir in tqdm(capdirs, desc="ImgToCaptions"):
            # there are in average 5 captions per image
            self.img_id_to_captions[capdir["image_id"]].append(np.array([capdir["caption"]]))

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        if self.split=="validation":
            self.cropper = albumentations.CenterCrop(height=self.crop_size, width=self.crop_size)
        else:
            self.cropper = albumentations.RandomCrop(height=self.crop_size, width=self.crop_size)
        self.preprocessor = albumentations.Compose(
            [self.rescaler, self.cropper],
            additional_targets={"segmentation": "image"})
        if force_no_crop:
            self.rescaler = albumentations.Resize(height=self.size, width=self.size)
            self.preprocessor = albumentations.Compose(
                [self.rescaler],
                additional_targets={"segmentation": "image"})

    def __len__(self):
        return len(self.labels["image_ids"])

    def preprocess_image(self, image_path, segmentation_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        segmentation = Image.open(segmentation_path)
        if not self.onehot and not segmentation.mode == "RGB":
            segmentation = segmentation.convert("RGB")
        segmentation = np.array(segmentation).astype(np.uint8)
        if self.onehot:
            assert self.stuffthing
            # stored in caffe format: unlabeled==255. stuff and thing from
            # 0-181. to be compatible with the labels in
            # https://github.com/nightrome/cocostuff/blob/master/labels.txt
            # we shift stuffthing one to the right and put unlabeled in zero
            # as long as segmentation is uint8 shifting to right handles the
            # latter too
            assert segmentation.dtype == np.uint8
            segmentation = segmentation + 1

        processed = self.preprocessor(image=image, segmentation=segmentation)
        image, segmentation = processed["image"], processed["segmentation"]
        image = (image / 127.5 - 1.0).astype(np.float32)

        if self.onehot:
            assert segmentation.dtype == np.uint8
            # make it one hot
            n_labels = 183
            flatseg = np.ravel(segmentation)
            onehot = np.zeros((flatseg.size, n_labels), dtype=np.bool)
            onehot[np.arange(flatseg.size), flatseg] = True
            onehot = onehot.reshape(segmentation.shape + (n_labels,)).astype(int)
            segmentation = onehot
        else:
            segmentation = (segmentation / 127.5 - 1.0).astype(np.float32)
        return image, segmentation

    def __getitem__(self, i):
        img_path = self.img_id_to_filepath[self.labels["image_ids"][i]]
        seg_path = self.img_id_to_segmentation_filepath[self.labels["image_ids"][i]]
        image, segmentation = self.preprocess_image(img_path, seg_path)
        captions = self.img_id_to_captions[self.labels["image_ids"][i]]
        # randomly draw one of all available captions per image
        caption = captions[np.random.randint(0, len(captions))]
        example = {"image": image,
                   "caption": [str(caption[0])],
                   "segmentation": segmentation,
                   "img_path": img_path,
                   "seg_path": seg_path,
                   "filename_": img_path.split(os.sep)[-1]
                    }
        return example


class CocoImagesAndCaptionsTrain(CocoBase):
    """returns a pair of (image, caption)"""
    def __init__(self, size, onehot_segmentation=False, use_stuffthing=False, crop_size=None, force_no_crop=False):
        super().__init__(size=size,
                         dataroot="data/coco/train2017",
                         datajson="data/coco/annotations/captions_train2017.json",
                         onehot_segmentation=onehot_segmentation,
                         use_stuffthing=use_stuffthing, crop_size=crop_size, force_no_crop=force_no_crop)

    def get_split(self):
        return "train"


class CocoImagesAndCaptionsValidation(CocoBase):
    """returns a pair of (image, caption)"""
    def __init__(self, size, onehot_segmentation=False, use_stuffthing=False, crop_size=None, force_no_crop=False,
                 given_files=None):
        super().__init__(size=size,
                         dataroot="data/coco/val2017",
                         datajson="data/coco/annotations/captions_val2017.json",
                         onehot_segmentation=onehot_segmentation,
                         use_stuffthing=use_stuffthing, crop_size=crop_size, force_no_crop=force_no_crop,
                         given_files=given_files)

    def get_split(self):
        return "validation"
