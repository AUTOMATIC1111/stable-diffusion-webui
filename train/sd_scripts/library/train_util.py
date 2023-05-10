# common functions for training

import argparse
import ast
import asyncio
import importlib
import json
import pathlib
import re
import shutil
import time
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from accelerate import Accelerator
import gc
import glob
import math
import os
import random
import hashlib
import subprocess
from io import BytesIO
import toml

from tqdm import tqdm
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torchvision import transforms
from transformers import CLIPTokenizer
import transformers
import diffusers
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
)
from huggingface_hub import hf_hub_download
import albumentations as albu
import numpy as np
from PIL import Image
import cv2
from einops import rearrange
from torch import einsum
import safetensors.torch
from library.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
import library.model_util as model_util
import library.huggingface_util as huggingface_util

# Tokenizer: checkpointから読み込むのではなくあらかじめ提供されているものを使う
TOKENIZER_PATH = "openai/clip-vit-large-patch14"
V2_STABLE_DIFFUSION_PATH = "stabilityai/stable-diffusion-2"  # ここからtokenizerだけ使う v2とv2.1はtokenizer仕様は同じ

# checkpointファイル名
EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
EPOCH_DIFFUSERS_DIR_NAME = "{}-{:06d}"
LAST_STATE_NAME = "{}-state"
DEFAULT_EPOCH_NAME = "epoch"
DEFAULT_LAST_OUTPUT_NAME = "last"

DEFAULT_STEP_NAME = "at"
STEP_STATE_NAME = "{}-step{:08d}-state"
STEP_FILE_NAME = "{}-step{:08d}"
STEP_DIFFUSERS_DIR_NAME = "{}-step{:08d}"

# region dataset

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]


class ImageInfo:
    def __init__(self, image_key: str, num_repeats: int, caption: str, is_reg: bool, absolute_path: str) -> None:
        self.image_key: str = image_key
        self.num_repeats: int = num_repeats
        self.caption: str = caption
        self.is_reg: bool = is_reg
        self.absolute_path: str = absolute_path
        self.image_size: Tuple[int, int] = None
        self.resized_size: Tuple[int, int] = None
        self.bucket_reso: Tuple[int, int] = None
        self.latents: torch.Tensor = None
        self.latents_flipped: torch.Tensor = None
        self.latents_npz: str = None
        self.latents_npz_flipped: str = None


class BucketManager:
    def __init__(self, no_upscale, max_reso, min_size, max_size, reso_steps) -> None:
        self.no_upscale = no_upscale
        if max_reso is None:
            self.max_reso = None
            self.max_area = None
        else:
            self.max_reso = max_reso
            self.max_area = max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps

        self.resos = []
        self.reso_to_id = {}
        self.buckets = []  # 前処理時は (image_key, image)、学習時は image_key

    def add_image(self, reso, image):
        bucket_id = self.reso_to_id[reso]
        self.buckets[bucket_id].append(image)

    def shuffle(self):
        for bucket in self.buckets:
            random.shuffle(bucket)

    def sort(self):
        # 解像度順にソートする（表示時、メタデータ格納時の見栄えをよくするためだけ）。bucketsも入れ替えてreso_to_idも振り直す
        sorted_resos = self.resos.copy()
        sorted_resos.sort()

        sorted_buckets = []
        sorted_reso_to_id = {}
        for i, reso in enumerate(sorted_resos):
            bucket_id = self.reso_to_id[reso]
            sorted_buckets.append(self.buckets[bucket_id])
            sorted_reso_to_id[reso] = i

        self.resos = sorted_resos
        self.buckets = sorted_buckets
        self.reso_to_id = sorted_reso_to_id

    def make_buckets(self):
        resos = model_util.make_bucket_resolutions(self.max_reso, self.min_size, self.max_size, self.reso_steps)
        self.set_predefined_resos(resos)

    def set_predefined_resos(self, resos):
        # 規定サイズから選ぶ場合の解像度、aspect ratioの情報を格納しておく
        self.predefined_resos = resos.copy()
        self.predefined_resos_set = set(resos)
        self.predefined_aspect_ratios = np.array([w / h for w, h in resos])

    def add_if_new_reso(self, reso):
        if reso not in self.reso_to_id:
            bucket_id = len(self.resos)
            self.reso_to_id[reso] = bucket_id
            self.resos.append(reso)
            self.buckets.append([])
            # print(reso, bucket_id, len(self.buckets))

    def round_to_steps(self, x):
        x = int(x + 0.5)
        return x - x % self.reso_steps

    def select_bucket(self, image_width, image_height):
        aspect_ratio = image_width / image_height
        if not self.no_upscale:
            # 同じaspect ratioがあるかもしれないので（fine tuningで、no_upscale=Trueで前処理した場合）、解像度が同じものを優先する
            reso = (image_width, image_height)
            if reso in self.predefined_resos_set:
                pass
            else:
                ar_errors = self.predefined_aspect_ratios - aspect_ratio
                predefined_bucket_id = np.abs(ar_errors).argmin()  # 当該解像度以外でaspect ratio errorが最も少ないもの
                reso = self.predefined_resos[predefined_bucket_id]

            ar_reso = reso[0] / reso[1]
            if aspect_ratio > ar_reso:  # 横が長い→縦を合わせる
                scale = reso[1] / image_height
            else:
                scale = reso[0] / image_width

            resized_size = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
            # print("use predef", image_width, image_height, reso, resized_size)
        else:
            if image_width * image_height > self.max_area:
                # 画像が大きすぎるのでアスペクト比を保ったまま縮小することを前提にbucketを決める
                resized_width = math.sqrt(self.max_area * aspect_ratio)
                resized_height = self.max_area / resized_width
                assert abs(resized_width / resized_height - aspect_ratio) < 1e-2, "aspect is illegal"

                # リサイズ後の短辺または長辺をreso_steps単位にする：aspect ratioの差が少ないほうを選ぶ
                # 元のbucketingと同じロジック
                b_width_rounded = self.round_to_steps(resized_width)
                b_height_in_wr = self.round_to_steps(b_width_rounded / aspect_ratio)
                ar_width_rounded = b_width_rounded / b_height_in_wr

                b_height_rounded = self.round_to_steps(resized_height)
                b_width_in_hr = self.round_to_steps(b_height_rounded * aspect_ratio)
                ar_height_rounded = b_width_in_hr / b_height_rounded

                # print(b_width_rounded, b_height_in_wr, ar_width_rounded)
                # print(b_width_in_hr, b_height_rounded, ar_height_rounded)

                if abs(ar_width_rounded - aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
                    resized_size = (b_width_rounded, int(b_width_rounded / aspect_ratio + 0.5))
                else:
                    resized_size = (int(b_height_rounded * aspect_ratio + 0.5), b_height_rounded)
                # print(resized_size)
            else:
                resized_size = (image_width, image_height)  # リサイズは不要

            # 画像のサイズ未満をbucketのサイズとする（paddingせずにcroppingする）
            bucket_width = resized_size[0] - resized_size[0] % self.reso_steps
            bucket_height = resized_size[1] - resized_size[1] % self.reso_steps
            # print("use arbitrary", image_width, image_height, resized_size, bucket_width, bucket_height)

            reso = (bucket_width, bucket_height)

        self.add_if_new_reso(reso)

        ar_error = (reso[0] / reso[1]) - aspect_ratio
        return reso, resized_size, ar_error


class BucketBatchIndex(NamedTuple):
    bucket_index: int
    bucket_batch_size: int
    batch_index: int


class AugHelper:
    def __init__(self):
        # prepare all possible augmentators
        color_aug_method = albu.OneOf(
            [
                albu.HueSaturationValue(8, 0, 0, p=0.5),
                albu.RandomGamma((95, 105), p=0.5),
            ],
            p=0.33,
        )
        flip_aug_method = albu.HorizontalFlip(p=0.5)

        # key: (use_color_aug, use_flip_aug)
        self.augmentors = {
            (True, True): albu.Compose(
                [
                    color_aug_method,
                    flip_aug_method,
                ],
                p=1.0,
            ),
            (True, False): albu.Compose(
                [
                    color_aug_method,
                ],
                p=1.0,
            ),
            (False, True): albu.Compose(
                [
                    flip_aug_method,
                ],
                p=1.0,
            ),
            (False, False): None,
        }

    def get_augmentor(self, use_color_aug: bool, use_flip_aug: bool) -> Optional[albu.Compose]:
        return self.augmentors[(use_color_aug, use_flip_aug)]


class BaseSubset:
    def __init__(
        self,
        image_dir: Optional[str],
        num_repeats: int,
        shuffle_caption: bool,
        keep_tokens: int,
        color_aug: bool,
        flip_aug: bool,
        face_crop_aug_range: Optional[Tuple[float, float]],
        random_crop: bool,
        caption_dropout_rate: float,
        caption_dropout_every_n_epochs: int,
        caption_tag_dropout_rate: float,
        token_warmup_min: int,
        token_warmup_step: Union[float, int],
    ) -> None:
        self.image_dir = image_dir
        self.num_repeats = num_repeats
        self.shuffle_caption = shuffle_caption
        self.keep_tokens = keep_tokens
        self.color_aug = color_aug
        self.flip_aug = flip_aug
        self.face_crop_aug_range = face_crop_aug_range
        self.random_crop = random_crop
        self.caption_dropout_rate = caption_dropout_rate
        self.caption_dropout_every_n_epochs = caption_dropout_every_n_epochs
        self.caption_tag_dropout_rate = caption_tag_dropout_rate

        self.token_warmup_min = token_warmup_min  # step=0におけるタグの数
        self.token_warmup_step = token_warmup_step  # N（N<1ならN*max_train_steps）ステップ目でタグの数が最大になる

        self.img_count = 0


class DreamBoothSubset(BaseSubset):
    def __init__(
        self,
        image_dir: str,
        is_reg: bool,
        class_tokens: Optional[str],
        caption_extension: str,
        num_repeats,
        shuffle_caption,
        keep_tokens,
        color_aug,
        flip_aug,
        face_crop_aug_range,
        random_crop,
        caption_dropout_rate,
        caption_dropout_every_n_epochs,
        caption_tag_dropout_rate,
        token_warmup_min,
        token_warmup_step,
    ) -> None:
        assert image_dir is not None, "image_dir must be specified / image_dirは指定が必須です"

        super().__init__(
            image_dir,
            num_repeats,
            shuffle_caption,
            keep_tokens,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            token_warmup_min,
            token_warmup_step,
        )

        self.is_reg = is_reg
        self.class_tokens = class_tokens
        self.caption_extension = caption_extension

    def __eq__(self, other) -> bool:
        if not isinstance(other, DreamBoothSubset):
            return NotImplemented
        return self.image_dir == other.image_dir


class FineTuningSubset(BaseSubset):
    def __init__(
        self,
        image_dir,
        metadata_file: str,
        num_repeats,
        shuffle_caption,
        keep_tokens,
        color_aug,
        flip_aug,
        face_crop_aug_range,
        random_crop,
        caption_dropout_rate,
        caption_dropout_every_n_epochs,
        caption_tag_dropout_rate,
        token_warmup_min,
        token_warmup_step,
    ) -> None:
        assert metadata_file is not None, "metadata_file must be specified / metadata_fileは指定が必須です"

        super().__init__(
            image_dir,
            num_repeats,
            shuffle_caption,
            keep_tokens,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            token_warmup_min,
            token_warmup_step,
        )

        self.metadata_file = metadata_file

    def __eq__(self, other) -> bool:
        if not isinstance(other, FineTuningSubset):
            return NotImplemented
        return self.metadata_file == other.metadata_file


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self, tokenizer: CLIPTokenizer, max_token_length: int, resolution: Optional[Tuple[int, int]], debug_dataset: bool
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        # width/height is used when enable_bucket==False
        self.width, self.height = (None, None) if resolution is None else resolution
        self.debug_dataset = debug_dataset

        self.subsets: List[Union[DreamBoothSubset, FineTuningSubset]] = []

        self.token_padding_disabled = False
        self.tag_frequency = {}
        self.XTI_layers = None
        self.token_strings = None

        self.enable_bucket = False
        self.bucket_manager: BucketManager = None  # not initialized
        self.min_bucket_reso = None
        self.max_bucket_reso = None
        self.bucket_reso_steps = None
        self.bucket_no_upscale = None
        self.bucket_info = None  # for metadata

        self.tokenizer_max_length = self.tokenizer.model_max_length if max_token_length is None else max_token_length + 2

        self.current_epoch: int = 0  # インスタンスがepochごとに新しく作られるようなので外側から渡さないとダメ

        self.current_step: int = 0
        self.max_train_steps: int = 0
        self.seed: int = 0

        # augmentation
        self.aug_helper = AugHelper()

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.image_data: Dict[str, ImageInfo] = {}
        self.image_to_subset: Dict[str, Union[DreamBoothSubset, FineTuningSubset]] = {}

        self.replacements = {}

    def set_seed(self, seed):
        self.seed = seed

    def set_current_epoch(self, epoch):
        if not self.current_epoch == epoch:  # epochが切り替わったらバケツをシャッフルする
            self.shuffle_buckets()
        self.current_epoch = epoch

    def set_current_step(self, step):
        self.current_step = step

    def set_max_train_steps(self, max_train_steps):
        self.max_train_steps = max_train_steps

    def set_tag_frequency(self, dir_name, captions):
        frequency_for_dir = self.tag_frequency.get(dir_name, {})
        self.tag_frequency[dir_name] = frequency_for_dir
        for caption in captions:
            for tag in caption.split(","):
                tag = tag.strip()
                if tag:
                    tag = tag.lower()
                    frequency = frequency_for_dir.get(tag, 0)
                    frequency_for_dir[tag] = frequency + 1

    def disable_token_padding(self):
        self.token_padding_disabled = True

    def enable_XTI(self, layers=None, token_strings=None):
        self.XTI_layers = layers
        self.token_strings = token_strings

    def add_replacement(self, str_from, str_to):
        self.replacements[str_from] = str_to

    def process_caption(self, subset: BaseSubset, caption):
        # dropoutの決定：tag dropがこのメソッド内にあるのでここで行うのが良い
        is_drop_out = subset.caption_dropout_rate > 0 and random.random() < subset.caption_dropout_rate
        is_drop_out = (
            is_drop_out
            or subset.caption_dropout_every_n_epochs > 0
            and self.current_epoch % subset.caption_dropout_every_n_epochs == 0
        )

        if is_drop_out:
            caption = ""
        else:
            if subset.shuffle_caption or subset.token_warmup_step > 0 or subset.caption_tag_dropout_rate > 0:
                tokens = [t.strip() for t in caption.strip().split(",")]
                if subset.token_warmup_step < 1:  # 初回に上書きする
                    subset.token_warmup_step = math.floor(subset.token_warmup_step * self.max_train_steps)
                if subset.token_warmup_step and self.current_step < subset.token_warmup_step:
                    tokens_len = (
                        math.floor((self.current_step) * ((len(tokens) - subset.token_warmup_min) / (subset.token_warmup_step)))
                        + subset.token_warmup_min
                    )
                    tokens = tokens[:tokens_len]

                def dropout_tags(tokens):
                    if subset.caption_tag_dropout_rate <= 0:
                        return tokens
                    l = []
                    for token in tokens:
                        if random.random() >= subset.caption_tag_dropout_rate:
                            l.append(token)
                    return l

                fixed_tokens = []
                flex_tokens = tokens[:]
                if subset.keep_tokens > 0:
                    fixed_tokens = flex_tokens[: subset.keep_tokens]
                    flex_tokens = tokens[subset.keep_tokens :]

                if subset.shuffle_caption:
                    random.shuffle(flex_tokens)

                flex_tokens = dropout_tags(flex_tokens)

                caption = ", ".join(fixed_tokens + flex_tokens)

            # textual inversion対応
            for str_from, str_to in self.replacements.items():
                if str_from == "":
                    # replace all
                    if type(str_to) == list:
                        caption = random.choice(str_to)
                    else:
                        caption = str_to
                else:
                    caption = caption.replace(str_from, str_to)

        return caption

    def get_input_ids(self, caption):
        input_ids = self.tokenizer(
            caption, padding="max_length", truncation=True, max_length=self.tokenizer_max_length, return_tensors="pt"
        ).input_ids

        if self.tokenizer_max_length > self.tokenizer.model_max_length:
            input_ids = input_ids.squeeze(0)
            iids_list = []
            if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                # v1
                # 77以上の時は "<BOS> .... <EOS> <EOS> <EOS>" でトータル227とかになっているので、"<BOS>...<EOS>"の三連に変換する
                # 1111氏のやつは , で区切る、とかしているようだが　とりあえず単純に
                for i in range(
                    1, self.tokenizer_max_length - self.tokenizer.model_max_length + 2, self.tokenizer.model_max_length - 2
                ):  # (1, 152, 75)
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),
                        input_ids[i : i + self.tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )
                    ids_chunk = torch.cat(ids_chunk)
                    iids_list.append(ids_chunk)
            else:
                # v2
                # 77以上の時は "<BOS> .... <EOS> <PAD> <PAD>..." でトータル227とかになっているので、"<BOS>...<EOS> <PAD> <PAD> ..."の三連に変換する
                for i in range(
                    1, self.tokenizer_max_length - self.tokenizer.model_max_length + 2, self.tokenizer.model_max_length - 2
                ):
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),  # BOS
                        input_ids[i : i + self.tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )  # PAD or EOS
                    ids_chunk = torch.cat(ids_chunk)

                    # 末尾が <EOS> <PAD> または <PAD> <PAD> の場合は、何もしなくてよい
                    # 末尾が x <PAD/EOS> の場合は末尾を <EOS> に変える（x <EOS> なら結果的に変化なし）
                    if ids_chunk[-2] != self.tokenizer.eos_token_id and ids_chunk[-2] != self.tokenizer.pad_token_id:
                        ids_chunk[-1] = self.tokenizer.eos_token_id
                    # 先頭が <BOS> <PAD> ... の場合は <BOS> <EOS> <PAD> ... に変える
                    if ids_chunk[1] == self.tokenizer.pad_token_id:
                        ids_chunk[1] = self.tokenizer.eos_token_id

                    iids_list.append(ids_chunk)

            input_ids = torch.stack(iids_list)  # 3,77
        return input_ids

    def register_image(self, info: ImageInfo, subset: BaseSubset):
        self.image_data[info.image_key] = info
        self.image_to_subset[info.image_key] = subset

    def make_buckets(self):
        """
        bucketingを行わない場合も呼び出し必須（ひとつだけbucketを作る）
        min_size and max_size are ignored when enable_bucket is False
        """
        print("loading image sizes.")
        for info in tqdm(self.image_data.values()):
            if info.image_size is None:
                info.image_size = self.get_image_size(info.absolute_path)

        if self.enable_bucket:
            print("make buckets")
        else:
            print("prepare dataset")

        # bucketを作成し、画像をbucketに振り分ける
        if self.enable_bucket:
            if self.bucket_manager is None:  # fine tuningの場合でmetadataに定義がある場合は、すでに初期化済み
                self.bucket_manager = BucketManager(
                    self.bucket_no_upscale,
                    (self.width, self.height),
                    self.min_bucket_reso,
                    self.max_bucket_reso,
                    self.bucket_reso_steps,
                )
                if not self.bucket_no_upscale:
                    self.bucket_manager.make_buckets()
                else:
                    print(
                        "min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, because bucket reso is defined by image size automatically / bucket_no_upscaleが指定された場合は、bucketの解像度は画像サイズから自動計算されるため、min_bucket_resoとmax_bucket_resoは無視されます"
                    )

            img_ar_errors = []
            for image_info in self.image_data.values():
                image_width, image_height = image_info.image_size
                image_info.bucket_reso, image_info.resized_size, ar_error = self.bucket_manager.select_bucket(
                    image_width, image_height
                )

                # print(image_info.image_key, image_info.bucket_reso)
                img_ar_errors.append(abs(ar_error))

            self.bucket_manager.sort()
        else:
            self.bucket_manager = BucketManager(False, (self.width, self.height), None, None, None)
            self.bucket_manager.set_predefined_resos([(self.width, self.height)])  # ひとつの固定サイズbucketのみ
            for image_info in self.image_data.values():
                image_width, image_height = image_info.image_size
                image_info.bucket_reso, image_info.resized_size, _ = self.bucket_manager.select_bucket(image_width, image_height)

        for image_info in self.image_data.values():
            for _ in range(image_info.num_repeats):
                self.bucket_manager.add_image(image_info.bucket_reso, image_info.image_key)

        # bucket情報を表示、格納する
        if self.enable_bucket:
            self.bucket_info = {"buckets": {}}
            print("number of images (including repeats) / 各bucketの画像枚数（繰り返し回数を含む）")
            for i, (reso, bucket) in enumerate(zip(self.bucket_manager.resos, self.bucket_manager.buckets)):
                count = len(bucket)
                if count > 0:
                    self.bucket_info["buckets"][i] = {"resolution": reso, "count": len(bucket)}
                    print(f"bucket {i}: resolution {reso}, count: {len(bucket)}")

            img_ar_errors = np.array(img_ar_errors)
            mean_img_ar_error = np.mean(np.abs(img_ar_errors))
            self.bucket_info["mean_img_ar_error"] = mean_img_ar_error
            print(f"mean ar error (without repeats): {mean_img_ar_error}")

        # データ参照用indexを作る。このindexはdatasetのshuffleに用いられる
        self.buckets_indices: List(BucketBatchIndex) = []
        for bucket_index, bucket in enumerate(self.bucket_manager.buckets):
            batch_count = int(math.ceil(len(bucket) / self.batch_size))
            for batch_index in range(batch_count):
                self.buckets_indices.append(BucketBatchIndex(bucket_index, self.batch_size, batch_index))

            # ↓以下はbucketごとのbatch件数があまりにも増えて混乱を招くので元に戻す
            # 　学習時はステップ数がランダムなので、同一画像が同一batch内にあってもそれほど悪影響はないであろう、と考えられる
            #
            # # bucketが細分化されることにより、ひとつのbucketに一種類の画像のみというケースが増え、つまりそれは
            # # ひとつのbatchが同じ画像で占められることになるので、さすがに良くないであろう
            # # そのためバッチサイズを画像種類までに制限する
            # # ただそれでも同一画像が同一バッチに含まれる可能性はあるので、繰り返し回数が少ないほうがshuffleの品質は良くなることは間違いない？
            # # TO DO 正則化画像をepochまたがりで利用する仕組み
            # num_of_image_types = len(set(bucket))
            # bucket_batch_size = min(self.batch_size, num_of_image_types)
            # batch_count = int(math.ceil(len(bucket) / bucket_batch_size))
            # # print(bucket_index, num_of_image_types, bucket_batch_size, batch_count)
            # for batch_index in range(batch_count):
            #   self.buckets_indices.append(BucketBatchIndex(bucket_index, bucket_batch_size, batch_index))
            # ↑ここまで

        self.shuffle_buckets()
        self._length = len(self.buckets_indices)

    def shuffle_buckets(self):
        # set random seed for this epoch
        random.seed(self.seed + self.current_epoch)

        random.shuffle(self.buckets_indices)
        self.bucket_manager.shuffle()

    def load_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        img = np.array(image, np.uint8)
        return img

    def trim_and_resize_if_required(self, subset: BaseSubset, image, reso, resized_size):
        image_height, image_width = image.shape[0:2]

        if image_width != resized_size[0] or image_height != resized_size[1]:
            # リサイズする
            image = cv2.resize(image, resized_size, interpolation=cv2.INTER_AREA)  # INTER_AREAでやりたいのでcv2でリサイズ

        image_height, image_width = image.shape[0:2]
        if image_width > reso[0]:
            trim_size = image_width - reso[0]
            p = trim_size // 2 if not subset.random_crop else random.randint(0, trim_size)
            # print("w", trim_size, p)
            image = image[:, p : p + reso[0]]
        if image_height > reso[1]:
            trim_size = image_height - reso[1]
            p = trim_size // 2 if not subset.random_crop else random.randint(0, trim_size)
            # print("h", trim_size, p)
            image = image[p : p + reso[1]]

        assert (
            image.shape[0] == reso[1] and image.shape[1] == reso[0]
        ), f"internal error, illegal trimmed size: {image.shape}, {reso}"
        return image

    def is_latent_cacheable(self):
        return all([not subset.color_aug and not subset.random_crop for subset in self.subsets])

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True):
        # ちょっと速くした
        print("caching latents.")

        image_infos = list(self.image_data.values())

        # sort by resolution
        image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])

        # split by resolution
        batches = []
        batch = []
        for info in image_infos:
            subset = self.image_to_subset[info.image_key]

            if info.latents_npz is not None:
                info.latents = self.load_latents_from_npz(info, False)
                info.latents = torch.FloatTensor(info.latents)

                # might be None, but that's ok because check is done in dataset
                info.latents_flipped = self.load_latents_from_npz(info, True)
                if info.latents_flipped is not None:
                    info.latents_flipped = torch.FloatTensor(info.latents_flipped)
                continue

            # check disk cache exists and size of latents
            if cache_to_disk:
                # TODO: refactor to unify with FineTuningDataset
                info.latents_npz = os.path.splitext(info.absolute_path)[0] + ".npz"
                info.latents_npz_flipped = os.path.splitext(info.absolute_path)[0] + "_flip.npz"
                if not is_main_process:
                    continue

                cache_available = False
                expected_latents_size = (info.bucket_reso[1] // 8, info.bucket_reso[0] // 8)  # bucket_resoはWxHなので注意
                if os.path.exists(info.latents_npz):
                    cached_latents = np.load(info.latents_npz)["arr_0"]
                    if cached_latents.shape[1:3] == expected_latents_size:
                        cache_available = True

                        if subset.flip_aug:
                            cache_available = False
                            if os.path.exists(info.latents_npz_flipped):
                                cached_latents_flipped = np.load(info.latents_npz_flipped)["arr_0"]
                                if cached_latents_flipped.shape[1:3] == expected_latents_size:
                                    cache_available = True

                if cache_available:
                    continue

            # if last member of batch has different resolution, flush the batch
            if len(batch) > 0 and batch[-1].bucket_reso != info.bucket_reso:
                batches.append(batch)
                batch = []

            batch.append(info)

            # if number of data in batch is enough, flush the batch
            if len(batch) >= vae_batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0:
            batches.append(batch)

        if cache_to_disk and not is_main_process:  # don't cache latents in non-main process, set to info only
            return

        # iterate batches
        for batch in tqdm(batches, smoothing=1, total=len(batches)):
            images = []
            for info in batch:
                image = self.load_image(info.absolute_path)
                image = self.trim_and_resize_if_required(subset, image, info.bucket_reso, info.resized_size)
                image = self.image_transforms(image)
                images.append(image)

            img_tensors = torch.stack(images, dim=0)
            img_tensors = img_tensors.to(device=vae.device, dtype=vae.dtype)

            latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")

            for info, latent in zip(batch, latents):
                if cache_to_disk:
                    np.savez(info.latents_npz, latent.float().numpy())
                else:
                    info.latents = latent

            if subset.flip_aug:
                img_tensors = torch.flip(img_tensors, dims=[3])
                latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")
                for info, latent in zip(batch, latents):
                    if cache_to_disk:
                        np.savez(info.latents_npz_flipped, latent.float().numpy())
                    else:
                        info.latents_flipped = latent

    def get_image_size(self, image_path):
        image = Image.open(image_path)
        return image.size

    def load_image_with_face_info(self, subset: BaseSubset, image_path: str):
        img = self.load_image(image_path)

        face_cx = face_cy = face_w = face_h = 0
        if subset.face_crop_aug_range is not None:
            tokens = os.path.splitext(os.path.basename(image_path))[0].split("_")
            if len(tokens) >= 5:
                face_cx = int(tokens[-4])
                face_cy = int(tokens[-3])
                face_w = int(tokens[-2])
                face_h = int(tokens[-1])

        return img, face_cx, face_cy, face_w, face_h

    # いい感じに切り出す
    def crop_target(self, subset: BaseSubset, image, face_cx, face_cy, face_w, face_h):
        height, width = image.shape[0:2]
        if height == self.height and width == self.width:
            return image

        # 画像サイズはsizeより大きいのでリサイズする
        face_size = max(face_w, face_h)
        size = min(self.height, self.width)  # 短いほう
        min_scale = max(self.height / height, self.width / width)  # 画像がモデル入力サイズぴったりになる倍率（最小の倍率）
        min_scale = min(1.0, max(min_scale, size / (face_size * subset.face_crop_aug_range[1])))  # 指定した顔最小サイズ
        max_scale = min(1.0, max(min_scale, size / (face_size * subset.face_crop_aug_range[0])))  # 指定した顔最大サイズ
        if min_scale >= max_scale:  # range指定がmin==max
            scale = min_scale
        else:
            scale = random.uniform(min_scale, max_scale)

        nh = int(height * scale + 0.5)
        nw = int(width * scale + 0.5)
        assert nh >= self.height and nw >= self.width, f"internal error. small scale {scale}, {width}*{height}"
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        face_cx = int(face_cx * scale + 0.5)
        face_cy = int(face_cy * scale + 0.5)
        height, width = nh, nw

        # 顔を中心として448*640とかへ切り出す
        for axis, (target_size, length, face_p) in enumerate(zip((self.height, self.width), (height, width), (face_cy, face_cx))):
            p1 = face_p - target_size // 2  # 顔を中心に持ってくるための切り出し位置

            if subset.random_crop:
                # 背景も含めるために顔を中心に置く確率を高めつつずらす
                range = max(length - face_p, face_p)  # 画像の端から顔中心までの距離の長いほう
                p1 = p1 + (random.randint(0, range) + random.randint(0, range)) - range  # -range ~ +range までのいい感じの乱数
            else:
                # range指定があるときのみ、すこしだけランダムに（わりと適当）
                if subset.face_crop_aug_range[0] != subset.face_crop_aug_range[1]:
                    if face_size > size // 10 and face_size >= 40:
                        p1 = p1 + random.randint(-face_size // 20, +face_size // 20)

            p1 = max(0, min(p1, length - target_size))

            if axis == 0:
                image = image[p1 : p1 + target_size, :]
            else:
                image = image[:, p1 : p1 + target_size]

        return image

    def load_latents_from_npz(self, image_info: ImageInfo, flipped):
        npz_file = image_info.latents_npz_flipped if flipped else image_info.latents_npz
        if npz_file is None:
            return None
        return np.load(npz_file)["arr_0"]

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        bucket = self.bucket_manager.buckets[self.buckets_indices[index].bucket_index]
        bucket_batch_size = self.buckets_indices[index].bucket_batch_size
        image_index = self.buckets_indices[index].batch_index * bucket_batch_size

        loss_weights = []
        captions = []
        input_ids_list = []
        latents_list = []
        images = []

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]
            loss_weights.append(self.prior_loss_weight if image_info.is_reg else 1.0)

            # image/latentsを処理する
            if image_info.latents is not None:  # cache_latents=Trueの場合
                latents = image_info.latents if not subset.flip_aug or random.random() < 0.5 else image_info.latents_flipped
                image = None
            elif image_info.latents_npz is not None:  # FineTuningDatasetまたはcache_latents_to_disk=Trueの場合
                latents = self.load_latents_from_npz(image_info, subset.flip_aug and random.random() >= 0.5)
                latents = torch.FloatTensor(latents)
                image = None
            else:
                # 画像を読み込み、必要ならcropする
                img, face_cx, face_cy, face_w, face_h = self.load_image_with_face_info(subset, image_info.absolute_path)
                im_h, im_w = img.shape[0:2]

                if self.enable_bucket:
                    img = self.trim_and_resize_if_required(subset, img, image_info.bucket_reso, image_info.resized_size)
                else:
                    if face_cx > 0:  # 顔位置情報あり
                        img = self.crop_target(subset, img, face_cx, face_cy, face_w, face_h)
                    elif im_h > self.height or im_w > self.width:
                        assert (
                            subset.random_crop
                        ), f"image too large, but cropping and bucketing are disabled / 画像サイズが大きいのでface_crop_aug_rangeかrandom_crop、またはbucketを有効にしてください: {image_info.absolute_path}"
                        if im_h > self.height:
                            p = random.randint(0, im_h - self.height)
                            img = img[p : p + self.height]
                        if im_w > self.width:
                            p = random.randint(0, im_w - self.width)
                            img = img[:, p : p + self.width]

                    im_h, im_w = img.shape[0:2]
                    assert (
                        im_h == self.height and im_w == self.width
                    ), f"image size is small / 画像サイズが小さいようです: {image_info.absolute_path}"

                # augmentation
                aug = self.aug_helper.get_augmentor(subset.color_aug, subset.flip_aug)
                if aug is not None:
                    img = aug(image=img)["image"]

                latents = None
                image = self.image_transforms(img)  # -1.0~1.0のtorch.Tensorになる

            images.append(image)
            latents_list.append(latents)

            caption = self.process_caption(subset, image_info.caption)
            if self.XTI_layers:
                caption_layer = []
                for layer in self.XTI_layers:
                    token_strings_from = " ".join(self.token_strings)
                    token_strings_to = " ".join([f"{x}_{layer}" for x in self.token_strings])
                    caption_ = caption.replace(token_strings_from, token_strings_to)
                    caption_layer.append(caption_)
                captions.append(caption_layer)
            else:
                captions.append(caption)
            if not self.token_padding_disabled:  # this option might be omitted in future
                if self.XTI_layers:
                    token_caption = self.get_input_ids(caption_layer)
                else:
                    token_caption = self.get_input_ids(caption)
                input_ids_list.append(token_caption)

        example = {}
        example["loss_weights"] = torch.FloatTensor(loss_weights)

        if self.token_padding_disabled:
            # padding=True means pad in the batch
            example["input_ids"] = self.tokenizer(captions, padding=True, truncation=True, return_tensors="pt").input_ids
        else:
            # batch processing seems to be good
            example["input_ids"] = torch.stack(input_ids_list)

        if images[0] is not None:
            images = torch.stack(images)
            images = images.to(memory_format=torch.contiguous_format).float()
        else:
            images = None
        example["images"] = images

        example["latents"] = torch.stack(latents_list) if latents_list[0] is not None else None
        example["captions"] = captions

        if self.debug_dataset:
            example["image_keys"] = bucket[image_index : image_index + self.batch_size]
        return example


class DreamBoothDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[DreamBoothSubset],
        batch_size: int,
        tokenizer,
        max_token_length,
        resolution,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        prior_loss_weight: float,
        debug_dataset,
    ) -> None:
        super().__init__(tokenizer, max_token_length, resolution, debug_dataset)

        assert resolution is not None, f"resolution is required / resolution（解像度）指定は必須です"

        self.batch_size = batch_size
        self.size = min(self.width, self.height)  # 短いほう
        self.prior_loss_weight = prior_loss_weight
        self.latents_cache = None

        self.enable_bucket = enable_bucket
        if self.enable_bucket:
            assert (
                min(resolution) >= min_bucket_reso
            ), f"min_bucket_reso must be equal or less than resolution / min_bucket_resoは最小解像度より大きくできません。解像度を大きくするかmin_bucket_resoを小さくしてください"
            assert (
                max(resolution) <= max_bucket_reso
            ), f"max_bucket_reso must be equal or greater than resolution / max_bucket_resoは最大解像度より小さくできません。解像度を小さくするかmin_bucket_resoを大きくしてください"
            self.min_bucket_reso = min_bucket_reso
            self.max_bucket_reso = max_bucket_reso
            self.bucket_reso_steps = bucket_reso_steps
            self.bucket_no_upscale = bucket_no_upscale
        else:
            self.min_bucket_reso = None
            self.max_bucket_reso = None
            self.bucket_reso_steps = None  # この情報は使われない
            self.bucket_no_upscale = False

        def read_caption(img_path, caption_extension):
            # captionの候補ファイル名を作る
            base_name = os.path.splitext(img_path)[0]
            base_name_face_det = base_name
            tokens = base_name.split("_")
            if len(tokens) >= 5:
                base_name_face_det = "_".join(tokens[:-4])
            cap_paths = [base_name + caption_extension, base_name_face_det + caption_extension]

            caption = None
            for cap_path in cap_paths:
                if os.path.isfile(cap_path):
                    with open(cap_path, "rt", encoding="utf-8") as f:
                        try:
                            lines = f.readlines()
                        except UnicodeDecodeError as e:
                            print(f"illegal char in file (not UTF-8) / ファイルにUTF-8以外の文字があります: {cap_path}")
                            raise e
                        assert len(lines) > 0, f"caption file is empty / キャプションファイルが空です: {cap_path}"
                        caption = lines[0].strip()
                    break
            return caption

        def load_dreambooth_dir(subset: DreamBoothSubset):
            if not os.path.isdir(subset.image_dir):
                print(f"not directory: {subset.image_dir}")
                return [], []

            img_paths = glob_images(subset.image_dir, "*")
            print(f"found directory {subset.image_dir} contains {len(img_paths)} image files")

            # 画像ファイルごとにプロンプトを読み込み、もしあればそちらを使う
            captions = []
            for img_path in img_paths:
                cap_for_img = read_caption(img_path, subset.caption_extension)
                if cap_for_img is None and subset.class_tokens is None:
                    print(f"neither caption file nor class tokens are found. use empty caption for {img_path}")
                    captions.append("")
                else:
                    captions.append(subset.class_tokens if cap_for_img is None else cap_for_img)

            self.set_tag_frequency(os.path.basename(subset.image_dir), captions)  # タグ頻度を記録

            return img_paths, captions

        print("prepare images.")
        num_train_images = 0
        num_reg_images = 0
        reg_infos: List[ImageInfo] = []
        for subset in subsets:
            if subset.num_repeats < 1:
                print(
                    f"ignore subset with image_dir='{subset.image_dir}': num_repeats is less than 1 / num_repeatsが1を下回っているためサブセットを無視します: {subset.num_repeats}"
                )
                continue

            if subset in self.subsets:
                print(
                    f"ignore duplicated subset with image_dir='{subset.image_dir}': use the first one / 既にサブセットが登録されているため、重複した後発のサブセットを無視します"
                )
                continue

            img_paths, captions = load_dreambooth_dir(subset)
            if len(img_paths) < 1:
                print(f"ignore subset with image_dir='{subset.image_dir}': no images found / 画像が見つからないためサブセットを無視します")
                continue

            if subset.is_reg:
                num_reg_images += subset.num_repeats * len(img_paths)
            else:
                num_train_images += subset.num_repeats * len(img_paths)

            for img_path, caption in zip(img_paths, captions):
                info = ImageInfo(img_path, subset.num_repeats, caption, subset.is_reg, img_path)
                if subset.is_reg:
                    reg_infos.append(info)
                else:
                    self.register_image(info, subset)

            subset.img_count = len(img_paths)
            self.subsets.append(subset)

        print(f"{num_train_images} train images with repeating.")
        self.num_train_images = num_train_images

        print(f"{num_reg_images} reg images.")
        if num_train_images < num_reg_images:
            print("some of reg images are not used / 正則化画像の数が多いので、一部使用されない正則化画像があります")

        if num_reg_images == 0:
            print("no regularization images / 正則化画像が見つかりませんでした")
        else:
            # num_repeatsを計算する：どうせ大した数ではないのでループで処理する
            n = 0
            first_loop = True
            while n < num_train_images:
                for info in reg_infos:
                    if first_loop:
                        self.register_image(info, subset)
                        n += info.num_repeats
                    else:
                        info.num_repeats += 1  # rewrite registered info
                        n += 1
                    if n >= num_train_images:
                        break
                first_loop = False

        self.num_reg_images = num_reg_images


class FineTuningDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[FineTuningSubset],
        batch_size: int,
        tokenizer,
        max_token_length,
        resolution,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        debug_dataset,
    ) -> None:
        super().__init__(tokenizer, max_token_length, resolution, debug_dataset)

        self.batch_size = batch_size

        self.num_train_images = 0
        self.num_reg_images = 0

        for subset in subsets:
            if subset.num_repeats < 1:
                print(
                    f"ignore subset with metadata_file='{subset.metadata_file}': num_repeats is less than 1 / num_repeatsが1を下回っているためサブセットを無視します: {subset.num_repeats}"
                )
                continue

            if subset in self.subsets:
                print(
                    f"ignore duplicated subset with metadata_file='{subset.metadata_file}': use the first one / 既にサブセットが登録されているため、重複した後発のサブセットを無視します"
                )
                continue

            # メタデータを読み込む
            if os.path.exists(subset.metadata_file):
                print(f"loading existing metadata: {subset.metadata_file}")
                with open(subset.metadata_file, "rt", encoding="utf-8") as f:
                    metadata = json.load(f)
            else:
                raise ValueError(f"no metadata / メタデータファイルがありません: {subset.metadata_file}")

            if len(metadata) < 1:
                print(f"ignore subset with '{subset.metadata_file}': no image entries found / 画像に関するデータが見つからないためサブセットを無視します")
                continue

            tags_list = []
            for image_key, img_md in metadata.items():
                # path情報を作る
                abs_path = None

                # まず画像を優先して探す
                if os.path.exists(image_key):
                    abs_path = image_key
                else:
                    # わりといい加減だがいい方法が思いつかん
                    paths = glob_images(subset.image_dir, image_key)
                    if len(paths) > 0:
                        abs_path = paths[0]

                # なければnpzを探す
                if abs_path is None:
                    if os.path.exists(os.path.splitext(image_key)[0] + ".npz"):
                        abs_path = os.path.splitext(image_key)[0] + ".npz"
                    else:
                        npz_path = os.path.join(subset.image_dir, image_key + ".npz")
                        if os.path.exists(npz_path):
                            abs_path = npz_path

                assert abs_path is not None, f"no image / 画像がありません: {image_key}"

                caption = img_md.get("caption")
                tags = img_md.get("tags")
                if caption is None:
                    caption = tags
                elif tags is not None and len(tags) > 0:
                    caption = caption + ", " + tags
                    tags_list.append(tags)

                if caption is None:
                    caption = ""

                image_info = ImageInfo(image_key, subset.num_repeats, caption, False, abs_path)
                image_info.image_size = img_md.get("train_resolution")

                if not subset.color_aug and not subset.random_crop:
                    # if npz exists, use them
                    image_info.latents_npz, image_info.latents_npz_flipped = self.image_key_to_npz_file(subset, image_key)

                self.register_image(image_info, subset)

            self.num_train_images += len(metadata) * subset.num_repeats

            # TODO do not record tag freq when no tag
            self.set_tag_frequency(os.path.basename(subset.metadata_file), tags_list)
            subset.img_count = len(metadata)
            self.subsets.append(subset)

        # check existence of all npz files
        use_npz_latents = all([not (subset.color_aug or subset.random_crop) for subset in self.subsets])
        if use_npz_latents:
            flip_aug_in_subset = False
            npz_any = False
            npz_all = True

            for image_info in self.image_data.values():
                subset = self.image_to_subset[image_info.image_key]

                has_npz = image_info.latents_npz is not None
                npz_any = npz_any or has_npz

                if subset.flip_aug:
                    has_npz = has_npz and image_info.latents_npz_flipped is not None
                    flip_aug_in_subset = True
                npz_all = npz_all and has_npz

                if npz_any and not npz_all:
                    break

            if not npz_any:
                use_npz_latents = False
                print(f"npz file does not exist. ignore npz files / npzファイルが見つからないためnpzファイルを無視します")
            elif not npz_all:
                use_npz_latents = False
                print(f"some of npz file does not exist. ignore npz files / いくつかのnpzファイルが見つからないためnpzファイルを無視します")
                if flip_aug_in_subset:
                    print("maybe no flipped files / 反転されたnpzファイルがないのかもしれません")
        # else:
        #   print("npz files are not used with color_aug and/or random_crop / color_augまたはrandom_cropが指定されているためnpzファイルは使用されません")

        # check min/max bucket size
        sizes = set()
        resos = set()
        for image_info in self.image_data.values():
            if image_info.image_size is None:
                sizes = None  # not calculated
                break
            sizes.add(image_info.image_size[0])
            sizes.add(image_info.image_size[1])
            resos.add(tuple(image_info.image_size))

        if sizes is None:
            if use_npz_latents:
                use_npz_latents = False
                print(f"npz files exist, but no bucket info in metadata. ignore npz files / メタデータにbucket情報がないためnpzファイルを無視します")

            assert (
                resolution is not None
            ), "if metadata doesn't have bucket info, resolution is required / メタデータにbucket情報がない場合はresolutionを指定してください"

            self.enable_bucket = enable_bucket
            if self.enable_bucket:
                self.min_bucket_reso = min_bucket_reso
                self.max_bucket_reso = max_bucket_reso
                self.bucket_reso_steps = bucket_reso_steps
                self.bucket_no_upscale = bucket_no_upscale
        else:
            if not enable_bucket:
                print("metadata has bucket info, enable bucketing / メタデータにbucket情報があるためbucketを有効にします")
            print("using bucket info in metadata / メタデータ内のbucket情報を使います")
            self.enable_bucket = True

            assert (
                not bucket_no_upscale
            ), "if metadata has bucket info, bucket reso is precalculated, so bucket_no_upscale cannot be used / メタデータ内にbucket情報がある場合はbucketの解像度は計算済みのため、bucket_no_upscaleは使えません"

            # bucket情報を初期化しておく、make_bucketsで再作成しない
            self.bucket_manager = BucketManager(False, None, None, None, None)
            self.bucket_manager.set_predefined_resos(resos)

        # npz情報をきれいにしておく
        if not use_npz_latents:
            for image_info in self.image_data.values():
                image_info.latents_npz = image_info.latents_npz_flipped = None

    def image_key_to_npz_file(self, subset: FineTuningSubset, image_key):
        base_name = os.path.splitext(image_key)[0]
        npz_file_norm = base_name + ".npz"

        if os.path.exists(npz_file_norm):
            # image_key is full path
            npz_file_flip = base_name + "_flip.npz"
            if not os.path.exists(npz_file_flip):
                npz_file_flip = None
            return npz_file_norm, npz_file_flip

        # if not full path, check image_dir. if image_dir is None, return None
        if subset.image_dir is None:
            return None, None

        # image_key is relative path
        npz_file_norm = os.path.join(subset.image_dir, image_key + ".npz")
        npz_file_flip = os.path.join(subset.image_dir, image_key + "_flip.npz")

        if not os.path.exists(npz_file_norm):
            npz_file_norm = None
            npz_file_flip = None
        elif not os.path.exists(npz_file_flip):
            npz_file_flip = None

        return npz_file_norm, npz_file_flip


# behave as Dataset mock
class DatasetGroup(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: Sequence[Union[DreamBoothDataset, FineTuningDataset]]):
        self.datasets: List[Union[DreamBoothDataset, FineTuningDataset]]

        super().__init__(datasets)

        self.image_data = {}
        self.num_train_images = 0
        self.num_reg_images = 0

        # simply concat together
        # TODO: handling image_data key duplication among dataset
        #   In practical, this is not the big issue because image_data is accessed from outside of dataset only for debug_dataset.
        for dataset in datasets:
            self.image_data.update(dataset.image_data)
            self.num_train_images += dataset.num_train_images
            self.num_reg_images += dataset.num_reg_images

    def add_replacement(self, str_from, str_to):
        for dataset in self.datasets:
            dataset.add_replacement(str_from, str_to)

    # def make_buckets(self):
    #   for dataset in self.datasets:
    #     dataset.make_buckets()

    def enable_XTI(self, *args, **kwargs):
        for dataset in self.datasets:
            dataset.enable_XTI(*args, **kwargs)

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True):
        for i, dataset in enumerate(self.datasets):
            print(f"[Dataset {i}]")
            dataset.cache_latents(vae, vae_batch_size, cache_to_disk, is_main_process)

    def is_latent_cacheable(self) -> bool:
        return all([dataset.is_latent_cacheable() for dataset in self.datasets])

    def set_current_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_current_epoch(epoch)

    def set_current_step(self, step):
        for dataset in self.datasets:
            dataset.set_current_step(step)

    def set_max_train_steps(self, max_train_steps):
        for dataset in self.datasets:
            dataset.set_max_train_steps(max_train_steps)

    def disable_token_padding(self):
        for dataset in self.datasets:
            dataset.disable_token_padding()


def debug_dataset(train_dataset, show_input_ids=False):
    print(f"Total dataset length (steps) / データセットの長さ（ステップ数）: {len(train_dataset)}")
    print("`S` for next step, `E` for next epoch no. , Escape for exit. / Sキーで次のステップ、Eキーで次のエポック、Escキーで中断、終了します")

    epoch = 1
    while True:
        print(f"epoch: {epoch}")

        steps = (epoch - 1) * len(train_dataset) + 1
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        k = 0
        for i, idx in enumerate(indices):
            train_dataset.set_current_epoch(epoch)
            train_dataset.set_current_step(steps)
            print(f"steps: {steps} ({i + 1}/{len(train_dataset)})")

            example = train_dataset[idx]
            if example["latents"] is not None:
                print(f"sample has latents from npz file: {example['latents'].size()}")
            for j, (ik, cap, lw, iid) in enumerate(
                zip(example["image_keys"], example["captions"], example["loss_weights"], example["input_ids"])
            ):
                print(f'{ik}, size: {train_dataset.image_data[ik].image_size}, loss weight: {lw}, caption: "{cap}"')
                if show_input_ids:
                    print(f"input ids: {iid}")
                if example["images"] is not None:
                    im = example["images"][j]
                    print(f"image size: {im.size()}")
                    im = ((im.numpy() + 1.0) * 127.5).astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))  # c,H,W -> H,W,c
                    im = im[:, :, ::-1]  # RGB -> BGR (OpenCV)
                    if os.name == "nt":  # only windows
                        cv2.imshow("img", im)
                        k = cv2.waitKey()
                        cv2.destroyAllWindows()
                    if k == 27 or k == ord("s") or k == ord("e"):
                        break
            steps += 1

            if k == ord("e"):
                break
            if k == 27 or (example["images"] is None and i >= 8):
                k = 27
                break
        if k == 27:
            break

        epoch += 1


def glob_images(directory, base="*"):
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        if base == "*":
            img_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            img_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    img_paths = list(set(img_paths))  # 重複を排除
    img_paths.sort()
    return img_paths


def glob_images_pathlib(dir_path, recursive):
    image_paths = []
    if recursive:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.rglob("*" + ext))
    else:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.glob("*" + ext))
    image_paths = list(set(image_paths))  # 重複を排除
    image_paths.sort()
    return image_paths


# endregion

# region モジュール入れ替え部
"""
高速化のためのモジュール入れ替え
"""

# FlashAttentionを使うCrossAttention
# based on https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/memory_efficient_attention_pytorch/flash_attention.py
# LICENSE MIT https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/LICENSE

# constants

EPSILON = 1e-6

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def model_hash(filename):
    """Old model hash used by stable-diffusion-webui"""
    try:
        with open(filename, "rb") as file:
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def calculate_sha256(filename):
    """New model hash used by stable-diffusion-webui"""
    try:
        hash_sha256 = hashlib.sha256()
        blksize = 1024 * 1024

        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(blksize), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def precalculate_safetensors_hashes(tensors, metadata):
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later."""

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash


def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode("ascii").strip()
    except:
        return "(unknown)"


# flash attention forwards and backwards

# https://arxiv.org/abs/2205.14135


class FlashAttentionFunction(torch.autograd.function.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, q, k, v, mask, causal, q_bucket_size, k_bucket_size):
        """Algorithm 2 in the paper"""

        device = q.device
        dtype = q.dtype
        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        o = torch.zeros_like(q)
        all_row_sums = torch.zeros((*q.shape[:-1], 1), dtype=dtype, device=device)
        all_row_maxes = torch.full((*q.shape[:-1], 1), max_neg_value, dtype=dtype, device=device)

        scale = q.shape[-1] ** -0.5

        if not exists(mask):
            mask = (None,) * math.ceil(q.shape[-2] / q_bucket_size)
        else:
            mask = rearrange(mask, "b n -> b 1 1 n")
            mask = mask.split(q_bucket_size, dim=-1)

        row_splits = zip(
            q.split(q_bucket_size, dim=-2),
            o.split(q_bucket_size, dim=-2),
            mask,
            all_row_sums.split(q_bucket_size, dim=-2),
            all_row_maxes.split(q_bucket_size, dim=-2),
        )

        for ind, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim=-2),
                v.split(k_bucket_size, dim=-2),
            )

            for k_ind, (kc, vc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum("... i d, ... j d -> ... i j", qc, kc) * scale

                if exists(row_mask):
                    attn_weights.masked_fill_(~row_mask, max_neg_value)

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool, device=device).triu(
                        q_start_index - k_start_index + 1
                    )
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                block_row_maxes = attn_weights.amax(dim=-1, keepdims=True)
                attn_weights -= block_row_maxes
                exp_weights = torch.exp(attn_weights)

                if exists(row_mask):
                    exp_weights.masked_fill_(~row_mask, 0.0)

                block_row_sums = exp_weights.sum(dim=-1, keepdims=True).clamp(min=EPSILON)

                new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

                exp_values = einsum("... i j, ... j d -> ... i d", exp_weights, vc)

                exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)
                exp_block_row_max_diff = torch.exp(block_row_maxes - new_row_maxes)

                new_row_sums = exp_row_max_diff * row_sums + exp_block_row_max_diff * block_row_sums

                oc.mul_((row_sums / new_row_sums) * exp_row_max_diff).add_((exp_block_row_max_diff / new_row_sums) * exp_values)

                row_maxes.copy_(new_row_maxes)
                row_sums.copy_(new_row_sums)

        ctx.args = (causal, scale, mask, q_bucket_size, k_bucket_size)
        ctx.save_for_backward(q, k, v, o, all_row_sums, all_row_maxes)

        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        """Algorithm 4 in the paper"""

        causal, scale, mask, q_bucket_size, k_bucket_size = ctx.args
        q, k, v, o, l, m = ctx.saved_tensors

        device = q.device

        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        row_splits = zip(
            q.split(q_bucket_size, dim=-2),
            o.split(q_bucket_size, dim=-2),
            do.split(q_bucket_size, dim=-2),
            mask,
            l.split(q_bucket_size, dim=-2),
            m.split(q_bucket_size, dim=-2),
            dq.split(q_bucket_size, dim=-2),
        )

        for ind, (qc, oc, doc, row_mask, lc, mc, dqc) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim=-2),
                v.split(k_bucket_size, dim=-2),
                dk.split(k_bucket_size, dim=-2),
                dv.split(k_bucket_size, dim=-2),
            )

            for k_ind, (kc, vc, dkc, dvc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum("... i d, ... j d -> ... i j", qc, kc) * scale

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool, device=device).triu(
                        q_start_index - k_start_index + 1
                    )
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                exp_attn_weights = torch.exp(attn_weights - mc)

                if exists(row_mask):
                    exp_attn_weights.masked_fill_(~row_mask, 0.0)

                p = exp_attn_weights / lc

                dv_chunk = einsum("... i j, ... i d -> ... j d", p, doc)
                dp = einsum("... i d, ... j d -> ... i j", doc, vc)

                D = (doc * oc).sum(dim=-1, keepdims=True)
                ds = p * scale * (dp - D)

                dq_chunk = einsum("... i j, ... j d -> ... i d", ds, kc)
                dk_chunk = einsum("... i j, ... i d -> ... j d", ds, qc)

                dqc.add_(dq_chunk)
                dkc.add_(dk_chunk)
                dvc.add_(dv_chunk)

        return dq, dk, dv, None, None, None, None


def replace_unet_modules(unet: diffusers.models.unet_2d_condition.UNet2DConditionModel, mem_eff_attn, xformers):
    if mem_eff_attn:
        replace_unet_cross_attn_to_memory_efficient()
    elif xformers:
        replace_unet_cross_attn_to_xformers()


def replace_unet_cross_attn_to_memory_efficient():
    print("Replace CrossAttention.forward to use FlashAttention (not xformers)")
    flash_func = FlashAttentionFunction

    def forward_flash_attn(self, x, context=None, mask=None):
        q_bucket_size = 512
        k_bucket_size = 1024

        h = self.heads
        q = self.to_q(x)

        context = context if context is not None else x
        context = context.to(x.dtype)

        if hasattr(self, "hypernetwork") and self.hypernetwork is not None:
            context_k, context_v = self.hypernetwork.forward(x, context)
            context_k = context_k.to(x.dtype)
            context_v = context_v.to(x.dtype)
        else:
            context_k = context
            context_v = context

        k = self.to_k(context_k)
        v = self.to_v(context_v)
        del context, x

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        out = flash_func.apply(q, k, v, mask, False, q_bucket_size, k_bucket_size)

        out = rearrange(out, "b h n d -> b n (h d)")

        # diffusers 0.7.0~  わざわざ変えるなよ (;´Д｀)
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out

    diffusers.models.attention.CrossAttention.forward = forward_flash_attn


def replace_unet_cross_attn_to_xformers():
    print("Replace CrossAttention.forward to use xformers")
    try:
        import xformers.ops
    except ImportError:
        raise ImportError("No xformers / xformersがインストールされていないようです")

    def forward_xformers(self, x, context=None, mask=None):
        h = self.heads
        q_in = self.to_q(x)

        context = default(context, x)
        context = context.to(x.dtype)

        if hasattr(self, "hypernetwork") and self.hypernetwork is not None:
            context_k, context_v = self.hypernetwork.forward(x, context)
            context_k = context_k.to(x.dtype)
            context_v = context_v.to(x.dtype)
        else:
            context_k = context
            context_v = context

        k_in = self.to_k(context_k)
        v_in = self.to_v(context_v)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)  # 最適なのを選んでくれる

        out = rearrange(out, "b n h d -> b n (h d)", h=h)

        # diffusers 0.7.0~
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out

    diffusers.models.attention.CrossAttention.forward = forward_xformers


# endregion


# region arguments


def add_sd_models_arguments(parser: argparse.ArgumentParser):
    # for pretrained models
    parser.add_argument("--v2", action="store_true", help="load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む")
    parser.add_argument(
        "--v_parameterization", action="store_true", help="enable v-parameterization training / v-parameterization学習を有効にする"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="pretrained model to train, directory to Diffusers model or StableDiffusion checkpoint / 学習元モデル、Diffusers形式モデルのディレクトリまたはStableDiffusionのckptファイル",
    )
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default=None,
        help="directory for caching Tokenizer (for offline training) / Tokenizerをキャッシュするディレクトリ（ネット接続なしでの学習のため）",
    )


def add_optimizer_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="",
        help="Optimizer to use / オプティマイザの種類: AdamW (default), AdamW8bit, Lion8bit, Lion, SGDNesterov, SGDNesterov8bit, DAdaptation(DAdaptAdam), DAdaptAdaGrad, DAdaptAdan, DAdaptSGD, AdaFactor",
    )

    # backward compatibility
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="use 8bit AdamW optimizer (requires bitsandbytes) / 8bit Adamオプティマイザを使う（bitsandbytesのインストールが必要）",
    )
    parser.add_argument(
        "--use_lion_optimizer",
        action="store_true",
        help="use Lion optimizer (requires lion-pytorch) / Lionオプティマイザを使う（ lion-pytorch のインストールが必要）",
    )

    parser.add_argument("--learning_rate", type=float, default=2.0e-6, help="learning rate / 学習率")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm, 0 for no clipping / 勾配正規化の最大norm、0でclippingを行わない"
    )

    parser.add_argument(
        "--optimizer_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for optimizer (like "weight_decay=0.01 betas=0.9,0.999 ...") / オプティマイザの追加引数（例： "weight_decay=0.01 betas=0.9,0.999 ..."）',
    )

    parser.add_argument("--lr_scheduler_type", type=str, default="", help="custom scheduler module / 使用するスケジューラ")
    parser.add_argument(
        "--lr_scheduler_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for scheduler (like "T_max=100") / スケジューラの追加引数（例： "T_max100"）',
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="scheduler to use for learning rate / 学習率のスケジューラ: linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup, adafactor",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler (default is 0) / 学習率のスケジューラをウォームアップするステップ数（デフォルト0）",
    )
    parser.add_argument(
        "--lr_scheduler_num_cycles",
        type=int,
        default=1,
        help="Number of restarts for cosine scheduler with restarts / cosine with restartsスケジューラでのリスタート回数",
    )
    parser.add_argument(
        "--lr_scheduler_power",
        type=float,
        default=1,
        help="Polynomial power for polynomial scheduler / polynomialスケジューラでのpolynomial power",
    )


def add_training_arguments(parser: argparse.ArgumentParser, support_dreambooth: bool):
    parser.add_argument("--output_dir", type=str, default=None, help="directory to output trained model / 学習後のモデル出力先ディレクトリ")
    parser.add_argument("--output_name", type=str, default=None, help="base name of trained model file / 学習後のモデルの拡張子を除くファイル名")
    parser.add_argument(
        "--huggingface_repo_id", type=str, default=None, help="huggingface repo name to upload / huggingfaceにアップロードするリポジトリ名"
    )
    parser.add_argument(
        "--huggingface_repo_type", type=str, default=None, help="huggingface repo type to upload / huggingfaceにアップロードするリポジトリの種類"
    )
    parser.add_argument(
        "--huggingface_path_in_repo",
        type=str,
        default=None,
        help="huggingface model path to upload files / huggingfaceにアップロードするファイルのパス",
    )
    parser.add_argument("--huggingface_token", type=str, default=None, help="huggingface token / huggingfaceのトークン")
    parser.add_argument(
        "--huggingface_repo_visibility",
        type=str,
        default=None,
        help="huggingface repository visibility ('public' for public, 'private' or None for private) / huggingfaceにアップロードするリポジトリの公開設定（'public'で公開、'private'またはNoneで非公開）",
    )
    parser.add_argument(
        "--save_state_to_huggingface", action="store_true", help="save state to huggingface / huggingfaceにstateを保存する"
    )
    parser.add_argument(
        "--resume_from_huggingface",
        action="store_true",
        help="resume from huggingface (ex: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type}) / huggingfaceから学習を再開する(例: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type})",
    )
    parser.add_argument(
        "--async_upload",
        action="store_true",
        help="upload to huggingface asynchronously / huggingfaceに非同期でアップロードする",
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving / 保存時に精度を変更して保存する",
    )
    parser.add_argument(
        "--save_every_n_epochs", type=int, default=None, help="save checkpoint every N epochs / 学習中のモデルを指定エポックごとに保存する"
    )
    parser.add_argument(
        "--save_every_n_steps", type=int, default=None, help="save checkpoint every N steps / 学習中のモデルを指定ステップごとに保存する"
    )
    parser.add_argument(
        "--save_n_epoch_ratio",
        type=int,
        default=None,
        help="save checkpoint N epoch ratio (for example 5 means save at least 5 files total) / 学習中のモデルを指定のエポック割合で保存する（たとえば5を指定すると最低5個のファイルが保存される）",
    )
    parser.add_argument(
        "--save_last_n_epochs",
        type=int,
        default=None,
        help="save last N checkpoints when saving every N epochs (remove older checkpoints) / 指定エポックごとにモデルを保存するとき最大Nエポック保存する（古いチェックポイントは削除する）",
    )
    parser.add_argument(
        "--save_last_n_epochs_state",
        type=int,
        default=None,
        help="save last N checkpoints of state (overrides the value of --save_last_n_epochs)/ 最大Nエポックstateを保存する（--save_last_n_epochsの指定を上書きする）",
    )
    parser.add_argument(
        "--save_last_n_steps",
        type=int,
        default=None,
        help="save checkpoints until N steps elapsed (remove older checkpoints if N steps elapsed) / 指定ステップごとにモデルを保存するとき、このステップ数経過するまで保存する（このステップ数経過したら削除する）",
    )
    parser.add_argument(
        "--save_last_n_steps_state",
        type=int,
        default=None,
        help="save states until N steps elapsed (remove older states if N steps elapsed, overrides --save_last_n_steps) / 指定ステップごとにstateを保存するとき、このステップ数経過するまで保存する（このステップ数経過したら削除する。--save_last_n_stepsを上書きする）",
    )
    parser.add_argument(
        "--save_state",
        action="store_true",
        help="save training state additionally (including optimizer states etc.) / optimizerなど学習状態も含めたstateを追加で保存する",
    )
    parser.add_argument("--resume", type=str, default=None, help="saved state to resume training / 学習再開するモデルのstate")

    parser.add_argument("--train_batch_size", type=int, default=1, help="batch size for training / 学習時のバッチサイズ")
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=None,
        choices=[None, 150, 225],
        help="max token length of text encoder (default for 75, 150 or 225) / text encoderのトークンの最大長（未指定で75、150または225が指定可）",
    )
    parser.add_argument(
        "--mem_eff_attn",
        action="store_true",
        help="use memory efficient attention for CrossAttention / CrossAttentionに省メモリ版attentionを使う",
    )
    parser.add_argument("--xformers", action="store_true", help="use xformers for CrossAttention / CrossAttentionにxformersを使う")
    parser.add_argument(
        "--vae", type=str, default=None, help="path to checkpoint of vae to replace / VAEを入れ替える場合、VAEのcheckpointファイルまたはディレクトリ"
    )

    parser.add_argument("--max_train_steps", type=int, default=1600, help="training steps / 学習ステップ数")
    parser.add_argument(
        "--max_train_epochs",
        type=int,
        default=None,
        help="training epochs (overrides max_train_steps) / 学習エポック数（max_train_stepsを上書きします）",
    )
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=8,
        help="max num workers for DataLoader (lower is less main RAM usage, faster epoch start and slower data loading) / DataLoaderの最大プロセス数（小さい値ではメインメモリの使用量が減りエポック間の待ち時間が減りますが、データ読み込みは遅くなります）",
    )
    parser.add_argument(
        "--persistent_data_loader_workers",
        action="store_true",
        help="persistent DataLoader workers (useful for reduce time gap between epoch, but may use more memory) / DataLoader のワーカーを持続させる (エポック間の時間差を少なくするのに有効だが、より多くのメモリを消費する可能性がある)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for training / 学習時の乱数のseed")
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", help="enable gradient checkpointing / grandient checkpointingを有効にする"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass / 学習時に逆伝播をする前に勾配を合計するステップ数",
    )
    parser.add_argument(
        "--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="use mixed precision / 混合精度を使う場合、その精度"
    )
    parser.add_argument("--full_fp16", action="store_true", help="fp16 training including gradients / 勾配も含めてfp16で学習する")
    parser.add_argument(
        "--clip_skip",
        type=int,
        default=None,
        help="use output of nth layer from back of text encoder (n>=1) / text encoderの後ろからn番目の層の出力を用いる（nは1以上）",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="enable logging and output TensorBoard log to this directory / ログ出力を有効にしてこのディレクトリにTensorBoard用のログを出力する",
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default=None,
        choices=["tensorboard", "wandb", "all"],
        help="what logging tool(s) to use (if 'all', TensorBoard and WandB are both used) / ログ出力に使用するツール (allを指定するとTensorBoardとWandBの両方が使用される)",
    )
    parser.add_argument("--log_prefix", type=str, default=None, help="add prefix for each log directory / ログディレクトリ名の先頭に追加する文字列")
    parser.add_argument(
        "--log_tracker_name",
        type=str,
        default=None,
        help="name of tracker to use for logging, default is script-specific default name / ログ出力に使用するtrackerの名前、省略時はスクリプトごとのデフォルト名",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="specify WandB API key to log in before starting training (optional). / WandB APIキーを指定して学習開始前にログインする（オプション）",
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=None,
        help="enable noise offset with this value (if enabled, around 0.1 is recommended) / Noise offsetを有効にしてこの値を設定する（有効にする場合は0.1程度を推奨）",
    )
    parser.add_argument(
        "--multires_noise_iterations",
        type=int,
        default=None,
        help="enable multires noise with this number of iterations (if enabled, around 6-10 is recommended) / Multires noiseを有効にしてこのイテレーション数を設定する（有効にする場合は6-10程度を推奨）",
    )
    parser.add_argument(
        "--multires_noise_discount",
        type=float,
        default=0.3,
        help="set discount value for multires noise (has no effect without --multires_noise_iterations) / Multires noiseのdiscount値を設定する（--multires_noise_iterations指定時のみ有効）",
    )
    parser.add_argument(
        "--adaptive_noise_scale",
        type=float,
        default=None,
        help="add `latent mean absolute value * this value` to noise_offset (disabled if None, default) / latentの平均値の絶対値 * この値をnoise_offsetに加算する（Noneの場合は無効、デフォルト）",
    )
    parser.add_argument(
        "--lowram",
        action="store_true",
        help="enable low RAM optimization. e.g. load models to VRAM instead of RAM (for machines which have bigger VRAM than RAM such as Colab and Kaggle) / メインメモリが少ない環境向け最適化を有効にする。たとえばVRAMにモデルを読み込むなど（ColabやKaggleなどRAMに比べてVRAMが多い環境向け）",
    )

    parser.add_argument(
        "--sample_every_n_steps", type=int, default=None, help="generate sample images every N steps / 学習中のモデルで指定ステップごとにサンプル出力する"
    )
    parser.add_argument(
        "--sample_every_n_epochs",
        type=int,
        default=None,
        help="generate sample images every N epochs (overwrites n_steps) / 学習中のモデルで指定エポックごとにサンプル出力する（ステップ数指定を上書きします）",
    )
    parser.add_argument(
        "--sample_prompts", type=str, default=None, help="file for prompts to generate sample images / 学習中モデルのサンプル出力用プロンプトのファイル"
    )
    parser.add_argument(
        "--sample_sampler",
        type=str,
        default="ddim",
        choices=[
            "ddim",
            "pndm",
            "lms",
            "euler",
            "euler_a",
            "heun",
            "dpm_2",
            "dpm_2_a",
            "dpmsolver",
            "dpmsolver++",
            "dpmsingle",
            "k_lms",
            "k_euler",
            "k_euler_a",
            "k_dpm_2",
            "k_dpm_2_a",
        ],
        help=f"sampler (scheduler) type for sample images / サンプル出力時のサンプラー（スケジューラ）の種類",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="using .toml instead of args to pass hyperparameter / ハイパーパラメータを引数ではなく.tomlファイルで渡す",
    )
    parser.add_argument(
        "--output_config", action="store_true", help="output command line args to given .toml file / 引数を.tomlファイルに出力する"
    )

    if support_dreambooth:
        # DreamBooth training
        parser.add_argument(
            "--prior_loss_weight", type=float, default=1.0, help="loss weight for regularization images / 正則化画像のlossの重み"
        )


def verify_training_args(args: argparse.Namespace):
    if args.v_parameterization and not args.v2:
        print("v_parameterization should be with v2 / v1でv_parameterizationを使用することは想定されていません")
    if args.v2 and args.clip_skip is not None:
        print("v2 with clip_skip will be unexpected / v2でclip_skipを使用することは想定されていません")

    if args.cache_latents_to_disk and not args.cache_latents:
        args.cache_latents = True
        print(
            "cache_latents_to_disk is enabled, so cache_latents is also enabled / cache_latents_to_diskが有効なため、cache_latentsを有効にします"
        )

    if args.noise_offset is not None and args.multires_noise_iterations is not None:
        raise ValueError(
            "noise_offset and multires_noise_iterations cannot be enabled at the same time / noise_offsetとmultires_noise_iterationsを同時に有効にすることはできません"
        )

    if args.adaptive_noise_scale is not None and args.noise_offset is None:
        raise ValueError(
            "adaptive_noise_scale requires noise_offset / adaptive_noise_scaleを使用するにはnoise_offsetが必要です"
        )


def add_dataset_arguments(
    parser: argparse.ArgumentParser, support_dreambooth: bool, support_caption: bool, support_caption_dropout: bool
):
    # dataset common
    parser.add_argument("--train_data_dir", type=str, default=None, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument(
        "--shuffle_caption", action="store_true", help="shuffle comma-separated caption / コンマで区切られたcaptionの各要素をshuffleする"
    )
    parser.add_argument(
        "--caption_extension", type=str, default=".caption", help="extension of caption files / 読み込むcaptionファイルの拡張子"
    )
    parser.add_argument(
        "--caption_extention",
        type=str,
        default=None,
        help="extension of caption files (backward compatibility) / 読み込むcaptionファイルの拡張子（スペルミスを残してあります）",
    )
    parser.add_argument(
        "--keep_tokens",
        type=int,
        default=0,
        help="keep heading N tokens when shuffling caption tokens (token means comma separated strings) / captionのシャッフル時に、先頭からこの個数のトークンをシャッフルしないで残す（トークンはカンマ区切りの各部分を意味する）",
    )
    parser.add_argument("--color_aug", action="store_true", help="enable weak color augmentation / 学習時に色合いのaugmentationを有効にする")
    parser.add_argument("--flip_aug", action="store_true", help="enable horizontal flip augmentation / 学習時に左右反転のaugmentationを有効にする")
    parser.add_argument(
        "--face_crop_aug_range",
        type=str,
        default=None,
        help="enable face-centered crop augmentation and its range (e.g. 2.0,4.0) / 学習時に顔を中心とした切り出しaugmentationを有効にするときは倍率を指定する（例：2.0,4.0）",
    )
    parser.add_argument(
        "--random_crop",
        action="store_true",
        help="enable random crop (for style training in face-centered crop augmentation) / ランダムな切り出しを有効にする（顔を中心としたaugmentationを行うときに画風の学習用に指定する）",
    )
    parser.add_argument(
        "--debug_dataset", action="store_true", help="show images for debugging (do not train) / デバッグ用に学習データを画面表示する（学習は行わない）"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="resolution in training ('size' or 'width,height') / 学習時の画像解像度（'サイズ'指定、または'幅,高さ'指定）",
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        help="cache latents to main memory to reduce VRAM usage (augmentations must be disabled) / VRAM削減のためにlatentをメインメモリにcacheする（augmentationは使用不可） ",
    )
    parser.add_argument("--vae_batch_size", type=int, default=1, help="batch size for caching latents / latentのcache時のバッチサイズ")
    parser.add_argument(
        "--cache_latents_to_disk",
        action="store_true",
        help="cache latents to disk to reduce VRAM usage (augmentations must be disabled) / VRAM削減のためにlatentをディスクにcacheする（augmentationは使用不可）",
    )
    parser.add_argument(
        "--enable_bucket", action="store_true", help="enable buckets for multi aspect ratio training / 複数解像度学習のためのbucketを有効にする"
    )
    parser.add_argument("--min_bucket_reso", type=int, default=256, help="minimum resolution for buckets / bucketの最小解像度")
    parser.add_argument("--max_bucket_reso", type=int, default=1024, help="maximum resolution for buckets / bucketの最大解像度")
    parser.add_argument(
        "--bucket_reso_steps",
        type=int,
        default=64,
        help="steps of resolution for buckets, divisible by 8 is recommended / bucketの解像度の単位、8で割り切れる値を推奨します",
    )
    parser.add_argument(
        "--bucket_no_upscale", action="store_true", help="make bucket for each image without upscaling / 画像を拡大せずbucketを作成します"
    )

    parser.add_argument(
        "--token_warmup_min",
        type=int,
        default=1,
        help="start learning at N tags (token means comma separated strinfloatgs) / タグ数をN個から増やしながら学習する",
    )

    parser.add_argument(
        "--token_warmup_step",
        type=float,
        default=0,
        help="tag length reaches maximum on N steps (or N*max_train_steps if N<1) / N（N<1ならN*max_train_steps）ステップでタグ長が最大になる。デフォルトは0（最初から最大）",
    )

    if support_caption_dropout:
        # Textual Inversion はcaptionのdropoutをsupportしない
        # いわゆるtensorのDropoutと紛らわしいのでprefixにcaptionを付けておく　every_n_epochsは他と平仄を合わせてdefault Noneに
        parser.add_argument(
            "--caption_dropout_rate", type=float, default=0.0, help="Rate out dropout caption(0.0~1.0) / captionをdropoutする割合"
        )
        parser.add_argument(
            "--caption_dropout_every_n_epochs",
            type=int,
            default=0,
            help="Dropout all captions every N epochs / captionを指定エポックごとにdropoutする",
        )
        parser.add_argument(
            "--caption_tag_dropout_rate",
            type=float,
            default=0.0,
            help="Rate out dropout comma separated tokens(0.0~1.0) / カンマ区切りのタグをdropoutする割合",
        )

    if support_dreambooth:
        # DreamBooth dataset
        parser.add_argument("--reg_data_dir", type=str, default=None, help="directory for regularization images / 正則化画像データのディレクトリ")

    if support_caption:
        # caption dataset
        parser.add_argument("--in_json", type=str, default=None, help="json metadata for dataset / データセットのmetadataのjsonファイル")
        parser.add_argument(
            "--dataset_repeats", type=int, default=1, help="repeat dataset when training with captions / キャプションでの学習時にデータセットを繰り返す回数"
        )


def add_sd_saving_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--save_model_as",
        type=str,
        default=None,
        choices=[None, "ckpt", "safetensors", "diffusers", "diffusers_safetensors"],
        help="format to save the model (default is same to original) / モデル保存時の形式（未指定時は元モデルと同じ）",
    )
    parser.add_argument(
        "--use_safetensors",
        action="store_true",
        help="use safetensors format to save (if save_model_as is not specified) / checkpoint、モデルをsafetensors形式で保存する（save_model_as未指定時）",
    )


def read_config_from_file(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if not args.config_file:
        return args

    config_path = args.config_file + ".toml" if not args.config_file.endswith(".toml") else args.config_file

    if args.output_config:
        # check if config file exists
        if os.path.exists(config_path):
            print(f"Config file already exists. Aborting... / 出力先の設定ファイルが既に存在します: {config_path}")
            exit(1)

        # convert args to dictionary
        args_dict = vars(args)

        # remove unnecessary keys
        for key in ["config_file", "output_config", "wandb_api_key"]:
            if key in args_dict:
                del args_dict[key]

        # get default args from parser
        default_args = vars(parser.parse_args([]))

        # remove default values: cannot use args_dict.items directly because it will be changed during iteration
        for key, value in list(args_dict.items()):
            if key in default_args and value == default_args[key]:
                del args_dict[key]

        # convert Path to str in dictionary
        for key, value in args_dict.items():
            if isinstance(value, pathlib.Path):
                args_dict[key] = str(value)

        # convert to toml and output to file
        with open(config_path, "w") as f:
            toml.dump(args_dict, f)

        print(f"Saved config file / 設定ファイルを保存しました: {config_path}")
        exit(0)

    if not os.path.exists(config_path):
        print(f"{config_path} not found.")
        exit(1)

    print(f"Loading settings from {config_path}...")
    with open(config_path, "r") as f:
        config_dict = toml.load(f)

    # combine all sections into one
    ignore_nesting_dict = {}
    for section_name, section_dict in config_dict.items():
        # if value is not dict, save key and value as is
        if not isinstance(section_dict, dict):
            ignore_nesting_dict[section_name] = section_dict
            continue

        # if value is dict, save all key and value into one dict
        for key, value in section_dict.items():
            ignore_nesting_dict[key] = value

    config_args = argparse.Namespace(**ignore_nesting_dict)
    args = parser.parse_args(namespace=config_args)
    args.config_file = os.path.splitext(args.config_file)[0]
    print(args.config_file)

    return args


# endregion

# region utils


def resume_from_local_or_hf_if_specified(accelerator, args):
    if not args.resume:
        return

    if not args.resume_from_huggingface:
        print(f"resume training from local state: {args.resume}")
        accelerator.load_state(args.resume)
        return

    print(f"resume training from huggingface state: {args.resume}")
    repo_id = args.resume.split("/")[0] + "/" + args.resume.split("/")[1]
    path_in_repo = "/".join(args.resume.split("/")[2:])
    revision = None
    repo_type = None
    if ":" in path_in_repo:
        divided = path_in_repo.split(":")
        if len(divided) == 2:
            path_in_repo, revision = divided
            repo_type = "model"
        else:
            path_in_repo, revision, repo_type = divided
    print(f"Downloading state from huggingface: {repo_id}/{path_in_repo}@{revision}")

    list_files = huggingface_util.list_dir(
        repo_id=repo_id,
        subfolder=path_in_repo,
        revision=revision,
        token=args.huggingface_token,
        repo_type=repo_type,
    )

    async def download(filename) -> str:
        def task():
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                repo_type=repo_type,
                token=args.huggingface_token,
            )

        return await asyncio.get_event_loop().run_in_executor(None, task)

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*[download(filename=filename.rfilename) for filename in list_files]))
    if len(results) == 0:
        raise ValueError("No files found in the specified repo id/path/revision / 指定されたリポジトリID/パス/リビジョンにファイルが見つかりませんでした")
    dirname = os.path.dirname(results[0])
    accelerator.load_state(dirname)


def get_optimizer(args, trainable_params):
    # "Optimizer to use: AdamW, AdamW8bit, Lion, SGDNesterov, SGDNesterov8bit, Lion8bit, DAdaptation, DAdaptation(DAdaptAdam), DAdaptAdaGrad, DAdaptAdan, DAdaptSGD, Adafactor"

    optimizer_type = args.optimizer_type
    if args.use_8bit_adam:
        assert (
            not args.use_lion_optimizer
        ), "both option use_8bit_adam and use_lion_optimizer are specified / use_8bit_adamとuse_lion_optimizerの両方のオプションが指定されています"
        assert (
            optimizer_type is None or optimizer_type == ""
        ), "both option use_8bit_adam and optimizer_type are specified / use_8bit_adamとoptimizer_typeの両方のオプションが指定されています"
        optimizer_type = "AdamW8bit"

    elif args.use_lion_optimizer:
        assert (
            optimizer_type is None or optimizer_type == ""
        ), "both option use_lion_optimizer and optimizer_type are specified / use_lion_optimizerとoptimizer_typeの両方のオプションが指定されています"
        optimizer_type = "Lion"

    if optimizer_type is None or optimizer_type == "":
        optimizer_type = "AdamW"
    optimizer_type = optimizer_type.lower()

    # 引数を分解する
    optimizer_kwargs = {}
    if args.optimizer_args is not None and len(args.optimizer_args) > 0:
        for arg in args.optimizer_args:
            key, value = arg.split("=")
            value = ast.literal_eval(value)

            # value = value.split(",")
            # for i in range(len(value)):
            #     if value[i].lower() == "true" or value[i].lower() == "false":
            #         value[i] = value[i].lower() == "true"
            #     else:
            #         value[i] = ast.float(value[i])
            # if len(value) == 1:
            #     value = value[0]
            # else:
            #     value = tuple(value)

            optimizer_kwargs[key] = value
    # print("optkwargs:", optimizer_kwargs)

    lr = args.learning_rate

    if optimizer_type == "AdamW8bit".lower():
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("No bitsand bytes / bitsandbytesがインストールされていないようです")
        print(f"use 8-bit AdamW optimizer | {optimizer_kwargs}")
        optimizer_class = bnb.optim.AdamW8bit
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "SGDNesterov8bit".lower():
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("No bitsand bytes / bitsandbytesがインストールされていないようです")
        print(f"use 8-bit SGD with Nesterov optimizer | {optimizer_kwargs}")
        if "momentum" not in optimizer_kwargs:
            print(
                f"8-bit SGD with Nesterov must be with momentum, set momentum to 0.9 / 8-bit SGD with Nesterovはmomentum指定が必須のため0.9に設定します"
            )
            optimizer_kwargs["momentum"] = 0.9

        optimizer_class = bnb.optim.SGD8bit
        optimizer = optimizer_class(trainable_params, lr=lr, nesterov=True, **optimizer_kwargs)

    elif optimizer_type == "Lion".lower():
        try:
            import lion_pytorch
        except ImportError:
            raise ImportError("No lion_pytorch / lion_pytorch がインストールされていないようです")
        print(f"use Lion optimizer | {optimizer_kwargs}")
        optimizer_class = lion_pytorch.Lion
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "Lion8bit".lower():
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("No bitsandbytes / bitsandbytesがインストールされていないようです")

        print(f"use 8-bit Lion optimizer | {optimizer_kwargs}")
        try:
            optimizer_class = bnb.optim.Lion8bit
        except AttributeError:
            raise AttributeError(
                "No Lion8bit. The version of bitsandbytes installed seems to be old. Please install 0.38.0 or later. / Lion8bitが定義されていません。インストールされているbitsandbytesのバージョンが古いようです。0.38.0以上をインストールしてください"
            )

        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "SGDNesterov".lower():
        print(f"use SGD with Nesterov optimizer | {optimizer_kwargs}")
        if "momentum" not in optimizer_kwargs:
            print(f"SGD with Nesterov must be with momentum, set momentum to 0.9 / SGD with Nesterovはmomentum指定が必須のため0.9に設定します")
            optimizer_kwargs["momentum"] = 0.9

        optimizer_class = torch.optim.SGD
        optimizer = optimizer_class(trainable_params, lr=lr, nesterov=True, **optimizer_kwargs)

    elif optimizer_type.startswith("DAdapt".lower()):
        # DAdaptation family
        # check dadaptation is installed
        try:
            import dadaptation
        except ImportError:
            raise ImportError("No dadaptation / dadaptation がインストールされていないようです")

        # check lr and lr_count, and print warning
        actual_lr = lr
        lr_count = 1
        if type(trainable_params) == list and type(trainable_params[0]) == dict:
            lrs = set()
            actual_lr = trainable_params[0].get("lr", actual_lr)
            for group in trainable_params:
                lrs.add(group.get("lr", actual_lr))
            lr_count = len(lrs)

        if actual_lr <= 0.1:
            print(
                f"learning rate is too low. If using dadaptation, set learning rate around 1.0 / 学習率が低すぎるようです。1.0前後の値を指定してください: lr={actual_lr}"
            )
            print("recommend option: lr=1.0 / 推奨は1.0です")
        if lr_count > 1:
            print(
                f"when multiple learning rates are specified with dadaptation (e.g. for Text Encoder and U-Net), only the first one will take effect / D-Adaptationで複数の学習率を指定した場合（Text EncoderとU-Netなど）、最初の学習率のみが有効になります: lr={actual_lr}"
            )

        # set optimizer
        if optimizer_type == "DAdaptation".lower() or optimizer_type == "DAdaptAdam".lower():
            optimizer_class = dadaptation.DAdaptAdam
            print(f"use D-Adaptation Adam optimizer | {optimizer_kwargs}")
        elif optimizer_type == "DAdaptAdaGrad".lower():
            optimizer_class = dadaptation.DAdaptAdaGrad
            print(f"use D-Adaptation AdaGrad optimizer | {optimizer_kwargs}")
        elif optimizer_type == "DAdaptAdan".lower():
            optimizer_class = dadaptation.DAdaptAdan
            print(f"use D-Adaptation Adan optimizer | {optimizer_kwargs}")
        elif optimizer_type == "DAdaptSGD".lower():
            optimizer_class = dadaptation.DAdaptSGD
            print(f"use D-Adaptation SGD optimizer | {optimizer_kwargs}")
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "Adafactor".lower():
        # 引数を確認して適宜補正する
        if "relative_step" not in optimizer_kwargs:
            optimizer_kwargs["relative_step"] = True  # default
        if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
            print(f"set relative_step to True because warmup_init is True / warmup_initがTrueのためrelative_stepをTrueにします")
            optimizer_kwargs["relative_step"] = True
        print(f"use Adafactor optimizer | {optimizer_kwargs}")

        if optimizer_kwargs["relative_step"]:
            print(f"relative_step is true / relative_stepがtrueです")
            if lr != 0.0:
                print(f"learning rate is used as initial_lr / 指定したlearning rateはinitial_lrとして使用されます")
            args.learning_rate = None

            # trainable_paramsがgroupだった時の処理：lrを削除する
            if type(trainable_params) == list and type(trainable_params[0]) == dict:
                has_group_lr = False
                for group in trainable_params:
                    p = group.pop("lr", None)
                    has_group_lr = has_group_lr or (p is not None)

                if has_group_lr:
                    # 一応argsを無効にしておく TODO 依存関係が逆転してるのであまり望ましくない
                    print(f"unet_lr and text_encoder_lr are ignored / unet_lrとtext_encoder_lrは無視されます")
                    args.unet_lr = None
                    args.text_encoder_lr = None

            if args.lr_scheduler != "adafactor":
                print(f"use adafactor_scheduler / スケジューラにadafactor_schedulerを使用します")
            args.lr_scheduler = f"adafactor:{lr}"  # ちょっと微妙だけど

            lr = None
        else:
            if args.max_grad_norm != 0.0:
                print(
                    f"because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0 / max_grad_normが設定されているためclip_grad_normが有効になります。0に設定して無効にしたほうがいいかもしれません"
                )
            if args.lr_scheduler != "constant_with_warmup":
                print(f"constant_with_warmup will be good / スケジューラはconstant_with_warmupが良いかもしれません")
            if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
                print(f"clip_threshold=1.0 will be good / clip_thresholdは1.0が良いかもしれません")

        optimizer_class = transformers.optimization.Adafactor
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    elif optimizer_type == "AdamW".lower():
        print(f"use AdamW optimizer | {optimizer_kwargs}")
        optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    else:
        # 任意のoptimizerを使う
        optimizer_type = args.optimizer_type  # lowerでないやつ（微妙）
        print(f"use {optimizer_type} | {optimizer_kwargs}")
        if "." not in optimizer_type:
            optimizer_module = torch.optim
        else:
            values = optimizer_type.split(".")
            optimizer_module = importlib.import_module(".".join(values[:-1]))
            optimizer_type = values[-1]

        optimizer_class = getattr(optimizer_module, optimizer_type)
        optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

    optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
    optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

    return optimizer_name, optimizer_args, optimizer


# Monkeypatch newer get_scheduler() function overridng current version of diffusers.optimizer.get_scheduler
# code is taken from https://github.com/huggingface/diffusers diffusers.optimizer, commit d87cc15977b87160c30abaace3894e802ad9e1e6
# Which is a newer release of diffusers than currently packaged with sd-scripts
# This code can be removed when newer diffusers version (v0.12.1 or greater) is tested and implemented to sd-scripts


def get_scheduler_fix(args, optimizer: Optimizer, num_processes: int):
    """
    Unified API to get any scheduler from its name.
    """
    name = args.lr_scheduler
    num_warmup_steps: Optional[int] = args.lr_warmup_steps
    num_training_steps = args.max_train_steps * num_processes * args.gradient_accumulation_steps
    num_cycles = args.lr_scheduler_num_cycles
    power = args.lr_scheduler_power

    lr_scheduler_kwargs = {}  # get custom lr_scheduler kwargs
    if args.lr_scheduler_args is not None and len(args.lr_scheduler_args) > 0:
        for arg in args.lr_scheduler_args:
            key, value = arg.split("=")

            value = ast.literal_eval(value)
            # value = value.split(",")
            # for i in range(len(value)):
            #     if value[i].lower() == "true" or value[i].lower() == "false":
            #         value[i] = value[i].lower() == "true"
            #     else:
            #         value[i] = ast.literal_eval(value[i])
            # if len(value) == 1:
            #     value = value[0]
            # else:
            #     value = list(value)  # some may use list?

            lr_scheduler_kwargs[key] = value

    def wrap_check_needless_num_warmup_steps(return_vals):
        if num_warmup_steps is not None and num_warmup_steps != 0:
            raise ValueError(f"{name} does not require `num_warmup_steps`. Set None or 0.")
        return return_vals

    # using any lr_scheduler from other library
    if args.lr_scheduler_type:
        lr_scheduler_type = args.lr_scheduler_type
        print(f"use {lr_scheduler_type} | {lr_scheduler_kwargs} as lr_scheduler")
        if "." not in lr_scheduler_type:  # default to use torch.optim
            lr_scheduler_module = torch.optim.lr_scheduler
        else:
            values = lr_scheduler_type.split(".")
            lr_scheduler_module = importlib.import_module(".".join(values[:-1]))
            lr_scheduler_type = values[-1]
        lr_scheduler_class = getattr(lr_scheduler_module, lr_scheduler_type)
        lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_kwargs)
        return wrap_check_needless_num_warmup_steps(lr_scheduler)

    if name.startswith("adafactor"):
        assert (
            type(optimizer) == transformers.optimization.Adafactor
        ), f"adafactor scheduler must be used with Adafactor optimizer / adafactor schedulerはAdafactorオプティマイザと同時に使ってください"
        initial_lr = float(name.split(":")[1])
        # print("adafactor scheduler init lr", initial_lr)
        return wrap_check_needless_num_warmup_steps(transformers.optimization.AdafactorSchedule(optimizer, initial_lr))

    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return wrap_check_needless_num_warmup_steps(schedule_func(optimizer))

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, num_cycles=num_cycles
        )

    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, power=power)

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)


def prepare_dataset_args(args: argparse.Namespace, support_metadata: bool):
    # backward compatibility
    if args.caption_extention is not None:
        args.caption_extension = args.caption_extention
        args.caption_extention = None

    # assert args.resolution is not None, f"resolution is required / resolution（解像度）を指定してください"
    if args.resolution is not None:
        args.resolution = tuple([int(r) for r in args.resolution.split(",")])
        if len(args.resolution) == 1:
            args.resolution = (args.resolution[0], args.resolution[0])
        assert (
            len(args.resolution) == 2
        ), f"resolution must be 'size' or 'width,height' / resolution（解像度）は'サイズ'または'幅','高さ'で指定してください: {args.resolution}"

    if args.face_crop_aug_range is not None:
        args.face_crop_aug_range = tuple([float(r) for r in args.face_crop_aug_range.split(",")])
        assert (
            len(args.face_crop_aug_range) == 2 and args.face_crop_aug_range[0] <= args.face_crop_aug_range[1]
        ), f"face_crop_aug_range must be two floats / face_crop_aug_rangeは'下限,上限'で指定してください: {args.face_crop_aug_range}"
    else:
        args.face_crop_aug_range = None

    if support_metadata:
        if args.in_json is not None and (args.color_aug or args.random_crop):
            print(
                f"latents in npz is ignored when color_aug or random_crop is True / color_augまたはrandom_cropを有効にした場合、npzファイルのlatentsは無視されます"
            )


def load_tokenizer(args: argparse.Namespace):
    print("prepare tokenizer")
    original_path = V2_STABLE_DIFFUSION_PATH if args.v2 else TOKENIZER_PATH

    tokenizer: CLIPTokenizer = None
    if args.tokenizer_cache_dir:
        local_tokenizer_path = os.path.join(args.tokenizer_cache_dir, original_path.replace("/", "_"))
        if os.path.exists(local_tokenizer_path):
            print(f"load tokenizer from cache: {local_tokenizer_path}")
            tokenizer = CLIPTokenizer.from_pretrained(local_tokenizer_path)  # same for v1 and v2

    if tokenizer is None:
        if args.v2:
            tokenizer = CLIPTokenizer.from_pretrained(original_path, subfolder="tokenizer")
        else:
            tokenizer = CLIPTokenizer.from_pretrained(original_path)

    if hasattr(args, "max_token_length") and args.max_token_length is not None:
        print(f"update token length: {args.max_token_length}")

    if args.tokenizer_cache_dir and not os.path.exists(local_tokenizer_path):
        print(f"save Tokenizer to cache: {local_tokenizer_path}")
        tokenizer.save_pretrained(local_tokenizer_path)

    return tokenizer


def prepare_accelerator(args: argparse.Namespace):
    if args.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if args.log_prefix is None else args.log_prefix
        logging_dir = args.logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

    if args.log_with is None:
        if logging_dir is not None:
            log_with = "tensorboard"
        else:
            log_with = None
    else:
        log_with = args.log_with
        if log_with in ["tensorboard", "all"]:
            if logging_dir is None:
                raise ValueError("logging_dir is required when log_with is tensorboard / Tensorboardを使う場合、logging_dirを指定してください")
        if log_with in ["wandb", "all"]:
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb / wandb がインストールされていないようです")
            if logging_dir is not None:
                os.makedirs(logging_dir, exist_ok=True)
                os.environ["WANDB_DIR"] = logging_dir
            if args.wandb_api_key is not None:
                wandb.login(key=args.wandb_api_key)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        logging_dir=logging_dir,
    )

    # accelerateの互換性問題を解決する
    accelerator_0_15 = True
    try:
        accelerator.unwrap_model("dummy", True)
        print("Using accelerator 0.15.0 or above.")
    except TypeError:
        accelerator_0_15 = False

    def unwrap_model(model):
        if accelerator_0_15:
            return accelerator.unwrap_model(model, True)
        return accelerator.unwrap_model(model)

    return accelerator, unwrap_model


def prepare_dtype(args: argparse.Namespace):
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    save_dtype = None
    if args.save_precision == "fp16":
        save_dtype = torch.float16
    elif args.save_precision == "bf16":
        save_dtype = torch.bfloat16
    elif args.save_precision == "float":
        save_dtype = torch.float32

    return weight_dtype, save_dtype


def _load_target_model(args: argparse.Namespace, weight_dtype, device="cpu"):
    name_or_path = args.pretrained_model_name_or_path
    name_or_path = os.readlink(name_or_path) if os.path.islink(name_or_path) else name_or_path
    load_stable_diffusion_format = os.path.isfile(name_or_path)  # determine SD or Diffusers
    if load_stable_diffusion_format:
        print("load StableDiffusion checkpoint")
        text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(args.v2, name_or_path, device)
    else:
        # Diffusers model is loaded to CPU
        print("load Diffusers pretrained models")
        try:
            pipe = StableDiffusionPipeline.from_pretrained(name_or_path, tokenizer=None, safety_checker=None)
        except EnvironmentError as ex:
            print(
                f"model is not found as a file or in Hugging Face, perhaps file name is wrong? / 指定したモデル名のファイル、またはHugging Faceのモデルが見つかりません。ファイル名が誤っているかもしれません: {name_or_path}"
            )
        text_encoder = pipe.text_encoder
        vae = pipe.vae
        unet = pipe.unet
        del pipe

    # VAEを読み込む
    if args.vae is not None:
        vae = model_util.load_vae(args.vae, weight_dtype)
        print("additional VAE loaded")

    return text_encoder, vae, unet, load_stable_diffusion_format


def transform_if_model_is_DDP(text_encoder, unet, network=None):
    # Transform text_encoder, unet and network from DistributedDataParallel
    return (model.module if type(model) == DDP else model for model in [text_encoder, unet, network] if model is not None)


def load_target_model(args, weight_dtype, accelerator):
    # load models for each process
    for pi in range(accelerator.state.num_processes):
        if pi == accelerator.state.local_process_index:
            print(f"loading model for process {accelerator.state.local_process_index}/{accelerator.state.num_processes}")

            text_encoder, vae, unet, load_stable_diffusion_format = _load_target_model(
                args, weight_dtype, accelerator.device if args.lowram else "cpu"
            )

            # work on low-ram device
            if args.lowram:
                text_encoder.to(accelerator.device)
                unet.to(accelerator.device)
                vae.to(accelerator.device)

            gc.collect()
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    text_encoder, unet = transform_if_model_is_DDP(text_encoder, unet)

    return text_encoder, vae, unet, load_stable_diffusion_format


def patch_accelerator_for_fp16_training(accelerator):
    org_unscale_grads = accelerator.scaler._unscale_grads_

    def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        return org_unscale_grads(optimizer, inv_scale, found_inf, True)

    accelerator.scaler._unscale_grads_ = _unscale_grads_replacer


def get_hidden_states(args: argparse.Namespace, input_ids, tokenizer, text_encoder, weight_dtype=None):
    # with no_token_padding, the length is not max length, return result immediately
    if input_ids.size()[-1] != tokenizer.model_max_length:
        return text_encoder(input_ids)[0]

    b_size = input_ids.size()[0]
    input_ids = input_ids.reshape((-1, tokenizer.model_max_length))  # batch_size*3, 77

    if args.clip_skip is None:
        encoder_hidden_states = text_encoder(input_ids)[0]
    else:
        enc_out = text_encoder(input_ids, output_hidden_states=True, return_dict=True)
        encoder_hidden_states = enc_out["hidden_states"][-args.clip_skip]
        encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

    # bs*3, 77, 768 or 1024
    encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))

    if args.max_token_length is not None:
        if args.v2:
            # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
            states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, args.max_token_length, tokenizer.model_max_length):
                chunk = encoder_hidden_states[:, i : i + tokenizer.model_max_length - 2]  # <BOS> の後から 最後の前まで
                if i > 0:
                    for j in range(len(chunk)):
                        if input_ids[j, 1] == tokenizer.eos_token:  # 空、つまり <BOS> <EOS> <PAD> ...のパターン
                            chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
                states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
            states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS> か <PAD> のどちらか
            encoder_hidden_states = torch.cat(states_list, dim=1)
        else:
            # v1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
            states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, args.max_token_length, tokenizer.model_max_length):
                states_list.append(encoder_hidden_states[:, i : i + tokenizer.model_max_length - 2])  # <BOS> の後から <EOS> の前まで
            states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS>
            encoder_hidden_states = torch.cat(states_list, dim=1)

    if weight_dtype is not None:
        # this is required for additional network training
        encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

    return encoder_hidden_states


def default_if_none(value, default):
    return default if value is None else value


def get_epoch_ckpt_name(args: argparse.Namespace, ext: str, epoch_no: int):
    model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)
    return EPOCH_FILE_NAME.format(model_name, epoch_no) + ext


def get_step_ckpt_name(args: argparse.Namespace, ext: str, step_no: int):
    model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)
    return STEP_FILE_NAME.format(model_name, step_no) + ext


def get_last_ckpt_name(args: argparse.Namespace, ext: str):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    return model_name + ext


def get_remove_epoch_no(args: argparse.Namespace, epoch_no: int):
    if args.save_last_n_epochs is None:
        return None

    remove_epoch_no = epoch_no - args.save_every_n_epochs * args.save_last_n_epochs
    if remove_epoch_no < 0:
        return None
    return remove_epoch_no


def get_remove_step_no(args: argparse.Namespace, step_no: int):
    if args.save_last_n_steps is None:
        return None

    # last_n_steps前のstep_noから、save_every_n_stepsの倍数のstep_noを計算して削除する
    # save_every_n_steps=10, save_last_n_steps=30の場合、50step目には30step分残し、10step目を削除する
    remove_step_no = step_no - args.save_last_n_steps - 1
    remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)
    if remove_step_no < 0:
        return None
    return remove_step_no


# epochとstepの保存、メタデータにepoch/stepが含まれ引数が同じになるため、統合している
# on_epoch_end: Trueならepoch終了時、Falseならstep経過時
def save_sd_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    src_path: str,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    text_encoder,
    unet,
    vae,
):
    if on_epoch_end:
        epoch_no = epoch + 1
        saving = epoch_no % args.save_every_n_epochs == 0 and epoch_no < num_train_epochs
        if not saving:
            return

        model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)
        remove_no = get_remove_epoch_no(args, epoch_no)
    else:
        # 保存するか否かは呼び出し側で判断済み

        model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)
        epoch_no = epoch  # 例: 最初のepochの途中で保存したら0になる、SDモデルに保存される
        remove_no = get_remove_step_no(args, global_step)

    os.makedirs(args.output_dir, exist_ok=True)
    if save_stable_diffusion_format:
        ext = ".safetensors" if use_safetensors else ".ckpt"

        if on_epoch_end:
            ckpt_name = get_epoch_ckpt_name(args, ext, epoch_no)
        else:
            ckpt_name = get_step_ckpt_name(args, ext, global_step)

        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        print(f"saving checkpoint: {ckpt_file}")
        model_util.save_stable_diffusion_checkpoint(
            args.v2, ckpt_file, text_encoder, unet, src_path, epoch_no, global_step, save_dtype, vae
        )

        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, ckpt_file, "/" + ckpt_name)

        # remove older checkpoints
        if remove_no is not None:
            if on_epoch_end:
                remove_ckpt_name = get_epoch_ckpt_name(args, ext, remove_no)
            else:
                remove_ckpt_name = get_step_ckpt_name(args, ext, remove_no)

            remove_ckpt_file = os.path.join(args.output_dir, remove_ckpt_name)
            if os.path.exists(remove_ckpt_file):
                print(f"removing old checkpoint: {remove_ckpt_file}")
                os.remove(remove_ckpt_file)

    else:
        if on_epoch_end:
            out_dir = os.path.join(args.output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(model_name, epoch_no))
        else:
            out_dir = os.path.join(args.output_dir, STEP_DIFFUSERS_DIR_NAME.format(model_name, global_step))

        print(f"saving model: {out_dir}")
        model_util.save_diffusers_checkpoint(
            args.v2, out_dir, text_encoder, unet, src_path, vae=vae, use_safetensors=use_safetensors
        )
        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, out_dir, "/" + model_name)

        # remove older checkpoints
        if remove_no is not None:
            if on_epoch_end:
                remove_out_dir = os.path.join(args.output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(model_name, remove_no))
            else:
                remove_out_dir = os.path.join(args.output_dir, STEP_DIFFUSERS_DIR_NAME.format(model_name, remove_no))

            if os.path.exists(remove_out_dir):
                print(f"removing old model: {remove_out_dir}")
                shutil.rmtree(remove_out_dir)

    if on_epoch_end:
        save_and_remove_state_on_epoch_end(args, accelerator, epoch_no)
    else:
        save_and_remove_state_stepwise(args, accelerator, global_step)


def save_and_remove_state_on_epoch_end(args: argparse.Namespace, accelerator, epoch_no):
    model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)

    print(f"saving state at epoch {epoch_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no))
    accelerator.save_state(state_dir)
    if args.save_state_to_huggingface:
        print("uploading state to huggingface.")
        huggingface_util.upload(args, state_dir, "/" + EPOCH_STATE_NAME.format(model_name, epoch_no))

    last_n_epochs = args.save_last_n_epochs_state if args.save_last_n_epochs_state else args.save_last_n_epochs
    if last_n_epochs is not None:
        remove_epoch_no = epoch_no - args.save_every_n_epochs * last_n_epochs
        state_dir_old = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, remove_epoch_no))
        if os.path.exists(state_dir_old):
            print(f"removing old state: {state_dir_old}")
            shutil.rmtree(state_dir_old)


def save_and_remove_state_stepwise(args: argparse.Namespace, accelerator, step_no):
    model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)

    print(f"saving state at step {step_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, step_no))
    accelerator.save_state(state_dir)
    if args.save_state_to_huggingface:
        print("uploading state to huggingface.")
        huggingface_util.upload(args, state_dir, "/" + STEP_STATE_NAME.format(model_name, step_no))

    last_n_steps = args.save_last_n_steps_state if args.save_last_n_steps_state else args.save_last_n_steps
    if last_n_steps is not None:
        # last_n_steps前のstep_noから、save_every_n_stepsの倍数のstep_noを計算して削除する
        remove_step_no = step_no - last_n_steps - 1
        remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)

        if remove_step_no > 0:
            state_dir_old = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, remove_step_no))
            if os.path.exists(state_dir_old):
                print(f"removing old state: {state_dir_old}")
                shutil.rmtree(state_dir_old)


def save_state_on_train_end(args: argparse.Namespace, accelerator):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)

    print("saving last state.")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, LAST_STATE_NAME.format(model_name))
    accelerator.save_state(state_dir)

    if args.save_state_to_huggingface:
        print("uploading last state to huggingface.")
        huggingface_util.upload(args, state_dir, "/" + LAST_STATE_NAME.format(model_name))


def save_sd_model_on_train_end(
    args: argparse.Namespace,
    src_path: str,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    text_encoder,
    unet,
    vae,
):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)

    if save_stable_diffusion_format:
        os.makedirs(args.output_dir, exist_ok=True)

        ckpt_name = model_name + (".safetensors" if use_safetensors else ".ckpt")
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        print(f"save trained model as StableDiffusion checkpoint to {ckpt_file}")
        model_util.save_stable_diffusion_checkpoint(
            args.v2, ckpt_file, text_encoder, unet, src_path, epoch, global_step, save_dtype, vae
        )
        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=True)
    else:
        out_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"save trained model as Diffusers to {out_dir}")
        model_util.save_diffusers_checkpoint(
            args.v2, out_dir, text_encoder, unet, src_path, vae=vae, use_safetensors=use_safetensors
        )
        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, out_dir, "/" + model_name, force_sync_upload=True)


# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"


def sample_images(
    accelerator, args: argparse.Namespace, epoch, steps, device, vae, tokenizer, text_encoder, unet, prompt_replacement=None
):
    """
    StableDiffusionLongPromptWeightingPipelineの改造版を使うようにしたので、clip skipおよびプロンプトの重みづけに対応した
    """
    if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
        return
    if args.sample_every_n_epochs is not None:
        # sample_every_n_steps は無視する
        if epoch is None or epoch % args.sample_every_n_epochs != 0:
            return
    else:
        if steps % args.sample_every_n_steps != 0 or epoch is not None:  # steps is not divisible or end of epoch
            return

    print(f"generating sample images at step / サンプル画像生成 ステップ: {steps}")
    if not os.path.isfile(args.sample_prompts):
        print(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
        return

    org_vae_device = vae.device  # CPUにいるはず
    vae.to(device)

    # read prompts
    with open(args.sample_prompts, "rt", encoding="utf-8") as f:
        prompts = f.readlines()

    # schedulerを用意する
    sched_init_args = {}
    if args.sample_sampler == "ddim":
        scheduler_cls = DDIMScheduler
    elif args.sample_sampler == "ddpm":  # ddpmはおかしくなるのでoptionから外してある
        scheduler_cls = DDPMScheduler
    elif args.sample_sampler == "pndm":
        scheduler_cls = PNDMScheduler
    elif args.sample_sampler == "lms" or args.sample_sampler == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
    elif args.sample_sampler == "euler" or args.sample_sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
    elif args.sample_sampler == "euler_a" or args.sample_sampler == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif args.sample_sampler == "dpmsolver" or args.sample_sampler == "dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = args.sample_sampler
    elif args.sample_sampler == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
    elif args.sample_sampler == "heun":
        scheduler_cls = HeunDiscreteScheduler
    elif args.sample_sampler == "dpm_2" or args.sample_sampler == "k_dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
    elif args.sample_sampler == "dpm_2_a" or args.sample_sampler == "k_dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    else:
        scheduler_cls = DDIMScheduler

    if args.v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction"

    scheduler = scheduler_cls(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
        **sched_init_args,
    )

    # clip_sample=Trueにする
    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
        # print("set clip_sample to True")
        scheduler.config.clip_sample = True

    pipeline = StableDiffusionLongPromptWeightingPipeline(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=scheduler,
        clip_skip=args.clip_skip,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipeline.to(device)

    save_dir = args.output_dir + "/sample"
    os.makedirs(save_dir, exist_ok=True)

    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    with torch.no_grad():
        with accelerator.autocast():
            for i, prompt in enumerate(prompts):
                if not accelerator.is_main_process:
                    continue
                prompt = prompt.strip()
                if len(prompt) == 0 or prompt[0] == "#":
                    continue

                # subset of gen_img_diffusers
                prompt_args = prompt.split(" --")
                prompt = prompt_args[0]
                negative_prompt = None
                sample_steps = 30
                width = height = 512
                scale = 7.5
                seed = None
                for parg in prompt_args:
                    try:
                        m = re.match(r"w (\d+)", parg, re.IGNORECASE)
                        if m:
                            width = int(m.group(1))
                            continue

                        m = re.match(r"h (\d+)", parg, re.IGNORECASE)
                        if m:
                            height = int(m.group(1))
                            continue

                        m = re.match(r"d (\d+)", parg, re.IGNORECASE)
                        if m:
                            seed = int(m.group(1))
                            continue

                        m = re.match(r"s (\d+)", parg, re.IGNORECASE)
                        if m:  # steps
                            sample_steps = max(1, min(1000, int(m.group(1))))
                            continue

                        m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
                        if m:  # scale
                            scale = float(m.group(1))
                            continue

                        m = re.match(r"n (.+)", parg, re.IGNORECASE)
                        if m:  # negative prompt
                            negative_prompt = m.group(1)
                            continue

                    except ValueError as ex:
                        print(f"Exception in parsing / 解析エラー: {parg}")
                        print(ex)

                if seed is not None:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed(seed)

                if prompt_replacement is not None:
                    prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
                    if negative_prompt is not None:
                        negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

                height = max(64, height - height % 8)  # round to divisible by 8
                width = max(64, width - width % 8)  # round to divisible by 8
                print(f"prompt: {prompt}")
                print(f"negative_prompt: {negative_prompt}")
                print(f"height: {height}")
                print(f"width: {width}")
                print(f"sample_steps: {sample_steps}")
                print(f"scale: {scale}")
                image = pipeline(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=sample_steps,
                    guidance_scale=scale,
                    negative_prompt=negative_prompt,
                ).images[0]

                ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
                num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
                seed_suffix = "" if seed is None else f"_{seed}"
                img_filename = (
                    f"{'' if args.output_name is None else args.output_name + '_'}{ts_str}_{num_suffix}_{i:02d}{seed_suffix}.png"
                )

                image.save(os.path.join(save_dir, img_filename))

                # wandb有効時のみログを送信
                try:
                    wandb_tracker = accelerator.get_tracker("wandb")
                    try:
                        import wandb
                    except ImportError:  # 事前に一度確認するのでここはエラー出ないはず
                        raise ImportError("No wandb / wandb がインストールされていないようです")

                    wandb_tracker.log({f"sample_{i}": wandb.Image(image)})
                except:  # wandb 無効時
                    pass

    # clear pipeline and cache to reduce vram usage
    del pipeline
    torch.cuda.empty_cache()

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    vae.to(org_vae_device)


# endregion

# region 前処理用


class ImageLoadingDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            # convert to tensor temporarily so dataloader will accept it
            tensor_pil = transforms.functional.pil_to_tensor(image)
        except Exception as e:
            print(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None

        return (tensor_pil, img_path)


# endregion


# collate_fn用 epoch,stepはmultiprocessing.Value
class collater_class:
    def __init__(self, epoch, step, dataset):
        self.current_epoch = epoch
        self.current_step = step
        self.dataset = dataset  # not used if worker_info is not None, in case of multiprocessing

    def __call__(self, examples):
        worker_info = torch.utils.data.get_worker_info()
        # worker_info is None in the main process
        if worker_info is not None:
            dataset = worker_info.dataset
        else:
            dataset = self.dataset

        # set epoch and step
        dataset.set_current_epoch(self.current_epoch.value)
        dataset.set_current_step(self.current_step.value)
        return examples[0]
