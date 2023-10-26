from pathlib import Path
from typing import Optional, List, Callable, Dict, Any, Union
import warnings

import PIL.Image as pil_image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from taming.data.conditional_builder.objects_bbox import ObjectsBoundingBoxConditionalBuilder
from taming.data.conditional_builder.objects_center_points import ObjectsCenterPointsConditionalBuilder
from taming.data.conditional_builder.utils import load_object_from_string
from taming.data.helper_types import BoundingBox, CropMethodType, Image, Annotation, SplitType
from taming.data.image_transforms import CenterCropReturnCoordinates, RandomCrop1dReturnCoordinates, \
    Random2dCropReturnCoordinates, RandomHorizontalFlipReturn, convert_pil_to_tensor


class AnnotatedObjectsDataset(Dataset):
    def __init__(self, data_path: Union[str, Path], split: SplitType, keys: List[str], target_image_size: int,
                 min_object_area: float, min_objects_per_image: int, max_objects_per_image: int,
                 crop_method: CropMethodType, random_flip: bool, no_tokens: int, use_group_parameter: bool,
                 encode_crop: bool, category_allow_list_target: str = "", category_mapping_target: str = "",
                 no_object_classes: Optional[int] = None):
        self.data_path = data_path
        self.split = split
        self.keys = keys
        self.target_image_size = target_image_size
        self.min_object_area = min_object_area
        self.min_objects_per_image = min_objects_per_image
        self.max_objects_per_image = max_objects_per_image
        self.crop_method = crop_method
        self.random_flip = random_flip
        self.no_tokens = no_tokens
        self.use_group_parameter = use_group_parameter
        self.encode_crop = encode_crop

        self.annotations = None
        self.image_descriptions = None
        self.categories = None
        self.category_ids = None
        self.category_number = None
        self.image_ids = None
        self.transform_functions: List[Callable] = self.setup_transform(target_image_size, crop_method, random_flip)
        self.paths = self.build_paths(self.data_path)
        self._conditional_builders = None
        self.category_allow_list = None
        if category_allow_list_target:
            allow_list = load_object_from_string(category_allow_list_target)
            self.category_allow_list = {name for name, _ in allow_list}
        self.category_mapping = {}
        if category_mapping_target:
            self.category_mapping = load_object_from_string(category_mapping_target)
        self.no_object_classes = no_object_classes

    def build_paths(self, top_level: Union[str, Path]) -> Dict[str, Path]:
        top_level = Path(top_level)
        sub_paths = {name: top_level.joinpath(sub_path) for name, sub_path in self.get_path_structure().items()}
        for path in sub_paths.values():
            if not path.exists():
                raise FileNotFoundError(f'{type(self).__name__} data structure error: [{path}] does not exist.')
        return sub_paths

    @staticmethod
    def load_image_from_disk(path: Path) -> Image:
        return pil_image.open(path).convert('RGB')

    @staticmethod
    def setup_transform(target_image_size: int, crop_method: CropMethodType, random_flip: bool):
        transform_functions = []
        if crop_method == 'none':
            transform_functions.append(transforms.Resize((target_image_size, target_image_size)))
        elif crop_method == 'center':
            transform_functions.extend([
                transforms.Resize(target_image_size),
                CenterCropReturnCoordinates(target_image_size)
            ])
        elif crop_method == 'random-1d':
            transform_functions.extend([
                transforms.Resize(target_image_size),
                RandomCrop1dReturnCoordinates(target_image_size)
            ])
        elif crop_method == 'random-2d':
            transform_functions.extend([
                Random2dCropReturnCoordinates(target_image_size),
                transforms.Resize(target_image_size)
            ])
        elif crop_method is None:
            return None
        else:
            raise ValueError(f'Received invalid crop method [{crop_method}].')
        if random_flip:
            transform_functions.append(RandomHorizontalFlipReturn())
        transform_functions.append(transforms.Lambda(lambda x: x / 127.5 - 1.))
        return transform_functions

    def image_transform(self, x: Tensor) -> (Optional[BoundingBox], Optional[bool], Tensor):
        crop_bbox = None
        flipped = None
        for t in self.transform_functions:
            if isinstance(t, (RandomCrop1dReturnCoordinates, CenterCropReturnCoordinates, Random2dCropReturnCoordinates)):
                crop_bbox, x = t(x)
            elif isinstance(t, RandomHorizontalFlipReturn):
                flipped, x = t(x)
            else:
                x = t(x)
        return crop_bbox, flipped, x

    @property
    def no_classes(self) -> int:
        return self.no_object_classes if self.no_object_classes else len(self.categories)

    @property
    def conditional_builders(self) -> ObjectsCenterPointsConditionalBuilder:
        # cannot set this up in init because no_classes is only known after loading data in init of superclass
        if self._conditional_builders is None:
            self._conditional_builders = {
                'objects_center_points': ObjectsCenterPointsConditionalBuilder(
                    self.no_classes,
                    self.max_objects_per_image,
                    self.no_tokens,
                    self.encode_crop,
                    self.use_group_parameter,
                    getattr(self, 'use_additional_parameters', False)
                ),
                'objects_bbox': ObjectsBoundingBoxConditionalBuilder(
                    self.no_classes,
                    self.max_objects_per_image,
                    self.no_tokens,
                    self.encode_crop,
                    self.use_group_parameter,
                    getattr(self, 'use_additional_parameters', False)
                )
            }
        return self._conditional_builders

    def filter_categories(self) -> None:
        if self.category_allow_list:
            self.categories = {id_: cat for id_, cat in self.categories.items() if cat.name in self.category_allow_list}
        if self.category_mapping:
            self.categories = {id_: cat for id_, cat in self.categories.items() if cat.id not in self.category_mapping}

    def setup_category_id_and_number(self) -> None:
        self.category_ids = list(self.categories.keys())
        self.category_ids.sort()
        if '/m/01s55n' in self.category_ids:
            self.category_ids.remove('/m/01s55n')
            self.category_ids.append('/m/01s55n')
        self.category_number = {category_id: i for i, category_id in enumerate(self.category_ids)}
        if self.category_allow_list is not None and self.category_mapping is None \
                and len(self.category_ids) != len(self.category_allow_list):
            warnings.warn('Unexpected number of categories: Mismatch with category_allow_list. '
                          'Make sure all names in category_allow_list exist.')

    def clean_up_annotations_and_image_descriptions(self) -> None:
        image_id_set = set(self.image_ids)
        self.annotations = {k: v for k, v in self.annotations.items() if k in image_id_set}
        self.image_descriptions = {k: v for k, v in self.image_descriptions.items() if k in image_id_set}

    @staticmethod
    def filter_object_number(all_annotations: Dict[str, List[Annotation]], min_object_area: float,
                             min_objects_per_image: int, max_objects_per_image: int) -> Dict[str, List[Annotation]]:
        filtered = {}
        for image_id, annotations in all_annotations.items():
            annotations_with_min_area = [a for a in annotations if a.area > min_object_area]
            if min_objects_per_image <= len(annotations_with_min_area) <= max_objects_per_image:
                filtered[image_id] = annotations_with_min_area
        return filtered

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, n: int) -> Dict[str, Any]:
        image_id = self.get_image_id(n)
        sample = self.get_image_description(image_id)
        sample['annotations'] = self.get_annotation(image_id)

        if 'image' in self.keys:
            sample['image_path'] = str(self.get_image_path(image_id))
            sample['image'] = self.load_image_from_disk(sample['image_path'])
            sample['image'] = convert_pil_to_tensor(sample['image'])
            sample['crop_bbox'], sample['flipped'], sample['image'] = self.image_transform(sample['image'])
            sample['image'] = sample['image'].permute(1, 2, 0)

        for conditional, builder in self.conditional_builders.items():
            if conditional in self.keys:
                sample[conditional] = builder.build(sample['annotations'], sample['crop_bbox'], sample['flipped'])

        if self.keys:
            # only return specified keys
            sample = {key: sample[key] for key in self.keys}
        return sample

    def get_image_id(self, no: int) -> str:
        return self.image_ids[no]

    def get_annotation(self, image_id: str) -> str:
        return self.annotations[image_id]

    def get_textual_label_for_category_id(self, category_id: str) -> str:
        return self.categories[category_id].name

    def get_textual_label_for_category_no(self, category_no: int) -> str:
        return self.categories[self.get_category_id(category_no)].name

    def get_category_number(self, category_id: str) -> int:
        return self.category_number[category_id]

    def get_category_id(self, category_no: int) -> str:
        return self.category_ids[category_no]

    def get_image_description(self, image_id: str) -> Dict[str, Any]:
        raise NotImplementedError()

    def get_path_structure(self):
        raise NotImplementedError

    def get_image_path(self, image_id: str) -> Path:
        raise NotImplementedError
