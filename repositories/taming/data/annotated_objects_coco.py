import json
from itertools import chain
from pathlib import Path
from typing import Iterable, Dict, List, Callable, Any
from collections import defaultdict

from tqdm import tqdm

from taming.data.annotated_objects_dataset import AnnotatedObjectsDataset
from taming.data.helper_types import Annotation, ImageDescription, Category

COCO_PATH_STRUCTURE = {
    'train': {
        'top_level': '',
        'instances_annotations': 'annotations/instances_train2017.json',
        'stuff_annotations': 'annotations/stuff_train2017.json',
        'files': 'train2017'
    },
    'validation': {
        'top_level': '',
        'instances_annotations': 'annotations/instances_val2017.json',
        'stuff_annotations': 'annotations/stuff_val2017.json',
        'files': 'val2017'
    }
}


def load_image_descriptions(description_json: List[Dict]) -> Dict[str, ImageDescription]:
    return {
        str(img['id']): ImageDescription(
            id=img['id'],
            license=img.get('license'),
            file_name=img['file_name'],
            coco_url=img['coco_url'],
            original_size=(img['width'], img['height']),
            date_captured=img.get('date_captured'),
            flickr_url=img.get('flickr_url')
        )
        for img in description_json
    }


def load_categories(category_json: Iterable) -> Dict[str, Category]:
    return {str(cat['id']): Category(id=str(cat['id']), super_category=cat['supercategory'], name=cat['name'])
            for cat in category_json if cat['name'] != 'other'}


def load_annotations(annotations_json: List[Dict], image_descriptions: Dict[str, ImageDescription],
                     category_no_for_id: Callable[[str], int], split: str) -> Dict[str, List[Annotation]]:
    annotations = defaultdict(list)
    total = sum(len(a) for a in annotations_json)
    for ann in tqdm(chain(*annotations_json), f'Loading {split} annotations', total=total):
        image_id = str(ann['image_id'])
        if image_id not in image_descriptions:
            raise ValueError(f'image_id [{image_id}] has no image description.')
        category_id = ann['category_id']
        try:
            category_no = category_no_for_id(str(category_id))
        except KeyError:
            continue

        width, height = image_descriptions[image_id].original_size
        bbox = (ann['bbox'][0] / width, ann['bbox'][1] / height, ann['bbox'][2] / width, ann['bbox'][3] / height)

        annotations[image_id].append(
            Annotation(
                id=ann['id'],
                area=bbox[2]*bbox[3],  # use bbox area
                is_group_of=ann['iscrowd'],
                image_id=ann['image_id'],
                bbox=bbox,
                category_id=str(category_id),
                category_no=category_no
            )
        )
    return dict(annotations)


class AnnotatedObjectsCoco(AnnotatedObjectsDataset):
    def __init__(self, use_things: bool = True, use_stuff: bool = True, **kwargs):
        """
        @param data_path: is the path to the following folder structure:
                          coco/
                          ├── annotations
                          │   ├── instances_train2017.json
                          │   ├── instances_val2017.json
                          │   ├── stuff_train2017.json
                          │   └── stuff_val2017.json
                          ├── train2017
                          │   ├── 000000000009.jpg
                          │   ├── 000000000025.jpg
                          │   └── ...
                          ├── val2017
                          │   ├── 000000000139.jpg
                          │   ├── 000000000285.jpg
                          │   └── ...
        @param: split: one of 'train' or 'validation'
        @param: desired image size (give square images)
        """
        super().__init__(**kwargs)
        self.use_things = use_things
        self.use_stuff = use_stuff

        with open(self.paths['instances_annotations']) as f:
            inst_data_json = json.load(f)
        with open(self.paths['stuff_annotations']) as f:
            stuff_data_json = json.load(f)

        category_jsons = []
        annotation_jsons = []
        if self.use_things:
            category_jsons.append(inst_data_json['categories'])
            annotation_jsons.append(inst_data_json['annotations'])
        if self.use_stuff:
            category_jsons.append(stuff_data_json['categories'])
            annotation_jsons.append(stuff_data_json['annotations'])

        self.categories = load_categories(chain(*category_jsons))
        self.filter_categories()
        self.setup_category_id_and_number()

        self.image_descriptions = load_image_descriptions(inst_data_json['images'])
        annotations = load_annotations(annotation_jsons, self.image_descriptions, self.get_category_number, self.split)
        self.annotations = self.filter_object_number(annotations, self.min_object_area,
                                                     self.min_objects_per_image, self.max_objects_per_image)
        self.image_ids = list(self.annotations.keys())
        self.clean_up_annotations_and_image_descriptions()

    def get_path_structure(self) -> Dict[str, str]:
        if self.split not in COCO_PATH_STRUCTURE:
            raise ValueError(f'Split [{self.split} does not exist for COCO data.]')
        return COCO_PATH_STRUCTURE[self.split]

    def get_image_path(self, image_id: str) -> Path:
        return self.paths['files'].joinpath(self.image_descriptions[str(image_id)].file_name)

    def get_image_description(self, image_id: str) -> Dict[str, Any]:
        # noinspection PyProtectedMember
        return self.image_descriptions[image_id]._asdict()
