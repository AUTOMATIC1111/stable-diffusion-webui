from collections import defaultdict
from csv import DictReader, reader as TupleReader
from pathlib import Path
from typing import Dict, List, Any
import warnings

from taming.data.annotated_objects_dataset import AnnotatedObjectsDataset
from taming.data.helper_types import Annotation, Category
from tqdm import tqdm

OPEN_IMAGES_STRUCTURE = {
    'train': {
        'top_level': '',
        'class_descriptions': 'class-descriptions-boxable.csv',
        'annotations': 'oidv6-train-annotations-bbox.csv',
        'file_list': 'train-images-boxable.csv',
        'files': 'train'
    },
    'validation': {
        'top_level': '',
        'class_descriptions': 'class-descriptions-boxable.csv',
        'annotations': 'validation-annotations-bbox.csv',
        'file_list': 'validation-images.csv',
        'files': 'validation'
    },
    'test': {
        'top_level': '',
        'class_descriptions': 'class-descriptions-boxable.csv',
        'annotations': 'test-annotations-bbox.csv',
        'file_list': 'test-images.csv',
        'files': 'test'
    }
}


def load_annotations(descriptor_path: Path, min_object_area: float, category_mapping: Dict[str, str],
                     category_no_for_id: Dict[str, int]) -> Dict[str, List[Annotation]]:
    annotations: Dict[str, List[Annotation]] = defaultdict(list)
    with open(descriptor_path) as file:
        reader = DictReader(file)
        for i, row in tqdm(enumerate(reader), total=14620000, desc='Loading OpenImages annotations'):
            width = float(row['XMax']) - float(row['XMin'])
            height = float(row['YMax']) - float(row['YMin'])
            area = width * height
            category_id = row['LabelName']
            if category_id in category_mapping:
                category_id = category_mapping[category_id]
            if area >= min_object_area and category_id in category_no_for_id:
                annotations[row['ImageID']].append(
                    Annotation(
                        id=i,
                        image_id=row['ImageID'],
                        source=row['Source'],
                        category_id=category_id,
                        category_no=category_no_for_id[category_id],
                        confidence=float(row['Confidence']),
                        bbox=(float(row['XMin']), float(row['YMin']), width, height),
                        area=area,
                        is_occluded=bool(int(row['IsOccluded'])),
                        is_truncated=bool(int(row['IsTruncated'])),
                        is_group_of=bool(int(row['IsGroupOf'])),
                        is_depiction=bool(int(row['IsDepiction'])),
                        is_inside=bool(int(row['IsInside']))
                    )
                )
        if 'train' in str(descriptor_path) and i < 14000000:
            warnings.warn(f'Running with subset of Open Images. Train dataset has length [{len(annotations)}].')
        return dict(annotations)


def load_image_ids(csv_path: Path) -> List[str]:
    with open(csv_path) as file:
        reader = DictReader(file)
        return [row['image_name'] for row in reader]


def load_categories(csv_path: Path) -> Dict[str, Category]:
    with open(csv_path) as file:
        reader = TupleReader(file)
        return {row[0]: Category(id=row[0], name=row[1], super_category=None) for row in reader}


class AnnotatedObjectsOpenImages(AnnotatedObjectsDataset):
    def __init__(self, use_additional_parameters: bool, **kwargs):
        """
        @param data_path: is the path to the following folder structure:
                          open_images/
                          │   oidv6-train-annotations-bbox.csv
                          ├── class-descriptions-boxable.csv
                          ├── oidv6-train-annotations-bbox.csv
                          ├── test
                          │   ├── 000026e7ee790996.jpg
                          │   ├── 000062a39995e348.jpg
                          │   └── ...
                          ├── test-annotations-bbox.csv
                          ├── test-images.csv
                          ├── train
                          │   ├── 000002b66c9c498e.jpg
                          │   ├── 000002b97e5471a0.jpg
                          │   └── ...
                          ├── train-images-boxable.csv
                          ├── validation
                          │   ├── 0001eeaf4aed83f9.jpg
                          │   ├── 0004886b7d043cfd.jpg
                          │   └── ...
                          ├── validation-annotations-bbox.csv
                          └── validation-images.csv
        @param: split: one of 'train', 'validation' or 'test'
        @param: desired image size (returns square images)
        """

        super().__init__(**kwargs)
        self.use_additional_parameters = use_additional_parameters

        self.categories = load_categories(self.paths['class_descriptions'])
        self.filter_categories()
        self.setup_category_id_and_number()

        self.image_descriptions = {}
        annotations = load_annotations(self.paths['annotations'], self.min_object_area, self.category_mapping,
                                       self.category_number)
        self.annotations = self.filter_object_number(annotations, self.min_object_area, self.min_objects_per_image,
                                                     self.max_objects_per_image)
        self.image_ids = list(self.annotations.keys())
        self.clean_up_annotations_and_image_descriptions()

    def get_path_structure(self) -> Dict[str, str]:
        if self.split not in OPEN_IMAGES_STRUCTURE:
            raise ValueError(f'Split [{self.split} does not exist for Open Images data.]')
        return OPEN_IMAGES_STRUCTURE[self.split]

    def get_image_path(self, image_id: str) -> Path:
        return self.paths['files'].joinpath(f'{image_id:0>16}.jpg')

    def get_image_description(self, image_id: str) -> Dict[str, Any]:
        image_path = self.get_image_path(image_id)
        return {'file_path': str(image_path), 'file_name': image_path.name}
