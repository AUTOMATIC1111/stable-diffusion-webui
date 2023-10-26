from typing import Dict, Tuple, Optional, NamedTuple, Union
from PIL.Image import Image as pil_image
from torch import Tensor

try:
  from typing import Literal
except ImportError:
  from typing_extensions import Literal

Image = Union[Tensor, pil_image]
BoundingBox = Tuple[float, float, float, float]  # x0, y0, w, h
CropMethodType = Literal['none', 'random', 'center', 'random-2d']
SplitType = Literal['train', 'validation', 'test']


class ImageDescription(NamedTuple):
    id: int
    file_name: str
    original_size: Tuple[int, int]  # w, h
    url: Optional[str] = None
    license: Optional[int] = None
    coco_url: Optional[str] = None
    date_captured: Optional[str] = None
    flickr_url: Optional[str] = None
    flickr_id: Optional[str] = None
    coco_id: Optional[str] = None


class Category(NamedTuple):
    id: str
    super_category: Optional[str]
    name: str


class Annotation(NamedTuple):
    area: float
    image_id: str
    bbox: BoundingBox
    category_no: int
    category_id: str
    id: Optional[int] = None
    source: Optional[str] = None
    confidence: Optional[float] = None
    is_group_of: Optional[bool] = None
    is_truncated: Optional[bool] = None
    is_occluded: Optional[bool] = None
    is_depiction: Optional[bool] = None
    is_inside: Optional[bool] = None
    segmentation: Optional[Dict] = None
