import math
import random
import warnings
from itertools import cycle
from typing import List, Optional, Tuple, Callable

from PIL import Image as pil_image, ImageDraw as pil_img_draw, ImageFont
from more_itertools.recipes import grouper
from taming.data.conditional_builder.utils import COLOR_PALETTE, WHITE, GRAY_75, BLACK, FULL_CROP, filter_annotations, \
    additional_parameters_string, horizontally_flip_bbox, pad_list, get_circle_size, get_plot_font_size, \
    absolute_bbox, rescale_annotations
from taming.data.helper_types import BoundingBox, Annotation
from taming.data.image_transforms import convert_pil_to_tensor
from torch import LongTensor, Tensor


class ObjectsCenterPointsConditionalBuilder:
    def __init__(self, no_object_classes: int, no_max_objects: int, no_tokens: int, encode_crop: bool,
                 use_group_parameter: bool, use_additional_parameters: bool):
        self.no_object_classes = no_object_classes
        self.no_max_objects = no_max_objects
        self.no_tokens = no_tokens
        self.encode_crop = encode_crop
        self.no_sections = int(math.sqrt(self.no_tokens))
        self.use_group_parameter = use_group_parameter
        self.use_additional_parameters = use_additional_parameters

    @property
    def none(self) -> int:
        return self.no_tokens - 1

    @property
    def object_descriptor_length(self) -> int:
        return 2

    @property
    def embedding_dim(self) -> int:
        extra_length = 2 if self.encode_crop else 0
        return self.no_max_objects * self.object_descriptor_length + extra_length

    def tokenize_coordinates(self, x: float, y: float) -> int:
        """
        Express 2d coordinates with one number.
        Example: assume self.no_tokens = 16, then no_sections = 4:
        0  0  0  0
        0  0  #  0
        0  0  0  0
        0  0  0  x
        Then the # position corresponds to token 6, the x position to token 15.
        @param x: float in [0, 1]
        @param y: float in [0, 1]
        @return: discrete tokenized coordinate
        """
        x_discrete = int(round(x * (self.no_sections - 1)))
        y_discrete = int(round(y * (self.no_sections - 1)))
        return y_discrete * self.no_sections + x_discrete

    def coordinates_from_token(self, token: int) -> (float, float):
        x = token % self.no_sections
        y = token // self.no_sections
        return x / (self.no_sections - 1), y / (self.no_sections - 1)

    def bbox_from_token_pair(self, token1: int, token2: int) -> BoundingBox:
        x0, y0 = self.coordinates_from_token(token1)
        x1, y1 = self.coordinates_from_token(token2)
        return x0, y0, x1 - x0, y1 - y0

    def token_pair_from_bbox(self, bbox: BoundingBox) -> Tuple[int, int]:
        return self.tokenize_coordinates(bbox[0], bbox[1]), \
               self.tokenize_coordinates(bbox[0] + bbox[2], bbox[1] + bbox[3])

    def inverse_build(self, conditional: LongTensor) \
            -> Tuple[List[Tuple[int, Tuple[float, float]]], Optional[BoundingBox]]:
        conditional_list = conditional.tolist()
        crop_coordinates = None
        if self.encode_crop:
            crop_coordinates = self.bbox_from_token_pair(conditional_list[-2], conditional_list[-1])
            conditional_list = conditional_list[:-2]
        table_of_content = grouper(conditional_list, self.object_descriptor_length)
        assert conditional.shape[0] == self.embedding_dim
        return [
            (object_tuple[0], self.coordinates_from_token(object_tuple[1]))
            for object_tuple in table_of_content if object_tuple[0] != self.none
        ], crop_coordinates

    def plot(self, conditional: LongTensor, label_for_category_no: Callable[[int], str], figure_size: Tuple[int, int],
             line_width: int = 3, font_size: Optional[int] = None) -> Tensor:
        plot = pil_image.new('RGB', figure_size, WHITE)
        draw = pil_img_draw.Draw(plot)
        circle_size = get_circle_size(figure_size)
        font = ImageFont.truetype('/usr/share/fonts/truetype/lato/Lato-Regular.ttf',
                                  size=get_plot_font_size(font_size, figure_size))
        width, height = plot.size
        description, crop_coordinates = self.inverse_build(conditional)
        for (representation, (x, y)), color in zip(description, cycle(COLOR_PALETTE)):
            x_abs, y_abs = x * width, y * height
            ann = self.representation_to_annotation(representation)
            label = label_for_category_no(ann.category_no) + ' ' + additional_parameters_string(ann)
            ellipse_bbox = [x_abs - circle_size, y_abs - circle_size, x_abs + circle_size, y_abs + circle_size]
            draw.ellipse(ellipse_bbox, fill=color, width=0)
            draw.text((x_abs, y_abs), label, anchor='md', fill=BLACK, font=font)
        if crop_coordinates is not None:
            draw.rectangle(absolute_bbox(crop_coordinates, width, height), outline=GRAY_75, width=line_width)
        return convert_pil_to_tensor(plot) / 127.5 - 1.

    def object_representation(self, annotation: Annotation) -> int:
        modifier = 0
        if self.use_group_parameter:
            modifier |= 1 * (annotation.is_group_of is True)
        if self.use_additional_parameters:
            modifier |= 2 * (annotation.is_occluded is True)
            modifier |= 4 * (annotation.is_depiction is True)
            modifier |= 8 * (annotation.is_inside is True)
        return annotation.category_no + self.no_object_classes * modifier

    def representation_to_annotation(self, representation: int) -> Annotation:
        category_no = representation % self.no_object_classes
        modifier = representation // self.no_object_classes
        # noinspection PyTypeChecker
        return Annotation(
            area=None, image_id=None, bbox=None, category_id=None, id=None, source=None, confidence=None,
            category_no=category_no,
            is_group_of=bool((modifier & 1) * self.use_group_parameter),
            is_occluded=bool((modifier & 2) * self.use_additional_parameters),
            is_depiction=bool((modifier & 4) * self.use_additional_parameters),
            is_inside=bool((modifier & 8) * self.use_additional_parameters)
        )

    def _crop_encoder(self, crop_coordinates: BoundingBox) -> List[int]:
        return list(self.token_pair_from_bbox(crop_coordinates))

    def _make_object_descriptors(self, annotations: List[Annotation]) -> List[Tuple[int, ...]]:
        object_tuples = [
            (self.object_representation(a),
             self.tokenize_coordinates(a.bbox[0] + a.bbox[2] / 2, a.bbox[1] + a.bbox[3] / 2))
            for a in annotations
        ]
        empty_tuple = (self.none, self.none)
        object_tuples = pad_list(object_tuples, empty_tuple, self.no_max_objects)
        return object_tuples

    def build(self, annotations: List, crop_coordinates: Optional[BoundingBox] = None, horizontal_flip: bool = False) \
            -> LongTensor:
        if len(annotations) == 0:
            warnings.warn('Did not receive any annotations.')
        if len(annotations) > self.no_max_objects:
            warnings.warn('Received more annotations than allowed.')
            annotations = annotations[:self.no_max_objects]

        if not crop_coordinates:
            crop_coordinates = FULL_CROP

        random.shuffle(annotations)
        annotations = filter_annotations(annotations, crop_coordinates)
        if self.encode_crop:
            annotations = rescale_annotations(annotations, FULL_CROP, horizontal_flip)
            if horizontal_flip:
                crop_coordinates = horizontally_flip_bbox(crop_coordinates)
            extra = self._crop_encoder(crop_coordinates)
        else:
            annotations = rescale_annotations(annotations, crop_coordinates, horizontal_flip)
            extra = []

        object_tuples = self._make_object_descriptors(annotations)
        flattened = [token for tuple_ in object_tuples for token in tuple_] + extra
        assert len(flattened) == self.embedding_dim
        assert all(0 <= value < self.no_tokens for value in flattened)
        return LongTensor(flattened)
