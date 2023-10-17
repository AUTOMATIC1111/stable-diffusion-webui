from itertools import cycle
from typing import List, Tuple, Callable, Optional

from PIL import Image as pil_image, ImageDraw as pil_img_draw, ImageFont
from more_itertools.recipes import grouper
from taming.data.image_transforms import convert_pil_to_tensor
from torch import LongTensor, Tensor

from taming.data.helper_types import BoundingBox, Annotation
from taming.data.conditional_builder.objects_center_points import ObjectsCenterPointsConditionalBuilder
from taming.data.conditional_builder.utils import COLOR_PALETTE, WHITE, GRAY_75, BLACK, additional_parameters_string, \
    pad_list, get_plot_font_size, absolute_bbox


class ObjectsBoundingBoxConditionalBuilder(ObjectsCenterPointsConditionalBuilder):
    @property
    def object_descriptor_length(self) -> int:
        return 3

    def _make_object_descriptors(self, annotations: List[Annotation]) -> List[Tuple[int, ...]]:
        object_triples = [
            (self.object_representation(ann), *self.token_pair_from_bbox(ann.bbox))
            for ann in annotations
        ]
        empty_triple = (self.none, self.none, self.none)
        object_triples = pad_list(object_triples, empty_triple, self.no_max_objects)
        return object_triples

    def inverse_build(self, conditional: LongTensor) -> Tuple[List[Tuple[int, BoundingBox]], Optional[BoundingBox]]:
        conditional_list = conditional.tolist()
        crop_coordinates = None
        if self.encode_crop:
            crop_coordinates = self.bbox_from_token_pair(conditional_list[-2], conditional_list[-1])
            conditional_list = conditional_list[:-2]
        object_triples = grouper(conditional_list, 3)
        assert conditional.shape[0] == self.embedding_dim
        return [
            (object_triple[0], self.bbox_from_token_pair(object_triple[1], object_triple[2]))
            for object_triple in object_triples if object_triple[0] != self.none
        ], crop_coordinates

    def plot(self, conditional: LongTensor, label_for_category_no: Callable[[int], str], figure_size: Tuple[int, int],
             line_width: int = 3, font_size: Optional[int] = None) -> Tensor:
        plot = pil_image.new('RGB', figure_size, WHITE)
        draw = pil_img_draw.Draw(plot)
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/lato/Lato-Regular.ttf",
            size=get_plot_font_size(font_size, figure_size)
        )
        width, height = plot.size
        description, crop_coordinates = self.inverse_build(conditional)
        for (representation, bbox), color in zip(description, cycle(COLOR_PALETTE)):
            annotation = self.representation_to_annotation(representation)
            class_label = label_for_category_no(annotation.category_no) + ' ' + additional_parameters_string(annotation)
            bbox = absolute_bbox(bbox, width, height)
            draw.rectangle(bbox, outline=color, width=line_width)
            draw.text((bbox[0] + line_width, bbox[1] + line_width), class_label, anchor='la', fill=BLACK, font=font)
        if crop_coordinates is not None:
            draw.rectangle(absolute_bbox(crop_coordinates, width, height), outline=GRAY_75, width=line_width)
        return convert_pil_to_tensor(plot) / 127.5 - 1.
