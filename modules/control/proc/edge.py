import warnings
import cv2
import numpy as np
from PIL import Image
from modules.control.util import HWC3, resize_image

ed = None
"""
    PFmode: bool
    EdgeDetectionOperator: int
    GradientThresholdValue: int
    AnchorThresholdValue: int
    ScanInterval: int
    MinPathLength: int
    Sigma: float
    SumFlag: bool
    NFAValidation: bool
    MinLineLength: int
    MaxDistanceBetweenTwoLines: float
    LineFitErrorThreshold: float
    MaxErrorThreshold: float
"""

class EdgeDetector:
    def __call__(self, input_image=None, pf=True, mode='edge', detect_resolution=512, image_resolution=512, output_type=None, **kwargs):
        global ed # pylint: disable=global-statement
        if ed is None:
            ed = cv2.ximgproc.createEdgeDrawing()
        params = cv2.ximgproc.EdgeDrawing.Params()
        params.PFmode = pf
        ed.setParams(params)
        if "img" in kwargs:
            warnings.warn("img is deprecated, please use `input_image=...` instead.", DeprecationWarning)
            input_image = kwargs.pop("img")
        if input_image is None:
            raise ValueError("input_image must be defined.")

        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
            output_type = output_type or "pil"
        else:
            output_type = output_type or "np"

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        img_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        edges    = ed.detectEdges(img_gray)
        if mode == 'edge':
            edge_map = ed.getEdgeImage(edges)
        else:
            edge_map = ed.getGradientImage(edges)
            edge_map = np.expand_dims(edge_map, axis=2)
            edge_map = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        edge_map = HWC3(edge_map)

        img = resize_image(input_image, image_resolution)
        H, W, _C = img.shape
        edge_map = cv2.resize(edge_map, (W, H), interpolation=cv2.INTER_LINEAR)

        if output_type == "pil":
            edge_map = Image.fromarray(edge_map)
            edge_map = edge_map.convert('L')

        return edge_map
