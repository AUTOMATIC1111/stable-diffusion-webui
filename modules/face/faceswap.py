from typing import List
import os
import cv2
import numpy as np
import huggingface_hub as hf
from PIL import Image
from modules import processing, shared, devices


debug = shared.log.trace if os.environ.get('SD_FACE_DEBUG', None) is not None else lambda *args, **kwargs: None
insightface_app = None
swapper = None


def face_swap(p: processing.StableDiffusionProcessing, app, input_images: List[Image.Image], source_image: Image.Image, cache: bool):
    import insightface.model_zoo
    global swapper # pylint: disable=global-statement
    if swapper is None:
        model_path = hf.hf_hub_download(repo_id='ezioruan/inswapper_128.onnx', filename='inswapper_128.onnx', cache_dir=shared.opts.diffusers_dir)
        router: insightface.model_zoo.model_zoo.INSwapper = insightface.model_zoo.model_zoo.ModelRouter(model_path)
        swapper = router.get_model()

    np_image = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)
    faces = app.get(np_image)
    source_face = faces[0]
    processed_images = []
    for image in input_images:
        np_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = app.get(np_image)
        for i, face in enumerate(faces):
            debug(f'FaceSwap: face={i} source={source_face.bbox} target={face.bbox}')
            np_image = swapper.get(img=np_image, target_face=face, source_face=source_face, paste_back=True) # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        p.extra_generation_params["FaceSwap"] = f'{len(faces)}'
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        processed_images.append(Image.fromarray(np_image))

    if not cache:
        swapper = None
    devices.torch_gc()

    return processed_images
