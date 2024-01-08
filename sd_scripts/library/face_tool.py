import cv2
import os
import numpy as np
from enum import IntEnum
from pydantic import BaseModel
from insightface.app import FaceAnalysis


class ImgFaceQuality(BaseModel):

    file_name: str
    quality: int
    gender: int
    age: int


class Gender(IntEnum):
    Girl = 1
    Boy = 2


class FaceQuality(IntEnum):
    NoFace = 1
    LowSimilarity = 2
    LowReso = 3
    MultiFace = 4
    OK = 10


MinFaceImgPixels = 250 * 250


def insightface_main_face(main_path, use_gpu=False):
    print("main_path:", main_path)
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], root='.')
    app.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
    img = cv2.imread(main_path)
    faces = app.get(img)
    # box=[ 346，643，756，1219] 图像从（346,643）到（756,1219）画一个矩形框
    fq = ImgFaceQuality(
        file_name=os.path.basename(main_path),
        quality=FaceQuality.OK,
        gender=Gender(faces[0].gender + 1) if faces else 0,
        age=faces[0].age if faces else 0,
    )
    if not faces:
        fq.quality = FaceQuality.NoFace
        return faces, fq
    if len(faces) > 1:
        fq.quality = FaceQuality.MultiFace
        return faces, fq

    box = faces[0].bbox.astype(np.int)
    width, height = box[2] - box[0], box[3] - box[1]
    if width * height < MinFaceImgPixels:
        fq.quality = FaceQuality.LowReso
        return faces, fq

    return faces, fq