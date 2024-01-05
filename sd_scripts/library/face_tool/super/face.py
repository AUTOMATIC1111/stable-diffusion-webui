import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis
from sklearn import preprocessing
from enum import IntEnum
from loguru import logger
from library.face_tool.filestorage.storage import PrivatizationFileStorage
from library.face_tool.data_models.face import ImgFaceQuality


class FaceQuality(IntEnum):
    NoFace = 1
    LowSimilarity = 2
    LowReso = 3
    MultiFace = 4
    OK = 10


class Gender(IntEnum):
    Girl = 1
    Boy = 2


MinFaceImgPixels = 250 * 250


def download_models():
    buffalo_l = os.path.join('models', 'buffalo_l')
    os.makedirs(buffalo_l, exist_ok=True)
    model_names = [
        '1k3d68.onnx',
        '2d106det.onnx',
        'det_10g.onnx',
        'genderage.onnx',
        'w600k_r50.onnx',
    ]
    base_url = 'https://xingzheassert.obs.cn-north-4.myhuaweicloud.com/face-analysis/buffalo_l/'
    fs = PrivatizationFileStorage()
    for m in model_names:
        local = os.path.join(buffalo_l, m)
        if not os.path.isfile(local):
            logger.info(f"download buffalo_l: {local}...")
            fs.download(base_url + m, local)


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


# 参数值：主图路径+训练集路径
# 返回值：{训练集图片1：type,训练集图片2：type}
# type：0-该图片中不存在人脸;1-存在跟主图的相似度太低的人脸;2-这张脸分辨率太低了;3-可用 分辨率且跟主图人脸相似
def insightface_face_recognition(main_path, train_path, use_gpu=False):
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], root='.')
    app.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
    faces, fq = insightface_main_face(main_path, use_gpu)
    quality = fq.quality
    if quality != FaceQuality.OK:
        raise OSError(f'main face quality err: {quality}({FaceQuality(quality).name})')
    embedding = np.array(faces[0].embedding).reshape((1, -1))
    embedding_ori = preprocessing.normalize(embedding)
    res = []
    for filename in os.listdir(train_path):
        full = os.path.join(train_path, filename)
        if full == main_path:
            continue

        img = cv2.imread(full)
        faces = app.get(img)
        # 分辨率-跟主图相似度高且分辨率大于300
        fq = ImgFaceQuality(
            file_name=os.path.basename(main_path),
            quality=FaceQuality.OK,
            gender=Gender(faces[0].gender + 1) if faces else 0,
            age=faces[0].age if faces else 0,
        )
        if not faces:
            fq.quality = FaceQuality.NoFace
        elif len(faces) > 1:
            fq.quality = FaceQuality.MultiFace
        else:
            embedding = np.array(faces[0].embedding).reshape((1, -1))
            embedding = preprocessing.normalize(embedding)
            diff = np.subtract(embedding_ori, embedding)
            dist = np.sum(np.square(diff), 1)[0]
            if dist >= 1.5:
                fq.quality = FaceQuality.LowSimilarity  # 这张脸跟主图的相似度太低了

            else:
                box = faces[0].bbox.astype(np.int)
                width, height = box[2] - box[0], box[3] - box[1]
                if width * height < MinFaceImgPixels:
                    fq.quality = FaceQuality.LowReso  # 这张脸分辨率太低了
        res.append(fq)

    return res


download_models()
