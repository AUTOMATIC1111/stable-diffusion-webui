import os.path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import deepdanbooru as dd
import tensorflow as tf


def _load_tf_and_return_tags(pil_image, threshold):
    this_folder = os.path.dirname(__file__)
    model_path = os.path.join(this_folder, '..', 'models', 'deepbooru', 'deepdanbooru-v3-20211112-sgd-e28')

    model_good = False
    for path_candidate in [model_path, os.path.dirname(model_path)]:
        if os.path.exists(os.path.join(path_candidate, 'project.json')):
            model_path = path_candidate
            model_good = True
    if not model_good:
        return ("Download https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/"
                "deepdanbooru-v3-20211112-sgd-e28.zip unpack and put into models/deepbooru")

    tags = dd.project.load_tags_from_project(model_path)
    model = dd.project.load_model_from_project(
        model_path, compile_model=True
    )

    width = model.input_shape[2]
    height = model.input_shape[1]
    image = np.array(pil_image)
    image = tf.image.resize(
        image,
        size=(height, width),
        method=tf.image.ResizeMethod.AREA,
        preserve_aspect_ratio=True,
    )
    image = image.numpy()  # EagerTensor to np.array
    image = dd.image.transform_and_pad_image(image, width, height)
    image = image / 255.0
    image_shape = image.shape
    image = image.reshape((1, image_shape[0], image_shape[1], image_shape[2]))

    y = model.predict(image)[0]

    result_dict = {}

    for i, tag in enumerate(tags):
        result_dict[tag] = y[i]
    result_tags_out = []
    result_tags_print = []
    for tag in tags:
        if result_dict[tag] >= threshold:
            if tag.startswith("rating:"):
                continue
            result_tags_out.append(tag)
            result_tags_print.append(f'{result_dict[tag]} {tag}')

    print('\n'.join(sorted(result_tags_print, reverse=True)))

    return ', '.join(result_tags_out).replace('_', ' ').replace(':', ' ')


def get_deepbooru_tags(pil_image, threshold=0.5):
    with ProcessPoolExecutor() as executor:
        f = executor.submit(_load_tf_and_return_tags, pil_image, threshold)
        ret = f.result()  # will rethrow any exceptions
    return ret