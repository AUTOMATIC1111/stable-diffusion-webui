import os.path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context


def _load_tf_and_return_tags(pil_image, threshold):
    import deepdanbooru as dd
    import tensorflow as tf
    import numpy as np

    this_folder = os.path.dirname(__file__)
    model_path = os.path.abspath(os.path.join(this_folder, '..', 'models', 'deepbooru'))
    if not os.path.exists(os.path.join(model_path, 'project.json')):
        # there is no point importing these every time
        import zipfile
        from basicsr.utils.download_util import load_file_from_url
        load_file_from_url(r"https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip",
                           model_path)
        with zipfile.ZipFile(os.path.join(model_path, "deepdanbooru-v3-20211112-sgd-e28.zip"), "r") as zip_ref:
            zip_ref.extractall(model_path)
        os.remove(os.path.join(model_path, "deepdanbooru-v3-20211112-sgd-e28.zip"))

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


def subprocess_init_no_cuda():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_deepbooru_tags(pil_image, threshold=0.5):
    context = get_context('spawn')
    with ProcessPoolExecutor(initializer=subprocess_init_no_cuda, mp_context=context) as executor:
        f = executor.submit(_load_tf_and_return_tags, pil_image, threshold, )
        ret = f.result()  # will rethrow any exceptions
    return ret