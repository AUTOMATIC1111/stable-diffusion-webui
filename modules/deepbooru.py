import os.path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import re

re_special = re.compile(r'([\\()])')

def get_deepbooru_tags(pil_image):
    """
    This method is for running only one image at a time for simple use.  Used to the img2img interrogate.
    """
    from modules import shared  # prevents circular reference

    try:
        create_deepbooru_process(shared.opts.interrogate_deepbooru_score_threshold, create_deepbooru_opts())
        return get_tags_from_process(pil_image)
    finally:
        release_process()


OPT_INCLUDE_RANKS = "include_ranks"
def create_deepbooru_opts():
    from modules import shared

    return {
        "use_spaces": shared.opts.deepbooru_use_spaces,
        "use_escape": shared.opts.deepbooru_escape,
        "alpha_sort": shared.opts.deepbooru_sort_alpha,
        OPT_INCLUDE_RANKS: shared.opts.interrogate_return_ranks,
    }


def deepbooru_process(queue, deepbooru_process_return, threshold, deepbooru_opts):
    model, tags = get_deepbooru_tags_model()
    while True: # while process is running, keep monitoring queue for new image
        pil_image = queue.get()
        if pil_image == "QUIT":
            break
        else:
            deepbooru_process_return["value"] = get_deepbooru_tags_from_model(model, tags, pil_image, threshold, deepbooru_opts)


def create_deepbooru_process(threshold, deepbooru_opts):
    """
    Creates deepbooru process.  A queue is created to send images into the process.  This enables multiple images
    to be processed in a row without reloading the model or creating a new process.  To return the data, a shared
    dictionary is created to hold the tags created.  To wait for tags to be returned, a value of -1 is assigned
    to the dictionary and the method adding the image to the queue should wait for this value to be updated with
    the tags.
    """
    from modules import shared  # prevents circular reference
    shared.deepbooru_process_manager = multiprocessing.Manager()
    shared.deepbooru_process_queue = shared.deepbooru_process_manager.Queue()
    shared.deepbooru_process_return = shared.deepbooru_process_manager.dict()
    shared.deepbooru_process_return["value"] = -1
    shared.deepbooru_process = multiprocessing.Process(target=deepbooru_process, args=(shared.deepbooru_process_queue, shared.deepbooru_process_return, threshold, deepbooru_opts))
    shared.deepbooru_process.start()


def get_tags_from_process(image):
    from modules import shared

    shared.deepbooru_process_return["value"] = -1
    shared.deepbooru_process_queue.put(image)
    while shared.deepbooru_process_return["value"] == -1:
        time.sleep(0.2)
    caption = shared.deepbooru_process_return["value"]
    shared.deepbooru_process_return["value"] = -1

    return caption


def release_process():
    """
    Stops the deepbooru process to return used memory
    """
    from modules import shared  # prevents circular reference
    shared.deepbooru_process_queue.put("QUIT")
    shared.deepbooru_process.join()
    shared.deepbooru_process_queue = None
    shared.deepbooru_process = None
    shared.deepbooru_process_return = None
    shared.deepbooru_process_manager = None

def get_deepbooru_tags_model():
    import deepdanbooru as dd
    import tensorflow as tf
    import numpy as np
    this_folder = os.path.dirname(__file__)
    model_path = os.path.abspath(os.path.join(this_folder, '..', 'models', 'deepbooru'))
    if not os.path.exists(os.path.join(model_path, 'project.json')):
        # there is no point importing these every time
        import zipfile
        from basicsr.utils.download_util import load_file_from_url
        load_file_from_url(
            r"https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip",
            model_path)
        with zipfile.ZipFile(os.path.join(model_path, "deepdanbooru-v3-20211112-sgd-e28.zip"), "r") as zip_ref:
            zip_ref.extractall(model_path)
        os.remove(os.path.join(model_path, "deepdanbooru-v3-20211112-sgd-e28.zip"))

    tags = dd.project.load_tags_from_project(model_path)
    model = dd.project.load_model_from_project(
        model_path, compile_model=True
    )
    return model, tags


def get_deepbooru_tags_from_model(model, tags, pil_image, threshold, deepbooru_opts):
    import deepdanbooru as dd
    import tensorflow as tf
    import numpy as np

    alpha_sort = deepbooru_opts['alpha_sort']
    use_spaces = deepbooru_opts['use_spaces']
    use_escape = deepbooru_opts['use_escape']
    include_ranks = deepbooru_opts['include_ranks']

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

    unsorted_tags_in_theshold = []
    result_tags_print = []
    for tag in tags:
        if result_dict[tag] >= threshold:
            if tag.startswith("rating:"):
                continue
            unsorted_tags_in_theshold.append((result_dict[tag], tag))
            result_tags_print.append(f'{result_dict[tag]} {tag}')

    # sort tags
    result_tags_out = []
    sort_ndx = 0
    if alpha_sort:
        sort_ndx = 1

    # sort by reverse by likelihood and normal for alpha, and format tag text as requested
    unsorted_tags_in_theshold.sort(key=lambda y: y[sort_ndx], reverse=(not alpha_sort))
    for weight, tag in unsorted_tags_in_theshold:
        # note: tag_outformat will still have a colon if include_ranks is True
        tag_outformat = tag.replace(':', ' ')
        if use_spaces:
            tag_outformat = tag_outformat.replace('_', ' ')
        if use_escape:
            tag_outformat = re.sub(re_special, r'\\\1', tag_outformat)
        if include_ranks:
            tag_outformat = f"({tag_outformat}:{weight:.3f})"

        result_tags_out.append(tag_outformat)

    print('\n'.join(sorted(result_tags_print, reverse=True)))

    return ', '.join(result_tags_out)
