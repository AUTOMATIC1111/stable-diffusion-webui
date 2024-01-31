import os
from copy import copy
from typing import Tuple, List

from modules import img2img, processing, shared, script_callbacks
from scripts import external_code
from scripts.enums import InputMode

class BatchHijack:
    def __init__(self):
        self.is_batch = False
        self.batch_index = 0
        self.batch_size = 1
        self.init_seed = None
        self.init_subseed = None
        self.process_batch_callbacks = [self.on_process_batch]
        self.process_batch_each_callbacks = []
        self.postprocess_batch_each_callbacks = [self.on_postprocess_batch_each]
        self.postprocess_batch_callbacks = [self.on_postprocess_batch]

    def img2img_process_batch_hijack(self, p, *args, **kwargs):
        cn_is_batch, batches, output_dir, _ = get_cn_batches(p)
        if not cn_is_batch:
            return getattr(img2img, '__controlnet_original_process_batch')(p, *args, **kwargs)

        self.dispatch_callbacks(self.process_batch_callbacks, p, batches, output_dir)

        try:
            return getattr(img2img, '__controlnet_original_process_batch')(p, *args, **kwargs)
        finally:
            self.dispatch_callbacks(self.postprocess_batch_callbacks, p)

    def processing_process_images_hijack(self, p, *args, **kwargs):
        if self.is_batch:
            # we are in img2img batch tab, do a single batch iteration
            return self.process_images_cn_batch(p, *args, **kwargs)

        cn_is_batch, batches, output_dir, input_file_names = get_cn_batches(p)
        if not cn_is_batch:
            # we are not in batch mode, fallback to original function
            return getattr(processing, '__controlnet_original_process_images_inner')(p, *args, **kwargs)

        output_images = []
        try:
            self.dispatch_callbacks(self.process_batch_callbacks, p, batches, output_dir)

            for batch_i in range(self.batch_size):
                processed = self.process_images_cn_batch(p, *args, **kwargs)
                if shared.opts.data.get('controlnet_show_batch_images_in_ui', False):
                    output_images.extend(processed.images[processed.index_of_first_image:])

                if output_dir:
                    self.save_images(output_dir, input_file_names[batch_i], processed.images[processed.index_of_first_image:])

                if shared.state.interrupted:
                    break

        finally:
            self.dispatch_callbacks(self.postprocess_batch_callbacks, p)

        if output_images:
            processed.images = output_images
        else:
            processed = processing.Processed(p, [], p.seed)

        return processed

    def process_images_cn_batch(self, p, *args, **kwargs):
        self.dispatch_callbacks(self.process_batch_each_callbacks, p)
        old_detectmap_output = shared.opts.data.get('control_net_no_detectmap', False)
        try:
            shared.opts.data.update({'control_net_no_detectmap': True})
            processed = getattr(processing, '__controlnet_original_process_images_inner')(p, *args, **kwargs)
        finally:
            shared.opts.data.update({'control_net_no_detectmap': old_detectmap_output})

        self.dispatch_callbacks(self.postprocess_batch_each_callbacks, p, processed)

        # do not go past control net batch size
        if self.batch_index >= self.batch_size:
            shared.state.interrupted = True

        return processed

    def save_images(self, output_dir, init_image_path, output_images):
        os.makedirs(output_dir, exist_ok=True)
        for n, processed_image in enumerate(output_images):
            filename = os.path.basename(init_image_path)

            if n > 0:
                left, right = os.path.splitext(filename)
                filename = f"{left}-{n}{right}"

            if processed_image.mode == 'RGBA':
                processed_image = processed_image.convert("RGB")
            processed_image.save(os.path.join(output_dir, filename))

    def do_hijack(self):
        script_callbacks.on_script_unloaded(self.undo_hijack)
        hijack_function(
            module=img2img,
            name='process_batch',
            new_name='__controlnet_original_process_batch',
            new_value=self.img2img_process_batch_hijack,
        )
        hijack_function(
            module=processing,
            name='process_images_inner',
            new_name='__controlnet_original_process_images_inner',
            new_value=self.processing_process_images_hijack
        )

    def undo_hijack(self):
        unhijack_function(
            module=img2img,
            name='process_batch',
            new_name='__controlnet_original_process_batch',
        )
        unhijack_function(
            module=processing,
            name='process_images_inner',
            new_name='__controlnet_original_process_images_inner',
        )

    def adjust_job_count(self, p):
        if shared.state.job_count == -1:
            shared.state.job_count = p.n_iter
        shared.state.job_count *= self.batch_size

    def on_process_batch(self, p, batches, output_dir, *args):
        print('controlnet batch mode')
        self.is_batch = True
        self.batch_index = 0
        self.batch_size = len(batches)
        processing.fix_seed(p)
        if shared.opts.data.get('controlnet_increment_seed_during_batch', False):
            self.init_seed = p.seed
            self.init_subseed = p.subseed
        self.adjust_job_count(p)
        p.do_not_save_grid = True
        p.do_not_save_samples = bool(output_dir)

    def on_postprocess_batch_each(self, p, *args):
        self.batch_index += 1
        if shared.opts.data.get('controlnet_increment_seed_during_batch', False):
            p.seed = p.seed + len(p.all_prompts)
            p.subseed = p.subseed + len(p.all_prompts)

    def on_postprocess_batch(self, p, *args):
        self.is_batch = False
        self.batch_index = 0
        self.batch_size = 1
        if shared.opts.data.get('controlnet_increment_seed_during_batch', False):
            p.seed = self.init_seed
            p.all_seeds = [self.init_seed]
            p.subseed = self.init_subseed
            p.all_subseeds = [self.init_subseed]

    def dispatch_callbacks(self, callbacks, *args):
        for callback in callbacks:
            callback(*args)


def hijack_function(module, name, new_name, new_value):
    # restore original function in case of reload
    unhijack_function(module=module, name=name, new_name=new_name)
    setattr(module, new_name, getattr(module, name))
    setattr(module, name, new_value)


def unhijack_function(module, name, new_name):
    if hasattr(module, new_name):
        setattr(module, name, getattr(module, new_name))
        delattr(module, new_name)


def get_cn_batches(p: processing.StableDiffusionProcessing) -> Tuple[bool, List[List[str]], str, List[str]]:
    units = external_code.get_all_units_in_processing(p)
    units = [copy(unit) for unit in units if getattr(unit, 'enabled', False)]
    any_unit_is_batch = False
    output_dir = ''
    input_file_names = []
    for unit in units:
        if getattr(unit, 'input_mode', InputMode.SIMPLE) == InputMode.BATCH:
            any_unit_is_batch = True
            output_dir = getattr(unit, 'output_dir', '')
            if isinstance(unit.batch_images, str):
                unit.batch_images = shared.listfiles(unit.batch_images)
                input_file_names = unit.batch_images

    if any_unit_is_batch:
        cn_batch_size = min(len(getattr(unit, 'batch_images', []))
                         for unit in units
                         if getattr(unit, 'input_mode', InputMode.SIMPLE) == InputMode.BATCH)
    else:
        cn_batch_size = 1

    batches = [[] for _ in range(cn_batch_size)]
    for i in range(cn_batch_size):
        for unit in units:
            input_mode = getattr(unit, 'input_mode', InputMode.SIMPLE)
            if input_mode == InputMode.BATCH:
                batches[i].append(unit.batch_images[i])
            else:
                batches[i].append(unit.image)

    return any_unit_is_batch, batches, output_dir, input_file_names


instance = BatchHijack()
