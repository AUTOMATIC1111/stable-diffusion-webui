import math
from PIL import Image, ImageChops
from modules import shared, errors


def test_processors(image):
    from modules.control import processors
    if image is None:
        shared.log.error('Image not loaded')
        return None, None, None
    from PIL import ImageDraw, ImageFont
    images = []
    for processor_id in processors.list_models():
        if shared.state.interrupted:
            continue
        shared.log.info(f'Testing processor: {processor_id}')
        processor = processors.Processor(processor_id)
        output = image
        if processor is None:
            shared.log.error(f'Processor load failed: id="{processor_id}"')
            processor_id = f'{processor_id} error'
        else:
            output = processor(image)
            processor.reset()
        diff = ImageChops.difference(image, output)
        if not diff.getbbox():
            processor_id = f'{processor_id} null'
        draw = ImageDraw.Draw(output)
        font = ImageFont.truetype('DejaVuSansMono', 48)
        draw.text((10, 10), processor_id, (0,0,0), font=font)
        draw.text((8, 8), processor_id, (255,255,255), font=font)
        images.append(output)
        yield output, None, None, images
    rows = round(math.sqrt(len(images)))
    cols = math.ceil(len(images) / rows)
    w, h = 256, 256
    size = (cols * w + cols, rows * h + rows)
    grid = Image.new('RGB', size=size, color='black')
    shared.log.info(f'Test processors: images={len(images)} grid={grid}')
    for i, image in enumerate(images):
        x = (i % cols * w) + (i % cols)
        y = (i // cols * h) + (i // cols)
        thumb = image.copy().convert('RGB')
        thumb.thumbnail((w, h), Image.Resampling.HAMMING)
        grid.paste(thumb, box=(x, y))
    yield None, grid, None, images
    return None, grid, None, images # preview_process, output_image, output_video, output_gallery


def test_controlnets(prompt, negative, image):
    from modules import devices, sd_models
    from modules.control import controlnets
    if image is None:
        shared.log.error('Image not loaded')
        return None, None, None
    from PIL import ImageDraw, ImageFont
    images = []
    for model_id in controlnets.list_models():
        if model_id is None:
            model_id = 'None'
        if shared.state.interrupted:
            continue
        output = image
        if model_id != 'None':
            controlnet = controlnets.ControlNet(model_id=model_id, device=devices.device, dtype=devices.dtype)
            if controlnet is None:
                shared.log.error(f'ControlNet load failed: id="{model_id}"')
                continue
            shared.log.info(f'Testing ControlNet: {model_id}')
            pipe = controlnets.ControlNetPipeline(controlnet=controlnet.model, pipeline=shared.sd_model)
            pipe.pipeline.to(device=devices.device, dtype=devices.dtype)
            sd_models.set_diffuser_options(pipe)
            try:
                res = pipe.pipeline(prompt=prompt, negative_prompt=negative, image=image, num_inference_steps=10, output_type='pil')
                output = res.images[0]
            except Exception as e:
                errors.display(e, f'ControlNet {model_id} inference')
                model_id = f'{model_id} error'
            pipe.restore()
        draw = ImageDraw.Draw(output)
        font = ImageFont.truetype('DejaVuSansMono', 48)
        draw.text((10, 10), model_id, (0,0,0), font=font)
        draw.text((8, 8), model_id, (255,255,255), font=font)
        images.append(output)
        yield output, None, None, images
    rows = round(math.sqrt(len(images)))
    cols = math.ceil(len(images) / rows)
    w, h = 256, 256
    size = (cols * w + cols, rows * h + rows)
    grid = Image.new('RGB', size=size, color='black')
    shared.log.info(f'Test ControlNets: images={len(images)} grid={grid}')
    for i, image in enumerate(images):
        x = (i % cols * w) + (i % cols)
        y = (i // cols * h) + (i // cols)
        thumb = image.copy().convert('RGB')
        thumb.thumbnail((w, h), Image.Resampling.HAMMING)
        grid.paste(thumb, box=(x, y))
    yield None, grid, None, images
    return None, grid, None, images # preview_process, output_image, output_video, output_gallery


def test_adapters(prompt, negative, image):
    from modules import devices, sd_models
    from modules.control import adapters
    if image is None:
        shared.log.error('Image not loaded')
        return None, None, None
    from PIL import ImageDraw, ImageFont
    images = []
    for model_id in adapters.list_models():
        if model_id is None:
            model_id = 'None'
        if shared.state.interrupted:
            continue
        output = image
        if model_id != 'None':
            adapter = adapters.Adapter(model_id=model_id, device=devices.device, dtype=devices.dtype)
            if adapter is None:
                shared.log.error(f'Adapter load failed: id="{model_id}"')
                continue
            shared.log.info(f'Testing Adapter: {model_id}')
            pipe = adapters.AdapterPipeline(adapter=adapter.model, pipeline=shared.sd_model)
            pipe.pipeline.to(device=devices.device, dtype=devices.dtype)
            sd_models.set_diffuser_options(pipe)
            try:
                res = pipe.pipeline(prompt=prompt, negative_prompt=negative, image=image, num_inference_steps=10, output_type='pil')
                output = res.images[0]
            except Exception as e:
                errors.display(e, f'Adapter {model_id} inference')
                model_id = f'{model_id} error'
            pipe.restore()
        draw = ImageDraw.Draw(output)
        font = ImageFont.truetype('DejaVuSansMono', 48)
        draw.text((10, 10), model_id, (0,0,0), font=font)
        draw.text((8, 8), model_id, (255,255,255), font=font)
        images.append(output)
        yield output, None, None, images
    rows = round(math.sqrt(len(images)))
    cols = math.ceil(len(images) / rows)
    w, h = 256, 256
    size = (cols * w + cols, rows * h + rows)
    grid = Image.new('RGB', size=size, color='black')
    shared.log.info(f'Test Adapters: images={len(images)} grid={grid}')
    for i, image in enumerate(images):
        x = (i % cols * w) + (i % cols)
        y = (i // cols * h) + (i // cols)
        thumb = image.copy().convert('RGB')
        thumb.thumbnail((w, h), Image.Resampling.HAMMING)
        grid.paste(thumb, box=(x, y))
    yield None, grid, None, images
    return None, grid, None, images # preview_process, output_image, output_video, output_gallery


def test_xs(prompt, negative, image):
    from modules import devices, sd_models
    from modules.control import controlnetsxs
    if image is None:
        shared.log.error('Image not loaded')
        return None, None, None
    from PIL import ImageDraw, ImageFont
    images = []
    for model_id in controlnetsxs.list_models():
        if model_id is None:
            model_id = 'None'
        if shared.state.interrupted:
            continue
        output = image
        if model_id != 'None':
            xs = controlnetsxs.ControlNetXS(model_id=model_id, device=devices.device, dtype=devices.dtype)
            if xs is None:
                shared.log.error(f'ControlNet-XS load failed: id="{model_id}"')
                continue
            shared.log.info(f'Testing ControlNet-XS: {model_id}')
            pipe = controlnetsxs.ControlNetXSPipeline(controlnet=xs.model, pipeline=shared.sd_model)
            pipe.pipeline.to(device=devices.device, dtype=devices.dtype)
            sd_models.set_diffuser_options(pipe)
            try:
                res = pipe.pipeline(prompt=prompt, negative_prompt=negative, image=image, num_inference_steps=10, output_type='pil')
                output = res.images[0]
            except Exception as e:
                errors.display(e, f'ControlNet-XS {model_id} inference')
                model_id = f'{model_id} error'
            pipe.restore()
        draw = ImageDraw.Draw(output)
        font = ImageFont.truetype('DejaVuSansMono', 48)
        draw.text((10, 10), model_id, (0,0,0), font=font)
        draw.text((8, 8), model_id, (255,255,255), font=font)
        images.append(output)
        yield output, None, None, images
    rows = round(math.sqrt(len(images)))
    cols = math.ceil(len(images) / rows)
    w, h = 256, 256
    size = (cols * w + cols, rows * h + rows)
    grid = Image.new('RGB', size=size, color='black')
    shared.log.info(f'Test ControlNet-XS: images={len(images)} grid={grid}')
    for i, image in enumerate(images):
        x = (i % cols * w) + (i % cols)
        y = (i // cols * h) + (i // cols)
        thumb = image.copy().convert('RGB')
        thumb.thumbnail((w, h), Image.Resampling.HAMMING)
        grid.paste(thumb, box=(x, y))
    yield None, grid, None, images
    return None, grid, None, images # preview_process, output_image, output_video, output_gallery
