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
        if output.size != image.size:
            output = output.resize(image.size, Image.Resampling.LANCZOS)
        if output.mode != image.mode:
            output = output.convert(image.mode)
        shared.log.debug(f'Testing processor: input={image} mode={image.mode} output={output} mode={output.mode}')
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
    from modules.control.units import controlnet
    if image is None:
        shared.log.error('Image not loaded')
        return None, None, None
    from PIL import ImageDraw, ImageFont
    images = []
    for model_id in controlnet.list_models():
        if model_id is None:
            model_id = 'None'
        if shared.state.interrupted:
            continue
        output = image
        if model_id != 'None':
            controlnet = controlnet.ControlNet(model_id=model_id, device=devices.device, dtype=devices.dtype)
            if controlnet is None:
                shared.log.error(f'ControlNet load failed: id="{model_id}"')
                continue
            shared.log.info(f'Testing ControlNet: {model_id}')
            pipe = controlnet.ControlNetPipeline(controlnet=controlnet.model, pipeline=shared.sd_model)
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
    from modules.control.units import t2iadapter
    if image is None:
        shared.log.error('Image not loaded')
        return None, None, None
    from PIL import ImageDraw, ImageFont
    images = []
    for model_id in t2iadapter.list_models():
        if model_id is None:
            model_id = 'None'
        if shared.state.interrupted:
            continue
        output = image.copy()
        if model_id != 'None':
            adapter = t2iadapter.Adapter(model_id=model_id, device=devices.device, dtype=devices.dtype)
            if adapter is None:
                shared.log.error(f'Adapter load failed: id="{model_id}"')
                continue
            shared.log.info(f'Testing Adapter: {model_id}')
            pipe = t2iadapter.AdapterPipeline(adapter=adapter.model, pipeline=shared.sd_model)
            pipe.pipeline.to(device=devices.device, dtype=devices.dtype)
            sd_models.set_diffuser_options(pipe)
            image = image.convert('L') if 'Canny' in model_id or 'Sketch' in model_id else image.convert('RGB')
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
    from modules.control.units import xs
    if image is None:
        shared.log.error('Image not loaded')
        return None, None, None
    from PIL import ImageDraw, ImageFont
    images = []
    for model_id in xs.list_models():
        if model_id is None:
            model_id = 'None'
        if shared.state.interrupted:
            continue
        output = image
        if model_id != 'None':
            xs = xs.ControlNetXS(model_id=model_id, device=devices.device, dtype=devices.dtype)
            if xs is None:
                shared.log.error(f'ControlNet-XS load failed: id="{model_id}"')
                continue
            shared.log.info(f'Testing ControlNet-XS: {model_id}')
            pipe = xs.ControlNetXSPipeline(controlnet=xs.model, pipeline=shared.sd_model)
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


def test_lite(prompt, negative, image):
    from modules import devices, sd_models
    from modules.control.units import lite
    if image is None:
        shared.log.error('Image not loaded')
        return None, None, None
    from PIL import ImageDraw, ImageFont
    images = []
    for model_id in lite.list_models():
        if model_id is None:
            model_id = 'None'
        if shared.state.interrupted:
            continue
        output = image
        if model_id != 'None':
            lite = lite.ControlLLLite(model_id=model_id, device=devices.device, dtype=devices.dtype)
            if lite is None:
                shared.log.error(f'Control-LLite load failed: id="{model_id}"')
                continue
            shared.log.info(f'Testing ControlNet-XS: {model_id}')
            pipe = lite.ControlLLitePipeline(pipeline=shared.sd_model)
            pipe.apply(controlnet=lite.model, image=image, conditioning=1.0)
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
