import math
from PIL import Image
from modules.control import processors # patrickvonplaten controlnet_aux
from modules import shared


def test_processors(image):
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
        if processor is None:
            shared.log.error(f'Processor load failed: id="{processor_id}"')
            continue
        output = processor(image)
        processor.reset()
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
