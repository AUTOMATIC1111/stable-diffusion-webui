from io import BytesIO

import modules.scripts as scripts
import gradio as gr

from extra.fileStorage import ExtraFileStorage
from modules.shared import opts
from PIL import PngImagePlugin
import piexif


class Script(scripts.Script):
    def title(self):
        return f"Image Storage"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        checkbox_save_to_oss = gr.inputs.Checkbox(label="Save to Oss", default=False)
        checkbox_send_images_url = gr.inputs.Checkbox(label="Send Images Url", default=False)

        return [checkbox_save_to_oss, checkbox_send_images_url]

    def postprocess(self, p, processed, checkbox_save_to_oss, checkbox_send_images_url):
        if checkbox_save_to_oss:
            for i in range(len(processed.images)):
                output_bytes = BytesIO()
                image = processed.images[i]
                if opts.samples_format.lower() == 'png':
                    use_metadata = False
                    metadata = PngImagePlugin.PngInfo()
                    for key, value in image.info.items():
                        if isinstance(key, str) and isinstance(value, str):
                            metadata.add_text(key, value)
                            use_metadata = True
                    image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None),
                               quality=opts.jpeg_quality)

                elif opts.samples_format.lower() in ("jpg", "jpeg", "webp"):
                    parameters = image.info.get('parameters', None)
                    exif_bytes = piexif.dump({
                        "Exif": {piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "",
                                                                                            encoding="unicode")}
                    })
                    if opts.samples_format.lower() in ("jpg", "jpeg"):
                        image.save(output_bytes, format="JPEG", exif=exif_bytes, quality=opts.jpeg_quality)
                    else:
                        image.save(output_bytes, format="WEBP", exif=exif_bytes, quality=opts.jpeg_quality)
                else:
                    raise Exception("Invalid image format")

                bytes_data = output_bytes.getvalue()
                storage = ExtraFileStorage()
                url = storage.saveByte2Server(bytes_data, opts.samples_format.lower())
                if checkbox_send_images_url:
                    processed.images[i] = url
        return True
