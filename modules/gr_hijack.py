from PIL import Image
import gradio as gr
import gradio.processing_utils


def gr_image_preprocess(self, x):
    if x is None:
        return x
    mask = None
    if isinstance(x, dict):
        x, mask = x["image"], x["mask"]
    im = gradio.processing_utils.decode_base64_to_image(x)
    im = im.convert(self.image_mode)
    if self.shape is not None:
        im = gradio.processing_utils.resize_and_crop(im, self.shape)
    if self.tool == "sketch" and self.source in ["upload"]:
        if mask is not None:
            mask_im = gradio.processing_utils.decode_base64_to_image(mask)
            if mask_im.mode == "RGBA":  # whiten any opaque pixels in the mask
                alpha_data = mask_im.getchannel("A").convert("L")
                mask_im = Image.merge("RGB", [alpha_data, alpha_data, alpha_data])
        else:
            mask_im = Image.new("L", im.size, 0)
        return { "image": self._format_image(im), "mask": self._format_image(mask_im) } # pylint: disable=protected-access
    return self._format_image(im) # pylint: disable=protected-access


def init():
    gr.components.Image.preprocess =  gr_image_preprocess
