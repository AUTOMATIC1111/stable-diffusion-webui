from typing import Union
from PIL import Image
import gradio as gr
from modules.shared import log
from modules.control import processors
from modules.control.units import controlnet
from modules.control.units import xs
from modules.control.units import lite
from modules.control.units import t2iadapter
from modules.control.units import reference # pylint: disable=unused-import


default_device = None
default_dtype = None


class Unit(): # mashup of gradio controls and mapping to actual implementation classes
    def __init__(self,
                 # values
                 enabled: bool = None,
                 strength: float = None,
                 unit_type: str = None,
                 start: float = 0,
                 end: float = 1,
                 # ui bindings
                 enabled_cb = None,
                 reset_btn = None,
                 process_id = None,
                 preview_btn = None,
                 model_id = None,
                 model_strength = None,
                 image_input = None,
                 preview_process = None,
                 image_upload = None,
                 image_preview = None,
                 control_start = None,
                 control_end = None,
                 result_txt = None,
                 extra_controls: list = [], # noqa B006
        ):
        self.enabled = enabled or False
        self.type = unit_type
        self.strength = strength or 1.0
        self.start = start or 0
        self.end = end or 1
        self.start = min(self.start, self.end)
        self.end = max(self.start, self.end)
        # processor always exists, adapter and controlnet are optional
        self.process: processors.Processor = processors.Processor()
        self.adapter: t2iadapter.Adapter = None
        self.controlnet: Union[controlnet.ControlNet, xs.ControlNetXS] = None
        # map to input image
        self.input: Image = image_input
        self.override: Image = None
        # global settings but passed per-unit
        self.factor = 1.0
        self.guess = False
        self.start = 0
        self.end = 1
        # reference settings
        self.attention = 'Attention'
        self.fidelity = 0.5
        self.query_weight = 1.0
        self.adain_weight = 1.0

        def reset():
            if self.process is not None:
                self.process.reset()
            if self.adapter is not None:
                self.adapter.reset()
            if self.controlnet is not None:
                self.controlnet.reset()
            self.override = None
            return [True, 'None', 'None', 1.0] # reset ui values

        def enabled_change(val):
            self.enabled = val

        def strength_change(val):
            self.strength = val

        def control_change(start, end):
            self.start = min(start, end)
            self.end = max(start, end)

        def adapter_extra(c1):
            self.factor = c1

        def controlnet_extra(c1):
            self.guess = c1

        def controlnetxs_extra(_c1):
            pass # gr.component passed directly to load method

        def reference_extra(c1, c2, c3, c4):
            self.attention = c1
            self.fidelity = c2
            self.query_weight = c3
            self.adain_weight = c4

        def upload_image(image_file):
            try:
                self.process.override = Image.open(image_file.name)
                self.override = self.process.override
                log.debug(f'Control process upload image: path="{image_file.name}" image={self.process.override}')
                return gr.update(visible=self.process.override is not None, value=self.process.override)
            except Exception as e:
                log.error(f'Control process upload image failed: path="{image_file.name}" error={e}')
                return gr.update(visible=False, value=None)

        # actual init
        if self.type == 'adapter':
            self.adapter = t2iadapter.Adapter(device=default_device, dtype=default_dtype)
        elif self.type == 'controlnet':
            self.controlnet = controlnet.ControlNet(device=default_device, dtype=default_dtype)
        elif self.type == 'xs':
            self.controlnet = xs.ControlNetXS(device=default_device, dtype=default_dtype)
        elif self.type == 'lite':
            self.controlnet = lite.ControlLLLite(device=default_device, dtype=default_dtype)
        elif self.type == 'reference':
            pass
        elif self.type == 'ip':
            pass
        else:
            log.error(f'Control unknown type: unit={unit_type}')
            return

        # bind ui controls to properties if present
        if self.type == 'adapter':
            if model_id is not None:
                model_id.change(fn=self.adapter.load, inputs=[model_id], outputs=[result_txt], show_progress=True)
            if extra_controls is not None and len(extra_controls) > 0:
                extra_controls[0].change(fn=adapter_extra, inputs=extra_controls)
        elif self.type == 'controlnet':
            if model_id is not None:
                model_id.change(fn=self.controlnet.load, inputs=[model_id], outputs=[result_txt], show_progress=True)
            if extra_controls is not None and len(extra_controls) > 0:
                extra_controls[0].change(fn=controlnet_extra, inputs=extra_controls)
        elif self.type == 'xs':
            if model_id is not None:
                model_id.change(fn=self.controlnet.load, inputs=[model_id, extra_controls[0]], outputs=[result_txt], show_progress=True)
            if extra_controls is not None and len(extra_controls) > 0:
                extra_controls[0].change(fn=controlnetxs_extra, inputs=extra_controls)
        elif self.type == 'lite':
            if model_id is not None:
                model_id.change(fn=self.controlnet.load, inputs=[model_id], outputs=[result_txt], show_progress=True)
            if extra_controls is not None and len(extra_controls) > 0:
                extra_controls[0].change(fn=controlnetxs_extra, inputs=extra_controls)
        elif self.type == 'reference':
            if extra_controls is not None and len(extra_controls) > 0:
                extra_controls[0].change(fn=reference_extra, inputs=extra_controls)
                extra_controls[1].change(fn=reference_extra, inputs=extra_controls)
                extra_controls[2].change(fn=reference_extra, inputs=extra_controls)
                extra_controls[3].change(fn=reference_extra, inputs=extra_controls)
        if enabled_cb is not None:
            enabled_cb.change(fn=enabled_change, inputs=[enabled_cb])
        if model_strength is not None:
            model_strength.change(fn=strength_change, inputs=[model_strength])
        if process_id is not None:
            process_id.change(fn=self.process.load, inputs=[process_id], outputs=[result_txt], show_progress=True)
        if reset_btn is not None:
            reset_btn.click(fn=reset, inputs=[], outputs=[enabled_cb, model_id, process_id, model_strength])
        if preview_btn is not None:
            preview_btn.click(fn=self.process.preview, inputs=[self.input], outputs=[preview_process]) # return list of images for gallery
        if image_upload is not None:
            image_upload.upload(fn=upload_image, inputs=[image_upload], outputs=[image_preview]) # return list of images for gallery
        if control_start is not None and control_end is not None:
            control_start.change(fn=control_change, inputs=[control_start, control_end])
            control_end.change(fn=control_change, inputs=[control_start, control_end])
