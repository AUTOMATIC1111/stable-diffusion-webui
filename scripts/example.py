import gradio as gr
from diffusers.pipelines import StableDiffusionPipeline, StableDiffusionXLPipeline # pylint: disable=unused-import
from modules import shared, scripts, processing, sd_models, devices

"""
This is a simpler template for script for SD.Next that implements a custom pipeline
Items that can be added:
- Any pipeline already in diffusers
  List of pipelines that can be directly used: <https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines>
- Any pipeline for which diffusers definiotion exists and can be copied
  List of pipelines with community definitions: <https://github.com/huggingface/diffusers/tree/main/examples/community>
- Any custom pipeline that you create

Author::
- Your details

Credits:
- Link to original implementation and author

Contributions:
- Submit a PR on SD.Next GitHub repo to be included in /scripts
- Before submitting a PR, make sure to test your script thoroughly and that it passes code quality checks
  Lint rules are part of SD.Next CI/CD pipeline
    > pip install ruff pylint
    > ruff scripts/example.py
    > pylint scriptts/example.py
"""

## Config

# script title
title = 'Example'

# is script available in txt2img tab
txt2img = False

# is script available in img2img tab
img2img = False

# is pipeline ok to run in pure latent mode without implicit conversions
# recommended so entire ecosystem can be used as-is, but requires that latent is in format that sdnext can understand
# some pipelines may not support this, in which case set to false and pipeline will implicitly do things like vae encode/decode on its own
latent = True

# base pipeline class from which this pipeline is derived, most commonly 'StableDiffusionPipeline' or 'StableDiffusionXLPipeline'
pipeline_base = 'StableDiffusionPipeline'

# class definition for this pipeline
# for built-in diffuser pipelines, simply import it from diffusers.pipelines above
# for example only, its set to same as base pipeline
# for community pipelines, copy class definition from community source code
# in which case only class definition code and required imports needs to be copied, not the entire source code
pipeline_class = StableDiffusionPipeline

# pipeline args values are defined in ui method below, here we need to define their exact names
# they also have to be in the exact order as they are defined in ui
# note: variable names should be exactly as defined in pipeline_class.__call__ method
# if pipeline requires a param and its not provided, it will result in runtime error
# if you provide param that is not defined by pipeline, sdnext will strip it
params = ['test1', 'test2', 'test3', 'test4']


### Script definition

class Script(scripts.Script):
    def title(self):
        return title

    def show(self, is_img2img):
        if shared.backend == shared.Backend.DIFFUSERS:
            return img2img if is_img2img else txt2img
        return False

    # Define UI for pipeline
    def ui(self, _is_img2img):
        ui_controls = []
        with gr.Row():
            ui_controls.append(gr.Slider(minimum=0, maximum=1, step=0.1, value=0.5, label="Test1"))
            ui_controls.append(gr.Slider(minimum=0, maximum=10, step=1, value=5, label="Test2"))
        with gr.Row():
            ui_controls.append(gr.Checkbox(label="Test3", value=True))
        with gr.Row():
            ui_controls.append(gr.Textbox(label="Test4", value="", placeholder="enter text here"))
        with gr.Row():
            gr.HTML('<a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl>Stable Diffusion SDXL pipeline docs</a>"')
        return ui_controls

    # Run pipeline
    def run(self, p: processing.StableDiffusionProcessing, *args): # pylint: disable=arguments-differ
        # prepare pipeline
        c = shared.sd_model.__class__.__name__ if shared.sd_model is not None else ''
        if c != pipeline_base:
            shared.log.warning(f'{title}: pipeline={c} required={pipeline_base}')
            return None
        orig_pipeline = shared.sd_model # backup current pipeline definition
        shared.sd_model = pipeline_class( # create new pipeline using currently loaded model which is always in `shared.sd_model`
            # different pipelines may need different init params, so you may need to change this
            # to see init params, see pipeline_class.__init__ method
            # if init params are incorrect you will also see a runtime error with unrecognized or missing params
            # for example:
            # > TypeError: StableDiffusionPipeline.__init__() missing 2 required positional arguments: 'safety_checker' and 'feature_extractor'
            vae = shared.sd_model.vae,
            text_encoder=shared.sd_model.text_encoder,
            tokenizer=shared.sd_model.tokenizer,
            unet=shared.sd_model.unet,
            scheduler=shared.sd_model.scheduler,
            safety_checker=shared.sd_model.safety_checker,
            feature_extractor=shared.sd_model.feature_extractor,
        )
        sd_models.copy_diffuser_options(shared.sd_model, orig_pipeline) # copy options from original pipeline
        sd_models.set_diffuser_options(shared.sd_model) # set all model options such as fp16, offload, etc.
        sd_models.move_model(shared.sd_model, devices.device) # move pipeline to device
        shared.sd_model.to(dtype=devices.dtype)

        # if pipeline also needs a specific type, you can set it here, but not commonly needed
        # shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)

        # prepare params
        # all pipeline params go into p.task_args and are automatically handled by sdnext from there
        for i in range(len(args)):
            p.task_args[params[i]] = args[i]

        # you can also re-use existing params from `p` object if pipeline wants them, but under a different name
        # for example, if pipeline expects 'image' param, but you want to use 'init_images' instead which is what img2img tab uses
        # p.task_args['image'] = p.init_images[0]

        if not latent:
            p.task_args['output_type'] = 'np'
        shared.log.debug(f'{c}: args={p.task_args}')

        # if you need to run any preprocessing, this is the place to do it

        # run processing
        processed: processing.Processed = processing.process_images(p)

        # if you need to run any postprocessing, this is the place to do it
        # you dont need to handle saving, metadata, etc - sdnext will do it for you

        # restore original pipeline
        shared.sd_model = orig_pipeline
        return processed
