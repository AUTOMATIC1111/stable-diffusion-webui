import os
import gradio as gr

from modules import errors, shared


class PostprocessedImage:
    def __init__(self, image):
        self.image = image
        self.info = {}


class ScriptPostprocessing:
    filename = None
    controls = None
    args_from = None
    args_to = None

    order = 1000
    """scripts will be ordred by this value in postprocessing UI"""

    name = None
    """this function should return the title of the script."""

    group = None
    """A gr.Group component that has all script's UI inside it"""

    def ui(self):
        """
        This function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be a dictionary that maps parameter names to components used in processing.
        Values of those components will be passed to process() function.
        """

        pass

    def process(self, pp: PostprocessedImage, **args):
        """
        This function is called to postprocess the image.
        args contains a dictionary with all values returned by components from ui()
        """

        pass

    def image_changed(self):
        pass




def wrap_call(func, filename, funcname, *args, default=None, **kwargs):
    try:
        res = func(*args, **kwargs)
        return res
    except Exception as e:
        errors.display(e, f"calling {filename}/{funcname}")

    return default


class ScriptPostprocessingRunner:
    def __init__(self):
        self.scripts = None
        self.ui_created = False

    def initialize_scripts(self, scripts_data):
        self.scripts = []

        for script_class, path, basedir, script_module in scripts_data:
            script: ScriptPostprocessing = script_class()
            script.filename = path

            if script.name == "Simple Upscale":
                continue

            self.scripts.append(script)

    def create_script_ui(self, script, inputs):
        script.args_from = len(inputs)
        script.args_to = len(inputs)

        script.controls = wrap_call(script.ui, script.filename, "ui")

        for control in script.controls.values():
            control.custom_script_source = os.path.basename(script.filename)

        inputs += list(script.controls.values())
        script.args_to = len(inputs)

    def scripts_in_preferred_order(self):
        if self.scripts is None:
            import modules.scripts
            self.initialize_scripts(modules.scripts.postprocessing_scripts_data)

        scripts_order = shared.opts.postprocessing_operation_order

        def script_score(name):
            for i, possible_match in enumerate(scripts_order):
                if possible_match == name:
                    return i

            return len(self.scripts)

        script_scores = {script.name: (script_score(script.name), script.order, script.name, original_index) for original_index, script in enumerate(self.scripts)}

        return sorted(self.scripts, key=lambda x: script_scores[x.name])

    def setup_ui(self):
        inputs = []

        for script in self.scripts_in_preferred_order():
            with gr.Box() as group:
                self.create_script_ui(script, inputs)

            script.group = group

        self.ui_created = True
        return inputs

    def run(self, pp: PostprocessedImage, args):
        for script in self.scripts_in_preferred_order():
            shared.state.job = script.name

            script_args = args[script.args_from:script.args_to]

            process_args = {}
            for (name, component), value in zip(script.controls.items(), script_args):
                process_args[name] = value

            script.process(pp, **process_args)

    def create_args_for_run(self, scripts_args):
        if not self.ui_created:
            with gr.Blocks(analytics_enabled=False):
                self.setup_ui()

        scripts = self.scripts_in_preferred_order()
        args = [None] * max([x.args_to for x in scripts])

        for script in scripts:
            script_args_dict = scripts_args.get(script.name, None)
            if script_args_dict is not None:

                for i, name in enumerate(script.controls):
                    args[script.args_from + i] = script_args_dict.get(name, None)

        return args

    def image_changed(self):
        for script in self.scripts_in_preferred_order():
            script.image_changed()

