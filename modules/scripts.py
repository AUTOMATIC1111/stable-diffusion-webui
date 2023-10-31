import os
import re
import sys
import inspect
from collections import namedtuple
from dataclasses import dataclass

import gradio as gr

from modules import shared, paths, script_callbacks, extensions, script_loading, scripts_postprocessing, errors, timer

AlwaysVisible = object()


class PostprocessImageArgs:
    def __init__(self, image):
        self.image = image


class PostprocessBatchListArgs:
    def __init__(self, images):
        self.images = images


@dataclass
class OnComponent:
    component: gr.blocks.Block


class Script:
    name = None
    """script's internal name derived from title"""

    section = None
    """name of UI section that the script's controls will be placed into"""

    filename = None
    args_from = None
    args_to = None
    alwayson = False

    is_txt2img = False
    is_img2img = False
    tabname = None

    group = None
    """A gr.Group component that has all script's UI inside it."""

    create_group = True
    """If False, for alwayson scripts, a group component will not be created."""

    infotext_fields = None
    """if set in ui(), this is a list of pairs of gradio component + text; the text will be used when
    parsing infotext to set the value for the component; see ui.py's txt2img_paste_fields for an example
    """

    paste_field_names = None
    """if set in ui(), this is a list of names of infotext fields; the fields will be sent through the
    various "Send to <X>" buttons when clicked
    """

    api_info = None
    """Generated value of type modules.api.models.ScriptInfo with information about the script for API"""

    on_before_component_elem_id = None
    """list of callbacks to be called before a component with an elem_id is created"""

    on_after_component_elem_id = None
    """list of callbacks to be called after a component with an elem_id is created"""

    setup_for_ui_only = False
    """If true, the script setup will only be run in Gradio UI, not in API"""

    def title(self):
        """this function should return the title of the script. This is what will be displayed in the dropdown menu."""

        raise NotImplementedError()

    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """

        pass

    def show(self, is_img2img):
        """
        is_img2img is True if this function is called for the img2img interface, and Fasle otherwise

        This function should return:
         - False if the script should not be shown in UI at all
         - True if the script should be shown in UI if it's selected in the scripts dropdown
         - script.AlwaysVisible if the script should be shown in UI at all times
         """

        return True

    def run(self, p, *args):
        """
        This function is called if the script has been selected in the script dropdown.
        It must do all processing and return the Processed object with results, same as
        one returned by processing.process_images.

        Usually the processing is done by calling the processing.process_images function.

        args contains all values returned by components from ui()
        """

        pass

    def setup(self, p, *args):
        """For AlwaysVisible scripts, this function is called when the processing object is set up, before any processing starts.
        args contains all values returned by components from ui().
        """
        pass


    def before_process(self, p, *args):
        """
        This function is called very early during processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """

        pass

    def process(self, p, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """

        pass

    def before_process_batch(self, p, *args, **kwargs):
        """
        Called before extra networks are parsed from the prompt, so you can add
        new extra network keywords to the prompt with this callback.

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """

        pass

    def after_extra_networks_activate(self, p, *args, **kwargs):
        """
        Called after extra networks activation, before conds calculation
        allow modification of the network after extra networks activation been applied
        won't be call if p.disable_extra_networks

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
          - extra_network_data - list of ExtraNetworkParams for current stage
        """
        pass

    def process_batch(self, p, *args, **kwargs):
        """
        Same as process(), but called for every batch.

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """

        pass

    def postprocess_batch(self, p, *args, **kwargs):
        """
        Same as process_batch(), but called for every batch after it has been generated.

        **kwargs will have same items as process_batch, and also:
          - batch_number - index of current batch, from 0 to number of batches-1
          - images - torch tensor with all generated images, with values ranging from 0 to 1;
        """

        pass

    def postprocess_batch_list(self, p, pp: PostprocessBatchListArgs, *args, **kwargs):
        """
        Same as postprocess_batch(), but receives batch images as a list of 3D tensors instead of a 4D tensor.
        This is useful when you want to update the entire batch instead of individual images.

        You can modify the postprocessing object (pp) to update the images in the batch, remove images, add images, etc.
        If the number of images is different from the batch size when returning,
        then the script has the responsibility to also update the following attributes in the processing object (p):
          - p.prompts
          - p.negative_prompts
          - p.seeds
          - p.subseeds

        **kwargs will have same items as process_batch, and also:
          - batch_number - index of current batch, from 0 to number of batches-1
        """

        pass

    def postprocess_image(self, p, pp: PostprocessImageArgs, *args):
        """
        Called for every image after it has been generated.
        """

        pass

    def postprocess(self, p, processed, *args):
        """
        This function is called after processing ends for AlwaysVisible scripts.
        args contains all values returned by components from ui()
        """

        pass

    def before_component(self, component, **kwargs):
        """
        Called before a component is created.
        Use elem_id/label fields of kwargs to figure out which component it is.
        This can be useful to inject your own components somewhere in the middle of vanilla UI.
        You can return created components in the ui() function to add them to the list of arguments for your processing functions
        """

        pass

    def after_component(self, component, **kwargs):
        """
        Called after a component is created. Same as above.
        """

        pass

    def on_before_component(self, callback, *, elem_id):
        """
        Calls callback before a component is created. The callback function is called with a single argument of type OnComponent.

        May be called in show() or ui() - but it may be too late in latter as some components may already be created.

        This function is an alternative to before_component in that it also cllows to run before a component is created, but
        it doesn't require to be called for every created component - just for the one you need.
        """
        if self.on_before_component_elem_id is None:
            self.on_before_component_elem_id = []

        self.on_before_component_elem_id.append((elem_id, callback))

    def on_after_component(self, callback, *, elem_id):
        """
        Calls callback after a component is created. The callback function is called with a single argument of type OnComponent.
        """
        if self.on_after_component_elem_id is None:
            self.on_after_component_elem_id = []

        self.on_after_component_elem_id.append((elem_id, callback))

    def describe(self):
        """unused"""
        return ""

    def elem_id(self, item_id):
        """helper function to generate id for a HTML element, constructs final id out of script name, tab and user-supplied item_id"""

        need_tabname = self.show(True) == self.show(False)
        tabkind = 'img2img' if self.is_img2img else 'txt2img'
        tabname = f"{tabkind}_" if need_tabname else ""
        title = re.sub(r'[^a-z_0-9]', '', re.sub(r'\s', '_', self.title().lower()))

        return f'script_{tabname}{title}_{item_id}'

    def before_hr(self, p, *args):
        """
        This function is called before hires fix start.
        """
        pass


class ScriptBuiltinUI(Script):
    setup_for_ui_only = True

    def elem_id(self, item_id):
        """helper function to generate id for a HTML element, constructs final id out of tab and user-supplied item_id"""

        need_tabname = self.show(True) == self.show(False)
        tabname = ('img2img' if self.is_img2img else 'txt2img') + "_" if need_tabname else ""

        return f'{tabname}{item_id}'


current_basedir = paths.script_path


def basedir():
    """returns the base directory for the current script. For scripts in the main scripts directory,
    this is the main directory (where webui.py resides), and for scripts in extensions directory
    (ie extensions/aesthetic/script/aesthetic.py), this is extension's directory (extensions/aesthetic)
    """
    return current_basedir


ScriptFile = namedtuple("ScriptFile", ["basedir", "filename", "path"])

scripts_data = []
postprocessing_scripts_data = []
ScriptClassData = namedtuple("ScriptClassData", ["script_class", "path", "basedir", "module"])


def list_scripts(scriptdirname, extension, *, include_extensions=True):
    scripts_list = []

    basedir = os.path.join(paths.script_path, scriptdirname)
    if os.path.exists(basedir):
        for filename in sorted(os.listdir(basedir)):
            scripts_list.append(ScriptFile(paths.script_path, filename, os.path.join(basedir, filename)))

    if include_extensions:
        for ext in extensions.active():
            scripts_list += ext.list_files(scriptdirname, extension)

    scripts_list = [x for x in scripts_list if os.path.splitext(x.path)[1].lower() == extension and os.path.isfile(x.path)]

    return scripts_list


def list_files_with_name(filename):
    res = []

    dirs = [paths.script_path] + [ext.path for ext in extensions.active()]

    for dirpath in dirs:
        if not os.path.isdir(dirpath):
            continue

        path = os.path.join(dirpath, filename)
        if os.path.isfile(path):
            res.append(path)

    return res


def load_scripts():
    global current_basedir
    scripts_data.clear()
    postprocessing_scripts_data.clear()
    script_callbacks.clear_callbacks()

    scripts_list = list_scripts("scripts", ".py") + list_scripts("modules/processing_scripts", ".py", include_extensions=False)

    syspath = sys.path

    def register_scripts_from_module(module):
        for script_class in module.__dict__.values():
            if not inspect.isclass(script_class):
                continue

            if issubclass(script_class, Script):
                scripts_data.append(ScriptClassData(script_class, scriptfile.path, scriptfile.basedir, module))
            elif issubclass(script_class, scripts_postprocessing.ScriptPostprocessing):
                postprocessing_scripts_data.append(ScriptClassData(script_class, scriptfile.path, scriptfile.basedir, module))

    def orderby(basedir):
        # 1st webui, 2nd extensions-builtin, 3rd extensions
        priority = {os.path.join(paths.script_path, "extensions-builtin"):1, paths.script_path:0}
        for key in priority:
            if basedir.startswith(key):
                return priority[key]
        return 9999

    for scriptfile in sorted(scripts_list, key=lambda x: [orderby(x.basedir), x]):
        try:
            if scriptfile.basedir != paths.script_path:
                sys.path = [scriptfile.basedir] + sys.path
            current_basedir = scriptfile.basedir

            script_module = script_loading.load_module(scriptfile.path)
            register_scripts_from_module(script_module)

        except Exception:
            errors.report(f"Error loading script: {scriptfile.filename}", exc_info=True)

        finally:
            sys.path = syspath
            current_basedir = paths.script_path
            timer.startup_timer.record(scriptfile.filename)

    global scripts_txt2img, scripts_img2img, scripts_postproc

    scripts_txt2img = ScriptRunner()
    scripts_img2img = ScriptRunner()
    scripts_postproc = scripts_postprocessing.ScriptPostprocessingRunner()


def wrap_call(func, filename, funcname, *args, default=None, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        errors.report(f"Error calling: {filename}/{funcname}", exc_info=True)

    return default


class ScriptRunner:
    def __init__(self):
        self.scripts = []
        self.selectable_scripts = []
        self.alwayson_scripts = []
        self.titles = []
        self.title_map = {}
        self.infotext_fields = []
        self.paste_field_names = []
        self.inputs = [None]

        self.on_before_component_elem_id = {}
        """dict of callbacks to be called before an element is created; key=elem_id, value=list of callbacks"""

        self.on_after_component_elem_id = {}
        """dict of callbacks to be called after an element is created; key=elem_id, value=list of callbacks"""

    def initialize_scripts(self, is_img2img):
        from modules import scripts_auto_postprocessing

        self.scripts.clear()
        self.alwayson_scripts.clear()
        self.selectable_scripts.clear()

        auto_processing_scripts = scripts_auto_postprocessing.create_auto_preprocessing_script_data()

        for script_data in auto_processing_scripts + scripts_data:
            script = script_data.script_class()
            script.filename = script_data.path
            script.is_txt2img = not is_img2img
            script.is_img2img = is_img2img
            script.tabname = "img2img" if is_img2img else "txt2img"

            visibility = script.show(script.is_img2img)

            if visibility == AlwaysVisible:
                self.scripts.append(script)
                self.alwayson_scripts.append(script)
                script.alwayson = True

            elif visibility:
                self.scripts.append(script)
                self.selectable_scripts.append(script)

        self.apply_on_before_component_callbacks()

    def apply_on_before_component_callbacks(self):
        for script in self.scripts:
            on_before = script.on_before_component_elem_id or []
            on_after = script.on_after_component_elem_id or []

            for elem_id, callback in on_before:
                if elem_id not in self.on_before_component_elem_id:
                    self.on_before_component_elem_id[elem_id] = []

                self.on_before_component_elem_id[elem_id].append((callback, script))

            for elem_id, callback in on_after:
                if elem_id not in self.on_after_component_elem_id:
                    self.on_after_component_elem_id[elem_id] = []

                self.on_after_component_elem_id[elem_id].append((callback, script))

            on_before.clear()
            on_after.clear()

    def create_script_ui(self, script):
        import modules.api.models as api_models

        script.args_from = len(self.inputs)
        script.args_to = len(self.inputs)

        controls = wrap_call(script.ui, script.filename, "ui", script.is_img2img)

        if controls is None:
            return

        script.name = wrap_call(script.title, script.filename, "title", default=script.filename).lower()
        api_args = []

        for control in controls:
            control.custom_script_source = os.path.basename(script.filename)

            arg_info = api_models.ScriptArg(label=control.label or "")

            for field in ("value", "minimum", "maximum", "step", "choices"):
                v = getattr(control, field, None)
                if v is not None:
                    setattr(arg_info, field, v)

            api_args.append(arg_info)

        script.api_info = api_models.ScriptInfo(
            name=script.name,
            is_img2img=script.is_img2img,
            is_alwayson=script.alwayson,
            args=api_args,
        )

        if script.infotext_fields is not None:
            self.infotext_fields += script.infotext_fields

        if script.paste_field_names is not None:
            self.paste_field_names += script.paste_field_names

        self.inputs += controls
        script.args_to = len(self.inputs)

    def setup_ui_for_section(self, section, scriptlist=None):
        if scriptlist is None:
            scriptlist = self.alwayson_scripts

        for script in scriptlist:
            if script.alwayson and script.section != section:
                continue

            if script.create_group:
                with gr.Group(visible=script.alwayson) as group:
                    self.create_script_ui(script)

                script.group = group
            else:
                self.create_script_ui(script)

    def prepare_ui(self):
        self.inputs = [None]

    def setup_ui(self):
        all_titles = [wrap_call(script.title, script.filename, "title") or script.filename for script in self.scripts]
        self.title_map = {title.lower(): script for title, script in zip(all_titles, self.scripts)}
        self.titles = [wrap_call(script.title, script.filename, "title") or f"{script.filename} [error]" for script in self.selectable_scripts]

        self.setup_ui_for_section(None)

        dropdown = gr.Dropdown(label="Script", elem_id="script_list", choices=["None"] + self.titles, value="None", type="index")
        self.inputs[0] = dropdown

        self.setup_ui_for_section(None, self.selectable_scripts)

        def select_script(script_index):
            selected_script = self.selectable_scripts[script_index - 1] if script_index>0 else None

            return [gr.update(visible=selected_script == s) for s in self.selectable_scripts]

        def init_field(title):
            """called when an initial value is set from ui-config.json to show script's UI components"""

            if title == 'None':
                return

            script_index = self.titles.index(title)
            self.selectable_scripts[script_index].group.visible = True

        dropdown.init_field = init_field

        dropdown.change(
            fn=select_script,
            inputs=[dropdown],
            outputs=[script.group for script in self.selectable_scripts]
        )

        self.script_load_ctr = 0

        def onload_script_visibility(params):
            title = params.get('Script', None)
            if title:
                title_index = self.titles.index(title)
                visibility = title_index == self.script_load_ctr
                self.script_load_ctr = (self.script_load_ctr + 1) % len(self.titles)
                return gr.update(visible=visibility)
            else:
                return gr.update(visible=False)

        self.infotext_fields.append((dropdown, lambda x: gr.update(value=x.get('Script', 'None'))))
        self.infotext_fields.extend([(script.group, onload_script_visibility) for script in self.selectable_scripts])

        self.apply_on_before_component_callbacks()

        return self.inputs

    def run(self, p, *args):
        script_index = args[0]

        if script_index == 0:
            return None

        script = self.selectable_scripts[script_index-1]

        if script is None:
            return None

        script_args = args[script.args_from:script.args_to]
        processed = script.run(p, *script_args)

        shared.total_tqdm.clear()

        return processed

    def before_process(self, p):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.before_process(p, *script_args)
            except Exception:
                errors.report(f"Error running before_process: {script.filename}", exc_info=True)

    def process(self, p):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.process(p, *script_args)
            except Exception:
                errors.report(f"Error running process: {script.filename}", exc_info=True)

    def before_process_batch(self, p, **kwargs):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.before_process_batch(p, *script_args, **kwargs)
            except Exception:
                errors.report(f"Error running before_process_batch: {script.filename}", exc_info=True)

    def after_extra_networks_activate(self, p, **kwargs):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.after_extra_networks_activate(p, *script_args, **kwargs)
            except Exception:
                errors.report(f"Error running after_extra_networks_activate: {script.filename}", exc_info=True)

    def process_batch(self, p, **kwargs):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.process_batch(p, *script_args, **kwargs)
            except Exception:
                errors.report(f"Error running process_batch: {script.filename}", exc_info=True)

    def postprocess(self, p, processed):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.postprocess(p, processed, *script_args)
            except Exception:
                errors.report(f"Error running postprocess: {script.filename}", exc_info=True)

    def postprocess_batch(self, p, images, **kwargs):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.postprocess_batch(p, *script_args, images=images, **kwargs)
            except Exception:
                errors.report(f"Error running postprocess_batch: {script.filename}", exc_info=True)

    def postprocess_batch_list(self, p, pp: PostprocessBatchListArgs, **kwargs):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.postprocess_batch_list(p, pp, *script_args, **kwargs)
            except Exception:
                errors.report(f"Error running postprocess_batch_list: {script.filename}", exc_info=True)

    def postprocess_image(self, p, pp: PostprocessImageArgs):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.postprocess_image(p, pp, *script_args)
            except Exception:
                errors.report(f"Error running postprocess_image: {script.filename}", exc_info=True)

    def before_component(self, component, **kwargs):
        for callback, script in self.on_before_component_elem_id.get(kwargs.get("elem_id"), []):
            try:
                callback(OnComponent(component=component))
            except Exception:
                errors.report(f"Error running on_before_component: {script.filename}", exc_info=True)

        for script in self.scripts:
            try:
                script.before_component(component, **kwargs)
            except Exception:
                errors.report(f"Error running before_component: {script.filename}", exc_info=True)

    def after_component(self, component, **kwargs):
        for callback, script in self.on_after_component_elem_id.get(component.elem_id, []):
            try:
                callback(OnComponent(component=component))
            except Exception:
                errors.report(f"Error running on_after_component: {script.filename}", exc_info=True)

        for script in self.scripts:
            try:
                script.after_component(component, **kwargs)
            except Exception:
                errors.report(f"Error running after_component: {script.filename}", exc_info=True)

    def script(self, title):
        return self.title_map.get(title.lower())

    def reload_sources(self, cache):
        for si, script in list(enumerate(self.scripts)):
            args_from = script.args_from
            args_to = script.args_to
            filename = script.filename

            module = cache.get(filename, None)
            if module is None:
                module = script_loading.load_module(script.filename)
                cache[filename] = module

            for script_class in module.__dict__.values():
                if type(script_class) == type and issubclass(script_class, Script):
                    self.scripts[si] = script_class()
                    self.scripts[si].filename = filename
                    self.scripts[si].args_from = args_from
                    self.scripts[si].args_to = args_to

    def before_hr(self, p):
        for script in self.alwayson_scripts:
            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.before_hr(p, *script_args)
            except Exception:
                errors.report(f"Error running before_hr: {script.filename}", exc_info=True)

    def setup_scrips(self, p, *, is_ui=True):
        for script in self.alwayson_scripts:
            if not is_ui and script.setup_for_ui_only:
                continue

            try:
                script_args = p.script_args[script.args_from:script.args_to]
                script.setup(p, *script_args)
            except Exception:
                errors.report(f"Error running setup: {script.filename}", exc_info=True)


scripts_txt2img: ScriptRunner = None
scripts_img2img: ScriptRunner = None
scripts_postproc: scripts_postprocessing.ScriptPostprocessingRunner = None
scripts_current: ScriptRunner = None


def reload_script_body_only():
    cache = {}
    scripts_txt2img.reload_sources(cache)
    scripts_img2img.reload_sources(cache)


reload_scripts = load_scripts  # compatibility alias
