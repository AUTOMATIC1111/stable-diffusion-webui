from typing import Optional
from fastapi.exceptions import HTTPException
import gradio as gr
from modules.api import models
from modules import scripts


def script_name_to_index(name, scripts_list):
    try:
        return [script.title().lower() for script in scripts_list].index(name.lower())
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Script '{name}' not found") from e

def get_selectable_script(script_name, script_runner):
    if script_name is None or script_name == "":
        return None, None
    script_idx = script_name_to_index(script_name, script_runner.selectable_scripts)
    script = script_runner.selectable_scripts[script_idx]
    return script, script_idx

def get_scripts_list():
    t2ilist = [script.name for script in scripts.scripts_txt2img.scripts if script.name is not None]
    i2ilist = [script.name for script in scripts.scripts_img2img.scripts if script.name is not None]
    control = [script.name for script in scripts.scripts_control.scripts if script.name is not None]
    return models.ResScripts(txt2img = t2ilist, img2img = i2ilist, control = control)

def get_script_info(script_name: Optional[str] = None):
    res = []
    for script_list in [scripts.scripts_txt2img.scripts, scripts.scripts_img2img.scripts, scripts.scripts_control.scripts]:
        for script in script_list:
            if script.api_info is not None and (script_name is None or script_name == script.api_info.name):
                res.append(script.api_info)
    return res

def get_script(script_name, script_runner):
    if script_name is None or script_name == "":
        return None, None
    script_idx = script_name_to_index(script_name, script_runner.scripts)
    return script_runner.scripts[script_idx]

def init_default_script_args(script_runner):
    #find max idx from the scripts in runner and generate a none array to init script_args
    last_arg_index = 1
    for script in script_runner.scripts:
        if last_arg_index < script.args_to:
            last_arg_index = script.args_to
    # None everywhere except position 0 to initialize script args
    script_args = [None]*last_arg_index
    script_args[0] = 0

    # get default values
    if gr is None:
        return script_args
    with gr.Blocks(): # will throw errors calling ui function without this
        for script in script_runner.scripts:
            if script.ui(script.is_img2img):
                ui_default_values = []
                for elem in script.ui(script.is_img2img):
                    ui_default_values.append(elem.value)
                script_args[script.args_from:script.args_to] = ui_default_values
    return script_args

def init_script_args(p, request, default_script_args, selectable_scripts, selectable_script_idx, script_runner):
    script_args = default_script_args.copy()
    # position 0 in script_arg is the idx+1 of the selectable script that is going to be run when using scripts.scripts_*2img.run()
    if selectable_scripts:
        script_args[selectable_scripts.args_from:selectable_scripts.args_to] = request.script_args
        script_args[0] = selectable_script_idx + 1
    # Now check for always on scripts
    if request.alwayson_scripts and (len(request.alwayson_scripts) > 0):
        for alwayson_script_name in request.alwayson_scripts.keys():
            alwayson_script = get_script(alwayson_script_name, script_runner)
            if alwayson_script is None:
                raise HTTPException(status_code=422, detail=f"Always on script not found: {alwayson_script_name}")
            if not alwayson_script.alwayson:
                raise HTTPException(status_code=422, detail=f"Selectable script cannot be in always on params: {alwayson_script_name}")
            if "args" in request.alwayson_scripts[alwayson_script_name]:
                # min between arg length in scriptrunner and arg length in the request
                for idx in range(0, min((alwayson_script.args_to - alwayson_script.args_from), len(request.alwayson_scripts[alwayson_script_name]["args"]))):
                    script_args[alwayson_script.args_from + idx] = request.alwayson_scripts[alwayson_script_name]["args"][idx]
                p.per_script_args[alwayson_script.title()] = request.alwayson_scripts[alwayson_script_name]["args"]
    return script_args
