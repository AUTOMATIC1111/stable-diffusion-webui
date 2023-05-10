import html

import gradio as gr
import modules.hypernetworks.hypernetwork
from modules import devices, sd_hijack, shared

not_available = ["hardswish", "multiheadattention"]
keys = [x for x in modules.hypernetworks.hypernetwork.HypernetworkModule.activation_dict if x not in not_available]


def create_hypernetwork(name, enable_sizes, overwrite_old, layer_structure=None, activation_func=None, weight_init=None, add_layer_norm=False, use_dropout=False, dropout_structure=None):
    filename = modules.hypernetworks.hypernetwork.create_hypernetwork(name, enable_sizes, overwrite_old, layer_structure, activation_func, weight_init, add_layer_norm, use_dropout, dropout_structure)

    return gr.Dropdown.update(choices=sorted(shared.hypernetworks)), f"Created: {filename}", ""


def train_hypernetwork(*args):
    shared.loaded_hypernetworks = []

    assert not shared.cmd_opts.lowvram, 'Training models with lowvram is not possible'

    try:
        sd_hijack.undo_optimizations()

        hypernetwork, filename = modules.hypernetworks.hypernetwork.train_hypernetwork(*args)

        res = f"""
Training {'interrupted' if shared.state.interrupted else 'finished'} at {hypernetwork.step} steps.
Hypernetwork saved to {html.escape(filename)}
"""
        return res, ""
    except Exception:
        raise
    finally:
        shared.sd_model.cond_stage_model.to(devices.device)
        shared.sd_model.first_stage_model.to(devices.device)
        sd_hijack.apply_optimizations()

