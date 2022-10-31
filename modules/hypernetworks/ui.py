import html
import os
import re

import gradio as gr
import modules.textual_inversion.preprocess
import modules.textual_inversion.textual_inversion
from modules import devices, sd_hijack, shared
from modules.hypernetworks import hypernetwork

not_available = ["hardswish", "multiheadattention"]
keys = ["linear"] + list(x for x in hypernetwork.HypernetworkModule.activation_dict.keys() if x not in not_available)

def create_hypernetwork(name, enable_sizes, overwrite_old, layer_structure=None, activation_func=None, weight_init=None, add_layer_norm=False, use_dropout=False):
    # Remove illegal characters from name.
    name = "".join( x for x in name if (x.isalnum() or x in "._- "))

    fn = os.path.join(shared.cmd_opts.hypernetwork_dir, f"{name}.pt")
    if not overwrite_old:
        assert not os.path.exists(fn), f"file {fn} already exists"

    if type(layer_structure) == str:
        layer_structure = [float(x.strip()) for x in layer_structure.split(",")]

    hypernet = modules.hypernetworks.hypernetwork.Hypernetwork(
        name=name,
        enable_sizes=[int(x) for x in enable_sizes],
        layer_structure=layer_structure,
        activation_func=activation_func,
        weight_init=weight_init,
        add_layer_norm=add_layer_norm,
        use_dropout=use_dropout,
    )
    hypernet.save(fn)

    shared.reload_hypernetworks()

    return gr.Dropdown.update(choices=sorted([x for x in shared.hypernetworks.keys()])), f"Created: {fn}", ""


def train_hypernetwork(*args):

    initial_hypernetwork = shared.loaded_hypernetwork

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
        shared.loaded_hypernetwork = initial_hypernetwork
        shared.sd_model.cond_stage_model.to(devices.device)
        shared.sd_model.first_stage_model.to(devices.device)
        sd_hijack.apply_optimizations()

