import html
import os

import gradio as gr

import modules.textual_inversion.textual_inversion
import modules.textual_inversion.preprocess
from modules import sd_hijack, shared, devices
from modules.hypernetworks import hypernetwork


def create_hypernetwork(name, enable_sizes):
    fn = os.path.join(shared.cmd_opts.hypernetwork_dir, f"{name}.pt")
    assert not os.path.exists(fn), f"file {fn} already exists"

    hypernet = modules.hypernetworks.hypernetwork.Hypernetwork(name=name, enable_sizes=[int(x) for x in enable_sizes])
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

