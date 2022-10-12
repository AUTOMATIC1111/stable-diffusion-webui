import html
import os

import gradio as gr

import modules.textual_inversion.textual_inversion
import modules.textual_inversion.preprocess
from modules import sd_hijack, shared
from modules.hypernetworks import hypernetwork


def create_hypernetwork(name):
    fn = os.path.join(shared.cmd_opts.hypernetwork_dir, f"{name}.pt")
    assert not os.path.exists(fn), f"file {fn} already exists"

    hypernet = modules.hypernetworks.hypernetwork.Hypernetwork(name=name)
    hypernet.save(fn)

    shared.reload_hypernetworks()

    return gr.Dropdown.update(choices=sorted([x for x in shared.hypernetworks.keys()])), f"Created: {fn}", ""


def train_hypernetwork(*args):

    initial_hypernetwork = shared.loaded_hypernetwork

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
        sd_hijack.apply_optimizations()

