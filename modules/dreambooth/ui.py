import html

import gradio as gr

import modules.textual_inversion.textual_inversion
import modules.textual_inversion.preprocess
from modules import sd_hijack, shared
from modules.dreambooth import dreambooth
from dreambooth import DreamBooth

init_text = None
class_text = None


def preprocess(*args):
    modules.textual_inversion.preprocess.preprocess(*args)

    return "Preprocessing finished.", ""


def train_embedding(*args):

    try:
        sd_hijack.undo_optimizations()
        db = DreamBooth(*args)
        out_path = db.train()

        res = f"""
Training {'interrupted' if shared.state.interrupted else 'finished'} at {db.max_train_steps} steps.
Data saved to {html.escape(out_path)}
"""
        return res, ""
    except Exception:
        raise
    finally:
        sd_hijack.apply_optimizations()

