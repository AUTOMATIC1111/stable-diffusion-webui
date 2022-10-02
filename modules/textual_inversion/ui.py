import html

import gradio as gr

import modules.textual_inversion.textual_inversion
import modules.textual_inversion.preprocess
from modules import sd_hijack, shared


def create_embedding(name, initialization_text, nvpt):
    filename = modules.textual_inversion.textual_inversion.create_embedding(name, nvpt, init_text=initialization_text)

    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()

    return gr.Dropdown.update(choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())), f"Created: {filename}", ""


def preprocess(*args):
    modules.textual_inversion.preprocess.preprocess(*args)

    return "Preprocessing finished.", ""


def train_embedding(*args):

    try:
        sd_hijack.undo_optimizations()

        embedding, filename = modules.textual_inversion.textual_inversion.train_embedding(*args)

        res = f"""
Training {'interrupted' if shared.state.interrupted else 'finished'} at {embedding.step} steps.
Embedding saved to {html.escape(filename)}
"""
        return res, ""
    except Exception:
        raise
    finally:
        sd_hijack.apply_optimizations()

