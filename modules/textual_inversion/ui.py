import html

import gradio as gr

import modules.textual_inversion.textual_inversion as ti
from modules import sd_hijack, shared


def create_embedding(name, nvpt):
    filename = ti.create_embedding(name, nvpt)

    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()

    return gr.Dropdown.update(choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())), f"Created: {filename}", ""


def train_embedding(*args):

    try:
        sd_hijack.undo_optimizations()

        embedding, filename = ti.train_embedding(*args)

        res = f"""
Training {'interrupted' if shared.state.interrupted else 'finished'} after {embedding.step} steps.
Embedding saved to {html.escape(filename)}
"""
        return res, ""
    except Exception:
        raise
    finally:
        sd_hijack.apply_optimizations()
