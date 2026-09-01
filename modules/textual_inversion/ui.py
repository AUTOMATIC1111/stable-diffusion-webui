import html

import gradio as gr

import modules.textual_inversion.textual_inversion
from modules import devices, sd_hijack, shared


def create_embedding(name, initialization_text, nvpt, overwrite_old):
    filename = modules.textual_inversion.textual_inversion.create_embedding(name, nvpt, overwrite_old, init_text=initialization_text)

    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()

    return gr.Dropdown.update(choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())), f"Created: {filename}", ""


def train_embedding(*args):

    assert not shared.cmd_opts.lowvram, 'Training models with lowvram not possible'

    # When the user has requested a low-VRAM profile (saver/ultra/safe mode), keep the
    # memory-efficient cross-attention path active during training rather than undoing it.
    # Undoing the optimizations (the default) is what makes training OOM on 8 GB cards.
    apply_optimizations = shared.opts.training_xattention_optimizations or devices.is_low_vram_training()
    try:
        if not apply_optimizations:
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
        if not apply_optimizations:
            sd_hijack.apply_optimizations()

