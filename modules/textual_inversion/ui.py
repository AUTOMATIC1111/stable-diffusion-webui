import html
import gradio as gr
import modules.textual_inversion.textual_inversion
import modules.textual_inversion.preprocess
from modules import shared


def create_embedding(name, initialization_text, nvpt, overwrite_old):
    from modules import sd_hijack
    filename = modules.textual_inversion.textual_inversion.create_embedding(name, nvpt, overwrite_old, init_text=initialization_text)
    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
    return gr.Dropdown.update(choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())), f"Created: {filename}", ""


def preprocess(*args):
    modules.textual_inversion.preprocess.preprocess(*args)
    return f"Preprocessing {'interrupted' if shared.state.interrupted else 'finished'}.", ""


def train_embedding(*args):
    from modules import sd_hijack
    assert not shared.cmd_opts.lowvram, 'Training models with lowvram not possible'
    apply_optimizations = False
    try:
        if not apply_optimizations:
            sd_hijack.undo_optimizations()
        embedding, filename = modules.textual_inversion.textual_inversion.train_embedding(*args)
        res = f"Training {'interrupted' if shared.state.interrupted else 'finished'} at {embedding.step} steps. Embedding saved to {html.escape(filename)}"
        return res, ""
    except Exception as e:
        shared.log.error(f"Exception in train_embedding: {e}")
        raise RuntimeError from e
    finally:
        if not apply_optimizations:
            sd_hijack.apply_optimizations()
