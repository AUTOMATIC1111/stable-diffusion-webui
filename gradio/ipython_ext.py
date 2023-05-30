try:
    from IPython.core.magic import needs_local_scope, register_cell_magic
except ImportError:
    pass

import warnings

import gradio as gr


def load_ipython_extension(ipython):
    __demo = gr.Blocks()

    @register_cell_magic
    @needs_local_scope
    def blocks(line, cell, local_ns=None):
        if "gr.Interface" in cell:
            warnings.warn(
                "Usage of gradio.Interface with %%blocks may result in errors."
            )
        with __demo.clear():
            exec(cell, None, local_ns)
            __demo.launch(quiet=True)
