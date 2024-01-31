import gradio as gr
from typing import List


class ModalInterface(gr.Interface):
    modal_id_counter = 0

    def __init__(
        self,
        html_content: str,
        open_button_text: str,
        open_button_classes: List[str] = [],
        open_button_extra_attrs: str = ''
    ):
        self.html_content = html_content
        self.open_button_text = open_button_text
        self.open_button_classes = open_button_classes
        self.open_button_extra_attrs = open_button_extra_attrs
        self.modal_id = ModalInterface.modal_id_counter
        ModalInterface.modal_id_counter += 1

    def __call__(self):
        return self.create_modal()

    def create_modal(self, visible=True):
        html_code = f"""
        <div id="cnet-modal-{self.modal_id}" class="cnet-modal">
            <span class="cnet-modal-close">&times;</span>
            <div class="cnet-modal-content">
                {self.html_content}
            </div>
        </div>
        <div id="cnet-modal-open-{self.modal_id}" 
                class="cnet-modal-open {' '.join(self.open_button_classes)}"
                {self.open_button_extra_attrs}
        >{self.open_button_text}</div>
        """
        return gr.HTML(value=html_code, visible=visible)
