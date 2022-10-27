
from modules import scripts, script_callbacks
import aesthetic_clip
import gradio as gr

aesthetic = aesthetic_clip.AestheticCLIP()
aesthetic_imgs_components = []


class AestheticScript(scripts.Script):
    def title(self):
        return "Aesthetic embeddings"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        aesthetic_weight, aesthetic_steps, aesthetic_lr, aesthetic_slerp, aesthetic_imgs, aesthetic_imgs_text, aesthetic_slerp_angle, aesthetic_text_negative = aesthetic_clip.create_ui()

        self.infotext_fields = [
            (aesthetic_lr, "Aesthetic LR"),
            (aesthetic_weight, "Aesthetic weight"),
            (aesthetic_steps, "Aesthetic steps"),
            (aesthetic_imgs, "Aesthetic embedding"),
            (aesthetic_slerp, "Aesthetic slerp"),
            (aesthetic_imgs_text, "Aesthetic text"),
            (aesthetic_text_negative, "Aesthetic text negative"),
            (aesthetic_slerp_angle, "Aesthetic slerp angle"),
        ]

        aesthetic_imgs_components.append(aesthetic_imgs)

        return [aesthetic_weight, aesthetic_steps, aesthetic_lr, aesthetic_slerp, aesthetic_imgs, aesthetic_imgs_text, aesthetic_slerp_angle, aesthetic_text_negative]

    def process(self, p, aesthetic_weight, aesthetic_steps, aesthetic_lr, aesthetic_slerp, aesthetic_imgs, aesthetic_imgs_text, aesthetic_slerp_angle, aesthetic_text_negative):
        aesthetic.set_aesthetic_params(p, float(aesthetic_lr), float(aesthetic_weight), int(aesthetic_steps), aesthetic_imgs, aesthetic_slerp, aesthetic_imgs_text, aesthetic_slerp_angle, aesthetic_text_negative)


def on_model_loaded(sd_model):
    aesthetic.process_tokens = sd_model.cond_stage_model.process_tokens
    sd_model.cond_stage_model.process_tokens = aesthetic


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as aesthetic_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                gr.HTML(value="Create an aesthetic embedding out of any number of images")

                new_embedding_name_ae = gr.Textbox(label="Name")
                process_src_ae = gr.Textbox(label='Source directory')
                batch_ae = gr.Slider(minimum=1, maximum=1024, step=1, label="Batch size", value=256)

                with gr.Row():
                    with gr.Column(scale=3):
                        gr.HTML(value="")

                    with gr.Column():
                        create_embedding_ae = gr.Button(value="Create images embedding", variant='primary')

            with gr.Column():
                output = gr.Text(value="", show_label=False)

        dropdown_components = aesthetic_imgs_components.copy()

        def generate_embs(*args):
            res = aesthetic_clip.generate_imgs_embd(*args)

            aesthetic_clip.update_aesthetic_embeddings()
            updates = [gr.Dropdown.update(choices=sorted(aesthetic_clip.aesthetic_embeddings.keys())) for _ in range(len(dropdown_components))]

            return [*updates, res]

        create_embedding_ae.click(
            fn=generate_embs,
            inputs=[
                new_embedding_name_ae,
                process_src_ae,
                batch_ae
            ],
            outputs=[
                *dropdown_components,
                output
            ]
        )

        aesthetic_imgs_components.clear()

    return [(aesthetic_interface, "Create aesthetic embedding", "aesthetic_interface")]


script_callbacks.on_model_loaded(on_model_loaded)
script_callbacks.on_ui_tabs(on_ui_tabs)
