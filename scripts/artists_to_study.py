import os
import gradio as gr
from modules import scripts, script_callbacks
root_path = os.path.join(scripts.basedir(), "ats\\thumbnail")

jpg_paths = ['anime', 'cartoon', 'digipa-high-impact', 'digipa-med-impact', 'digipa-low-impact', 'fareast', 'fineart', 'scribbles', 'special', 'ukioe', 'weird', 'black-white', 'nudity', 'c', 'n']
prompt_paths = ['dog', 'house', 'portrait', 'spaceship']

def get_images(jpg_path):
    images = []
    for jpg_image in os.listdir(f"{jpg_path}"):
      final_path = f"{jpg_path}\\{jpg_image}"
      final_path = final_path.replace('\\', os.sep).replace('/', os.sep)
      try:
        images.append((final_path, f"{jpg_image}"))
      except Exception as e:
        print(final_path, e)
    return images

def on_ui_tabs():     
    with gr.Blocks() as artists_to_study:
        for prompt_path in prompt_paths:
            with gr.Tab(prompt_path):
                for jpg_path in jpg_paths:
                    with gr.Tab(jpg_path):
                        input_path = f"{root_path}\\{prompt_path}\\{jpg_path}"
                        input_path = input_path.replace('\\', os.sep).replace('/', os.sep)
                        gallery_label = f"{prompt_path}-{jpg_path}"
                        txt = gr.Textbox(value=input_path, interactive=False, show_label=False, visible=True)
                        btn = gr.Button(value="Get Images", elem_id=f"ats-button-{prompt_path}-{jpg_path}")
                        gallery = gr.Gallery(label=gallery_label, show_label=True, elem_id=f"ats-gallery-{prompt_path}-{jpg_path}").style(grid=[5], height="auto")
                        btn.click(get_images, txt, gallery)
        gr.HTML(
            """
                <p style="font-size: 12px" align="right">artists to study extension by camenduru | <a href="https://github.com/camenduru" target="_blank">github</a> | <a href="https://twitter.com/camenduru" target="_blank">twitter</a> | <a href="https://www.youtube.com/channel/UCdk3FaULpDK8kRCPG5jbgHQ" target="_blank">youtube</a> | <a href="https://artiststostudy.pages.dev" target="_blank">hi-res images</a><br />All images generated with CompVis/stable-diffusion-v1-4 + <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/artists.csv" target="_blank">artists.csv</a> | License: Attribution 4.0 International (CC BY 4.0)</p>
            """
        )
    return (artists_to_study, "Artists To Study", "artists_to_study"),

script_callbacks.on_ui_tabs(on_ui_tabs)
