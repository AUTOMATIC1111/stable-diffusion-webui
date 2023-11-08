import os
from pathlib import Path
import gradio as gr
import modules.scripts as scripts
from modules import script_callbacks, shared

mapping = [(info.infotext, k) for k, info in shared.opts.data_labels.items() if info.infotext]
shared.options_templates.update(shared.options_section(('uiux_core', "Anapnoe UI-UX"), {
    "uiux_enable_console_log": shared.OptionInfo(False, "Enable console log"),
    "uiux_max_resolution_output": shared.OptionInfo(2048, "Max resolution output for txt2img and img2img"),
    "uiux_show_input_range_ticks": shared.OptionInfo(True, "Show ticks for input range slider"),
    "uiux_no_slider_layout": shared.OptionInfo(False, "No input range sliders"),
    "uiux_disable_transitions": shared.OptionInfo(False, "Disable transitions"),
    "uiux_default_layout": shared.OptionInfo("Auto", "Layout", gr.Radio, {"choices": ["Auto","Desktop", "Mobile"]}),  
    "uiux_mobile_scale": shared.OptionInfo(0.7, "Mobile scale", gr.Slider, {"minimum": 0.5, "maximum": 1, "step": 0.05}),
    "uiux_show_labels_aside": shared.OptionInfo(False, "Show labels for aside tabs"),
    "uiux_show_labels_main": shared.OptionInfo(False, "Show labels for main tabs"),
    "uiux_show_labels_tabs": shared.OptionInfo(False, "Show labels for page tabs"),
    "uiux_ignore_overrides": shared.OptionInfo([], "Ignore Overrides", gr.CheckboxGroup, lambda: {"choices": list(mapping)})
}))


basedir = scripts.basedir() 
html_folder = os.path.join(basedir, "html") 

layouts_folder = os.path.join(basedir, "layouts") 
javascript_folder = os.path.join(basedir, "javascript")

def get_files(folder, file_filter=[], file_list=[], split=False):
    file_list = [file_name if not split else os.path.splitext(file_name)[0] for file_name in os.listdir(folder) if os.path.isfile(os.path.join(folder, file_name)) and file_name not in file_filter] 
    return file_list

html = """
<html>
  <body>
    <h1>My First JavaScript</h1>
    <button type="testButton" onclick="testFn()"> Start </button>
    <div id="root-dock-no"></div>
  </body>
</html>
"""

scripts = """
async () => {
   // set testFn() function on globalThis, so you html onlclick can access it
    globalThis.testFn = () => {
      document.getElementById('demo').innerHTML = "Hello"
    }
}
"""

# with gr.Blocks() as demo:   
    # input_mic = gr.HTML(html)
    # out_text  = gr.Textbox()
    # # run script function on load,
    # demo.load(None,None,None,_js=scripts)
# static_dir = Path('./static')
# static_dir.mkdir(parents=True, exist_ok=True)    
    
# def predict(text_input):
    # file_name = f"{datetime.utcnow().strftime('%s')}.html"
    # file_path = html_folder / file_name
    # print(file_path)
    # with open(file_path, "w") as f:
        # f.write(f"""
        # <script src="https://cdn.tailwindcss.com"></script>
        # <body>
        # <div id=q-app></div>
        # <h1 class="text-3xl font-bold">Hello <i>{text_input}</i> From Gradio Iframe</h1>
        # <h3>Filename: {file_name}</h3>
        # </body>
        # """)
    # iframe = f"""<iframe src="file={file_path}" width="100%" height="500px"></iframe>"""
    # link = f'<a href="file={file_path}" target="_blank">{file_name}</a>'
    # return link, iframe



def on_ui_tabs():

    #print(mapping)

    with gr.Blocks(analytics_enabled=False) as anapnoe_sd_uiux_core: 
        #override_settings = gr.CheckboxGroup(label="Ignore override settings", elem_id="ignore_overrides", choices=list(mapping))
        """ with gr.Row():
            with gr.Column(): 
                with gr.Row():            
                    layouts_dropdown = gr.Dropdown(label="Layout", elem_id="layout_drop_down", interactive=True, choices=get_files(layouts_folder,[".json"]), type="value")
                    layout_save_as_filename = gr.Text(label="Save / Save as", elem_id="layout_save_as_name",)
                with gr.Row(): 
                    layout_reset_button = gr.Button(elem_id="layout_reset_btn", value="Reset", variant="primary")
                    layout_save_button = gr.Button(value="Save", elem_id="layout_save_btn", variant="primary")           
            with gr.Row(elem_id="layout_hidden"):
                layout_json = gr.Textbox(label="Json", elem_id="layout_json", show_label=True, lines=7, interactive=False, visible=True)  

        def save_layout( layout_json, filename):                
            with open(os.path.join(layouts_folder, f"{filename}.json"), 'w', encoding="utf-8") as file:                
                file.write(layout_json)
                file.close()       
            layouts_dropdown.choices=get_files(layouts_folder,[".json"])
            return gr.update(choices=layouts_dropdown.choices, value=f"{filename}.json")

        def open_layout(filename):                           
            with open(os.path.join(layouts_folder, f"{filename}"), 'r') as file:
                layout_json=file.read()
            no_ext=filename.rsplit('.', 1)[0]
            return [layout_json, no_ext]
               
        
        layout_save_button.click(
            fn=save_layout,
            inputs=[layout_json, layout_save_as_filename],
            outputs=layouts_dropdown
        )
        
        layouts_dropdown.change(
            fn=open_layout,
            inputs=[layouts_dropdown],
            outputs=[layout_json, layout_save_as_filename]
        )      """
                         


    return (anapnoe_sd_uiux_core, 'UI-UX Core', 'anapnoe_sd_uiux_core'),



script_callbacks.on_ui_tabs(on_ui_tabs)