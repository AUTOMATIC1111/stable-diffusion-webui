import os
import random
import csv, os, shutil
import modules.scripts as scripts
from modules import processing, shared, sd_samplers, images
from modules.processing import Processed
from modules.shared import opts
import gradio as gr
import modules.ui
from modules import shared
from modules import script_callbacks
inspiration_dir = os.path.join(scripts.basedir(), "inspiration")
inspiration_system_path = os.path.join(inspiration_dir, "system")
class Script(scripts.Script):
    def title(self):
        return "Create inspiration images"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        file = gr.Files(label="Artist or styles name list. '.txt' files with one name per line",)
        with gr.Row():
            prefix = gr.Textbox("a painting in", label="Prompt words before artist or style name", file_count="multiple")
            suffix= gr.Textbox("style", label="Prompt words after artist or style name")
        negative_prompt = gr.Textbox("picture frame, portrait photo", label="Negative Prompt")
        with gr.Row():
            batch_size = gr.Number(1, label="Batch size")
            batch_count = gr.Number(2, label="Batch count")            
        return [batch_size, batch_count, prefix, suffix, negative_prompt, file]

    def run(self, p, batch_size, batch_count, prefix, suffix, negative_prompt, files):
        p.batch_size = int(batch_size)
        p.n_iterint = int(batch_count)
        p.negative_prompt = negative_prompt        
        p.do_not_save_samples = True
        p.do_not_save_grid = True        
        for file in files:
            tp = file.orig_name.split(".")[0]
            print(tp)
            path = os.path.join(inspiration_dir, tp)
            if not os.path.exists(path):
                os.makedirs(path) 
            f = open(file.name, "r", encoding='utf-8')
            line = f.readline() 
            while len(line) > 0: 
                name = line.rstrip("\n").split(",")[0]
                line = f.readline() 
                artist_path = os.path.join(path, name)  
                if not os.path.exists(artist_path):
                    os.mkdir(artist_path)  
                if len(os.listdir(artist_path)) >= opts.inspiration_max_samples:
                    continue
                p.prompt = f"{prefix} {name} {suffix}"
                print(p.prompt)
                processed = processing.process_images(p)
                for img in  processed.images:
                    i = 0
                    filename = os.path.join(artist_path,  format(0, "03d") + ".jpg")
                    while os.path.exists(filename):
                        i += 1
                        filename = os.path.join(artist_path,  format(i, "03d") + ".jpg")
                    img.save(filename, quality=80)
        return processed



def read_name_list(file, types=None, keyword=None):
    if not os.path.exists(file):
        return []
    ret = []
    f = open(file, "r")    
    line = f.readline()
    while len(line) > 0:
        line = line.rstrip("\n")
        if types is not None:
            dirname = os.path.split(line)
            if dirname[0] in types and keyword in dirname[1].lower():
                ret.append(line)
        else:
            ret.append(line)
        line = f.readline()
    return ret

def save_name_list(file, name): 
    name_list = read_name_list(file)
    if name not in name_list:
        with open(file, "a") as f:
            f.write(name + "\n")

def get_types_list():
    files = os.listdir(inspiration_dir)
    types = []
    for x in files:
        path = os.path.join(inspiration_dir, x)
        if x[0] == ".":
            continue
        if not os.path.isdir(path):
            continue
        if path == inspiration_system_path:
            continue
        types.append(x)
    return types

def get_inspiration_images(source, types, keyword):
    keyword = keyword.strip(" ").lower()
    get_num = int(opts.inspiration_rows_num * opts.inspiration_cols_num)
    if source == "Favorites":
        names = read_name_list(os.path.join(inspiration_system_path, "faverites.txt"), types, keyword)
        names = random.sample(names, get_num) if len(names) > get_num else names
    elif source == "Abandoned":
        names = read_name_list(os.path.join(inspiration_system_path, "abandoned.txt"), types, keyword)
        names = random.sample(names, get_num) if len(names) > get_num else names
    elif source == "Exclude abandoned":        
        abandoned = read_name_list(os.path.join(inspiration_system_path, "abandoned.txt"), types, keyword)  
        all_names = []
        for tp in types:
            name_list = os.listdir(os.path.join(inspiration_dir, tp))
            all_names += [os.path.join(tp, x) for x in name_list if keyword in x.lower()]
        
        if len(all_names) > get_num:
            names = []
            while len(names) < get_num:
                name = random.choice(all_names)
                if name not in abandoned:
                    names.append(name)
        else:
            names = all_names
    else:
        all_names = []
        for tp in types:
            name_list = os.listdir(os.path.join(inspiration_dir, tp))
            all_names += [os.path.join(tp, x) for x in name_list if keyword in x.lower()]
        names = random.sample(all_names, get_num) if len(all_names) > get_num else all_names
    image_list = []
    for a in names:
        image_path = os.path.join(inspiration_dir, a)
        images = os.listdir(image_path)
        if len(images) > 0:        
            image_list.append((os.path.join(image_path, random.choice(images)), a))
        else:
            print(image_path)
    return image_list, names

def select_click(index, name_list):
    name = name_list[int(index)]
    path = os.path.join(inspiration_dir, name)
    images = os.listdir(path)
    return name, [os.path.join(path, x) for x in images], ""

def give_up_click(name):
    file = os.path.join(inspiration_system_path, "abandoned.txt")
    save_name_list(file, name)
    return "Added to abandoned list"
   
def collect_click(name):
    file = os.path.join(inspiration_system_path, "faverites.txt")
    save_name_list(file, name)
    return "Added to faverite list"

def moveout_click(name, source):
    if source == "Abandoned":
        file = os.path.join(inspiration_system_path, "abandoned.txt")
    elif source == "Favorites":
        file = os.path.join(inspiration_system_path, "faverites.txt")
    else:
        return None
    name_list = read_name_list(file)
    os.remove(file)
    with open(file, "a") as f:
        for a in name_list:
            if a != name:
                f.write(a + "\n")
    return f"Moved out {name} from {source} list"

def source_change(source):
    if source in ["Abandoned", "Favorites"]:
        return gr.update(visible=True), []
    else:
        return gr.update(visible=False), []
def add_to_prompt(name, prompt):
    name = os.path.basename(name)
    return prompt + "," + name

def clear_keyword():
    return ""

def on_ui_tabs():     
    txt2img_prompt = modules.ui.txt2img_paste_fields[0][0]
    img2img_prompt = modules.ui.img2img_paste_fields[0][0]
    with gr.Blocks(analytics_enabled=False) as inspiration:
        flag = os.path.exists(inspiration_dir)        
        if flag:
            types = get_types_list()
            flag = len(types) > 0
        else:
            os.makedirs(inspiration_dir)
        if not flag:            
            gr.HTML("""
                <div align='center' width="50%"><h2>To activate inspiration function, you need get "inspiration" images first. </h2><br>
                You can create these images by run "Create inspiration images" script in txt2img page, <br> you can get the artists or art styles list from here<br>
                <a href="https://github.com/pharmapsychotic/clip-interrogator/tree/main/data">https://github.com/pharmapsychotic/clip-interrogator/tree/main/data</a><br>
                download these files, and select these files in the "Create inspiration images" script UI<br>
                There about 6000 artists and art styles in these files. <br>This takes server hours depending on your GPU type and how many pictures  you generate for each artist/style
                <br>I suggest at least four images for each<br><br><br>
                <h2>You can also download generated pictures from here:</h2><br>
                <a href="https://huggingface.co/datasets/yfszzx/inspiration">https://huggingface.co/datasets/yfszzx/inspiration</a><br>
                unzip the file to <stable-diffusion-webui>/extections/stable-diffusion-webui-inspiration<br>
                and restart webui, and enjoy the joy of creation!<br></div>
                """)      
            return  (inspiration, "Inspiration", "inspiration"),
        if not os.path.exists(inspiration_system_path):
            os.mkdir(inspiration_system_path)
        with gr.Row():
            with gr.Column(scale=2):                
                inspiration_gallery = gr.Gallery(show_label=False, elem_id="inspiration_gallery").style(grid=opts.inspiration_cols_num, height='auto')
            with gr.Column(scale=1):
                types = gr.CheckboxGroup(choices=types, value=types)
                with gr.Row(): 
                    source = gr.Dropdown(choices=["All", "Favorites", "Exclude abandoned", "Abandoned"], value="Exclude abandoned", label="Source")
                    keyword = gr.Textbox("", label="Key word")                                                
                get_inspiration = gr.Button("Get inspiration", elem_id="inspiration_get_button")   
                name = gr.Textbox(show_label=False, interactive=False)      
                with gr.Row():                     
                    send_to_txt2img = gr.Button('to txt2img')
                    send_to_img2img = gr.Button('to img2img')
                    collect = gr.Button('Collect')     
                    give_up = gr.Button("Don't show again")
                    moveout = gr.Button("Move out", visible=False)
                warning = gr.HTML()
                style_gallery = gr.Gallery(show_label=False).style(grid=2, height='auto') 
                    
               
                
        with gr.Row(visible=False):
            select_button = gr.Button('set button', elem_id="inspiration_select_button")
            name_list = gr.State()
        
        get_inspiration.click(get_inspiration_images, inputs=[source, types, keyword], outputs=[inspiration_gallery, name_list])
        keyword.submit(fn=None, _js="inspiration_click_get_button", inputs=None, outputs=None)
        source.change(source_change, inputs=[source], outputs=[moveout, style_gallery])
        source.change(fn=clear_keyword, _js="inspiration_click_get_button", inputs=None, outputs=[keyword])       
        types.change(fn=clear_keyword, _js="inspiration_click_get_button", inputs=None, outputs=[keyword])  

        select_button.click(select_click, _js="inspiration_selected", inputs=[name, name_list], outputs=[name, style_gallery, warning])
        give_up.click(give_up_click, inputs=[name], outputs=[warning])
        collect.click(collect_click, inputs=[name], outputs=[warning])
        moveout.click(moveout_click, inputs=[name, source], outputs=[warning])
        moveout.click(fn=None, _js="inspiration_click_get_button", inputs=None, outputs=None)

        send_to_txt2img.click(add_to_prompt, inputs=[name, txt2img_prompt], outputs=[txt2img_prompt])
        send_to_img2img.click(add_to_prompt, inputs=[name, img2img_prompt], outputs=[img2img_prompt])
        send_to_txt2img.click(collect_click, inputs=[name], outputs=[warning])
        send_to_img2img.click(collect_click, inputs=[name], outputs=[warning])
        send_to_txt2img.click(None, _js='switch_to_txt2img', inputs=None, outputs=None)
        send_to_img2img.click(None, _js="switch_to_img2img_img2img", inputs=None, outputs=None)
    return (inspiration, "Inspiration", "inspiration"),

def on_ui_settings():
    section = ('inspiration', "Inspiration")
    shared.opts.add_option("inspiration_max_samples", shared.OptionInfo(4, "Maximum number of samples, used to determine which folders to skip when continue running the create script", gr.Slider, {"minimum": 1, "maximum": 20, "step": 1}, section=section))
    shared.opts.add_option("inspiration_rows_num", shared.OptionInfo(4, "Number of rows on the page",  gr.Slider, {"minimum": 4, "maximum": 16, "step": 1}, section=section))
    shared.opts.add_option( "inspiration_cols_num", shared.OptionInfo(6, "Minimum number of pages per load", gr.Slider, {"minimum": 4, "maximum": 16, "step": 1}, section=section))


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)

