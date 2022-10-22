import os
import random
import gradio
from modules.shared import opts
inspiration_system_path = os.path.join(opts.inspiration_dir, "system")
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
    files = os.listdir(opts.inspiration_dir)
    types = []
    for x in files:
        path = os.path.join(opts.inspiration_dir, x)
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
            name_list = os.listdir(os.path.join(opts.inspiration_dir, tp))
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
            name_list = os.listdir(os.path.join(opts.inspiration_dir, tp))
            all_names += [os.path.join(tp, x) for x in name_list if keyword in x.lower()]
        names = random.sample(all_names, get_num) if len(all_names) > get_num else all_names
    image_list = []
    for a in names:
        image_path = os.path.join(opts.inspiration_dir, a)
        images = os.listdir(image_path)        
        image_list.append((os.path.join(image_path, random.choice(images)), a))
    return image_list, names

def select_click(index, name_list):
    name = name_list[int(index)]
    path = os.path.join(opts.inspiration_dir, name)
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
        return gradio.update(visible=True), []
    else:
        return gradio.update(visible=False), []
def add_to_prompt(name, prompt):
    name = os.path.basename(name)
    return prompt + "," + name

def clear_keyword():
    return ""

def ui(gr, opts, txt2img_prompt, img2img_prompt):     
    with gr.Blocks(analytics_enabled=False) as inspiration:
        flag = os.path.exists(opts.inspiration_dir)        
        if flag:
            types = get_types_list()
            flag = len(types) > 0
        else:
            os.makedirs(opts.inspiration_dir)
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
                unzip the file to the project directory of webui<br>
                and restart webui, and enjoy the joy of creation!<br></div>
                """)      
            return inspiration
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
                style_gallery = gr.Gallery(show_label=False).style(grid=2, height='auto') 
                warning = gr.HTML()
                with gr.Row():
                    collect = gr.Button('Collect')     
                    give_up = gr.Button("Don't show again")
                moveout = gr.Button("Move out", visible=False)
                
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
    return inspiration
