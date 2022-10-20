import os
import random
import gradio 
inspiration_path = "inspiration"
inspiration_system_path = os.path.join(inspiration_path, "system")
def read_name_list(file):
    if not os.path.exists(file):
        return []
    f = open(file, "r")
    ret = []
    line = f.readline()
    while len(line) > 0:
        line = line.rstrip("\n")
        ret.append(line)
    print(ret)
    return ret

def save_name_list(file, name): 
    print(file)
    f = open(file, "a")
    f.write(name + "\n")

def get_inspiration_images(source, types):   
    path = os.path.join(inspiration_path , types) 
    if source == "Favorites":
        names = read_name_list(os.path.join(inspiration_system_path, types + "_faverites.txt"))
        names = random.sample(names, 25)
    elif source == "Abandoned":
        names = read_name_list(os.path.join(inspiration_system_path, types + "_abondened.txt"))
        names = random.sample(names, 25)
    elif source == "Exclude abandoned":
        abondened = read_name_list(os.path.join(inspiration_system_path, types + "_abondened.txt"))        
        all_names = os.listdir(path)
        names = []
        while len(names) < 25:
            name = random.choice(all_names)
            if name not in abondened:
                names.append(name)
    else:
        names = random.sample(os.listdir(path), 25)
    names = random.sample(names, 25)
    image_list = []
    for a in names:
        image_path = os.path.join(path, a)
        images = os.listdir(image_path)        
        image_list.append(os.path.join(image_path, random.choice(images)))
    return image_list, names

def select_click(index, types, name_list):
    name = name_list[int(index)]
    path = os.path.join(inspiration_path, types, name)
    images = os.listdir(path)
    return name, [os.path.join(path, x) for x in images]

def give_up_click(name, types):
    file = os.path.join(inspiration_system_path, types + "_abandoned.txt")
    name_list = read_name_list(file)
    if name not in name_list:
        save_name_list(file, name)
   
def collect_click(name, types):
    file = os.path.join(inspiration_system_path, types + "_faverites.txt")
    print(file)
    name_list = read_name_list(file)
    print(name_list)
    if name not in name_list:
        save_name_list(file, name)

def moveout_click(name, types):
    file = os.path.join(inspiration_system_path, types + "_faverites.txt")
    name_list = read_name_list(file)
    if name not in name_list:
        save_name_list(file, name)

def source_change(source):
    if source == "Abandoned" or source == "Favorites":
        return gradio.Button.update(visible=True, value=f"Move out {source}")
    else:
        return gradio.Button.update(visible=False)

def ui(gr, opts):     
    with gr.Blocks(analytics_enabled=False) as inspiration:
        flag = os.path.exists(inspiration_path)        
        if flag:
            types = os.listdir(inspiration_path)
            types = [x for x in types if x != "system"]
            flag = len(types) > 0
        if not flag:
            os.mkdir(inspiration_path)
            gr.HTML("""
                <div align='center' width="50%>You need get "inspiration" images first. You can create these images by run "Create inspiration images" script in txt2img page, or download zip file from here and unzip these file under fold "inpiration".</div>"
                """)           
            return inspiration
        if not os.path.exists(inspiration_system_path):
            os.mkdir(inspiration_system_path)
        gallery, names = get_inspiration_images("Exclude abandoned", types[0])   
        with gr.Row():
            with gr.Column(scale=2):                
                inspiration_gallery = gr.Gallery(gallery, show_label=False, elem_id="inspiration_gallery").style(grid=5, height='auto')
            with gr.Column(scale=1):
                types = gr.Dropdown(choices=types, value=types[0], label="Type", visible=len(types) > 1)
                with gr.Row():                    
                    source = gr.Dropdown(choices=["All", "Favorites", "Exclude abandoned", "Abandoned"], value="Exclude abandoned", label="Source")
                    get_inspiration = gr.Button("Get inspiration")
                name = gr.Textbox(show_label=False, interactive=False)
                with gr.Row(): 
                    send_to_txt2img = gr.Button('to txt2img')
                    send_to_img2img = gr.Button('to img2img')
                style_gallery = gr.Gallery(show_label=False, elem_id="inspiration_style_gallery").style(grid=2, height='auto')                          
               
                collect = gr.Button('Collect')     
                give_up = gr.Button("Don't show any more")
                moveout = gr.Button("Move out", visible=False)
        with gr.Row():
            select_button = gr.Button('set button', elem_id="inspiration_select_button")
            name_list = gr.State(names)
        source.change(source_change, inputs=[source], outputs=[moveout])
        get_inspiration.click(get_inspiration_images, inputs=[source, types], outputs=[inspiration_gallery, name_list])
        select_button.click(select_click, _js="inspiration_selected", inputs=[name, types, name_list], outputs=[name, style_gallery])
        give_up.click(give_up_click, inputs=[name, types], outputs=None)
        collect.click(collect_click, inputs=[name, types], outputs=None)
    return inspiration
