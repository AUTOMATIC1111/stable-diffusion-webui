import os
import shutil
import time
import hashlib
import gradio as gr
import modules.extras
import modules.ui
from modules.shared import opts, cmd_opts
from modules import shared, scripts
from modules import script_callbacks
from pathlib import Path

faverate_tab_name = "favorites"
tabs_list = ["txt2img", "img2img", "extras", faverate_tab_name, "others"]
num_of_imgs_per_page = 0
loads_files_num = 0
path_recorder_filename = os.path.join(scripts.basedir(), "path_recorder.txt")
image_ext_list = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"]

def reduplicative_file_move(src, dst):
    def same_name_file(basename, path):
        name, ext = os.path.splitext(basename)
        f_list = os.listdir(path)
        max_num = 0
        for f in f_list:
            if len(f) <= len(basename):
                continue
            f_ext = f[-len(ext):] if len(ext) > 0 else ""
            if f[:len(name)] == name and f_ext == ext:                
                if f[len(name)] == "(" and f[-len(ext)-1] == ")":
                    number = f[len(name)+1:-len(ext)-1]
                    if number.isdigit():
                        if int(number) > max_num:
                            max_num = int(number)
        return f"{name}({max_num + 1}){ext}"
    name = os.path.basename(src)
    save_name = os.path.join(dst, name)
    if not os.path.exists(save_name):
        shutil.move(src, dst)
    else:
        name = same_name_file(name, dst)
        shutil.move(src, os.path.join(dst, name))

def save_image(file_name):
    if file_name is not None and os.path.exists(file_name):
        reduplicative_file_move(file_name, opts.outdir_save)
        return "<div style='color:#999'>Added to faverites</div>"

def delete_image(delete_num, name, filenames, image_index, visible_num):
    if name == "":
        return filenames, delete_num
    else:
        delete_num = int(delete_num)
        visible_num = int(visible_num)
        image_index = int(image_index)
        index = list(filenames).index(name)
        i = 0
        new_file_list = []
        for name in filenames:
            if i >= index and i < index + delete_num:
                if os.path.exists(name):
                    if visible_num == image_index:
                        new_file_list.append(name)
                        i += 1
                        continue                    
                    print(f"Delete file {name}")
                    os.remove(name)
                    visible_num -= 1
                    txt_file = os.path.splitext(name)[0] + ".txt"
                    if os.path.exists(txt_file):
                        os.remove(txt_file)
                else:
                    print(f"Not exists file {name}")
            else:
                new_file_list.append(name)
            i += 1
    return new_file_list, 1, visible_num

def traverse_all_files(curr_path, image_list):
    try:
        f_list = os.listdir(curr_path)
    except:
        if os.path.splitext(curr_path)[1] in image_ext_list:
            image_list.append(curr_path)
        return image_list
    for file in f_list:
        file = os.path.join(curr_path, file)
        if os.path.isfile(file) and os.path.splitext(file)[1] in image_ext_list:
            image_list.append(file)
        else:
            image_list = traverse_all_files(file, image_list)
    return image_list

def get_all_images(dir_name, sort_by, keyword):    
    filenames = []   
    filenames = traverse_all_files(dir_name, filenames)  
    keyword = keyword.strip(" ")
    if len(keyword) != 0:        
        filenames = [x for x in filenames if keyword in x]
    total_num = len(filenames) 
    if sort_by == "date":
        filenames = [(os.path.getmtime(file), file) for file in filenames ]
        sort_array = sorted(filenames, key=lambda x:-x[0])
        filenames = [x[1] for x in sort_array]
    elif sort_by == "path name":
        sort_array = sorted(filenames)        
    return filenames

def get_image_page(img_path, page_index, filenames, keyword, sort_by):
    if page_index == 1 or page_index == 0 or len(filenames) == 0:
        filenames = get_all_images(img_path, sort_by, keyword)
    page_index = int(page_index)
    length = len(filenames)
    max_page_index = length // num_of_imgs_per_page + 1
    page_index = max_page_index if page_index == -1 else page_index
    page_index = 1 if page_index < 1 else page_index
    page_index = max_page_index if page_index > max_page_index else page_index
    idx_frm = (page_index - 1) * num_of_imgs_per_page
    image_list = filenames[idx_frm:idx_frm + num_of_imgs_per_page]
    
    visible_num = num_of_imgs_per_page if  idx_frm + num_of_imgs_per_page < length else length % num_of_imgs_per_page 
    visible_num = num_of_imgs_per_page if visible_num == 0 else visible_num

    load_info = "<div style='color:#999' align='center'>"
    load_info += f"{length} images in this directory, divided into {int((length + 1) // num_of_imgs_per_page  + 1)} pages"
    load_info += "</div>"
    return filenames, page_index, image_list,  "", "",  "", visible_num, load_info

def show_image_info(tabname_box, num, page_index, filenames):
    file = filenames[int(num) + int((page_index - 1) * num_of_imgs_per_page)]   
    tm =   "<div style='color:#999' align='right'>" + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(os.path.getmtime(file))) + "</div>"
    return file, tm, num, file, ""

def change_dir(img_dir, path_recorder, load_switch, img_path_history):
    warning = None
    try:
        if not cmd_opts.administrator:        
            head = os.path.realpath(".")
            real_path = os.path.realpath(img_dir)
            if len(real_path) < len(head) or real_path[:len(head)] != head:
                warning = f"You have not permission to visit {img_dir}. If you want visit all directories, add command line argument option '--administrator', <a style='color:#990' href='https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings'>More detail here</a>"
    except:
        pass  
    if warning is None:
        try:
            if os.path.exists(img_dir):
                try:
                    f = os.listdir(img_dir)                
                except:
                    warning = f"'{img_dir} is not a directory"
            else:
                warning = "The directory is not exist"
        except:
            warning = "The format of the directory is incorrect"   

    if warning is None: 
        if img_dir not in path_recorder:
            path_recorder.append(img_dir)
        if os.path.exists(path_recorder_filename):
            os.remove(path_recorder_filename)
        with open(path_recorder_filename, "a") as f:
            for x in path_recorder:
                f.write(x + "\n")
        return "", gr.update(visible=True), gr.Dropdown.update(choices=path_recorder, value=img_dir), path_recorder, img_dir
    else:
        return warning, gr.update(visible=False), img_path_history, path_recorder, load_switch

def create_tab(tabname):
    custom_dir = False
    path_recorder = []
    if tabname == "txt2img":
        dir_name = opts.outdir_txt2img_samples
    elif tabname == "img2img":
        dir_name = opts.outdir_img2img_samples
    elif tabname == "extras":
        dir_name = opts.outdir_extras_samples
    elif tabname == faverate_tab_name:
        dir_name = opts.outdir_save
    else:
        custom_dir = True
        dir_name = None        
        if os.path.exists(path_recorder_filename):
            with open(path_recorder_filename) as f:
                path = f.readline().rstrip("\n")
                while len(path) > 0:
                    path_recorder.append(path)
                    path = f.readline().rstrip("\n")

    if not custom_dir:
        dir_name = str(Path(dir_name))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    with gr.Row(visible= custom_dir): 
        img_path = gr.Textbox(dir_name, label="Images directory", placeholder="Input images directory", interactive=custom_dir)  
        img_path_history = gr.Dropdown(path_recorder)
        path_recorder = gr.State(path_recorder)
 
    with gr.Row(visible= not custom_dir, elem_id=tabname + "_images_history") as main_panel:
        with gr.Column():  
            with gr.Row():    
                with gr.Column(scale=2):     
                    with gr.Row():       
                        first_page = gr.Button('First Page')
                        prev_page = gr.Button('Prev Page')
                        page_index = gr.Number(value=1, label="Page Index")
                        next_page = gr.Button('Next Page')
                        end_page = gr.Button('End Page') 
                    history_gallery = gr.Gallery(show_label=False, elem_id=tabname + "_images_history_gallery").style(grid=opts.images_history_page_columns)
                    with gr.Row() as delete_panel:
                        delete_num = gr.Number(value=1, interactive=True, label="number of images to delete consecutively next")
                        delete = gr.Button('Delete', elem_id=tabname + "_images_history_del_button")
                        
                with gr.Column(): 
                    with gr.Row():  
                        sort_by = gr.Radio(value="date", choices=["path name", "date"], label="sort by")   
                        keyword = gr.Textbox(value="", label="keyword")                 
                    with gr.Row():
                        with gr.Column():
                            img_file_info = gr.Textbox(label="Generate Info", interactive=False, lines=6)
                            img_file_name = gr.Textbox(value="", label="File Name", interactive=False)
                            img_file_time= gr.HTML()
                    with gr.Row(elem_id=tabname + "_images_history_button_panel") as button_panel:
                        if tabname != faverate_tab_name:
                            save_btn = gr.Button('Collect')
                        send_to_txt2img = gr.Button('To txt2img')
                        send_to_img2img = gr.Button('To img2img')
                        send_to_inpaint = gr.Button('To inpaint')
                        send_to_extras = gr.Button('To extras')
                    with gr.Row():
                        collected_warning = gr.HTML()                       
                            
                    # hiden items
                    with gr.Row(visible=False): 
                        renew_page = gr.Button("Renew Page", elem_id=tabname + "_images_history_renew_page")
                        visible_img_num = gr.Number()                     
                        tabname_box = gr.Textbox(tabname)
                        image_index = gr.Textbox(value=-1)
                        set_index = gr.Button('set_index', elem_id=tabname + "_images_history_set_index")
                        filenames = gr.State([])
                        all_images_list = gr.State()
                        hidden = gr.Image(type="pil")
                        info1 = gr.Textbox()
                        info2 = gr.Textbox()
                        load_switch = gr.Textbox(value="load_switch", label="load_switch")
                        turn_page_switch = gr.Number(value=1, label="turn_page_switch")
    with gr.Row():                 
        warning_box = gr.HTML() 

    change_dir_outputs = [warning_box, main_panel, img_path_history, path_recorder, load_switch]
    img_path.submit(change_dir, inputs=[img_path, path_recorder, load_switch, img_path_history], outputs=change_dir_outputs)
    img_path_history.change(change_dir, inputs=[img_path_history, path_recorder, load_switch, img_path_history], outputs=change_dir_outputs)
    img_path_history.change(lambda x:x, inputs=[img_path_history], outputs=[img_path])

    #delete
    delete.click(delete_image, inputs=[delete_num, img_file_name, filenames, image_index, visible_img_num], outputs=[filenames, delete_num, visible_img_num])
    delete.click(fn=None, _js="images_history_delete", inputs=[delete_num, tabname_box, image_index], outputs=None) 
    if tabname != faverate_tab_name: 
        save_btn.click(save_image, inputs=[img_file_name], outputs=[collected_warning])     

    #turn page
    first_page.click(lambda s:(1, -s) , inputs=[turn_page_switch], outputs=[page_index, turn_page_switch])
    next_page.click(lambda p, s: (p + 1, -s), inputs=[page_index, turn_page_switch], outputs=[page_index, turn_page_switch])
    prev_page.click(lambda p, s: (p - 1, -s), inputs=[page_index, turn_page_switch], outputs=[page_index, turn_page_switch])
    end_page.click(lambda s: (-1, -s), inputs=[turn_page_switch], outputs=[page_index, turn_page_switch])    
    load_switch.change(lambda s:(1, -s), inputs=[turn_page_switch], outputs=[page_index, turn_page_switch])
    keyword.submit(lambda s:(1, -s), inputs=[turn_page_switch], outputs=[page_index, turn_page_switch])
    sort_by.change(lambda s:(1, -s), inputs=[turn_page_switch], outputs=[page_index, turn_page_switch])
    page_index.submit(lambda s: -s, inputs=[turn_page_switch], outputs=[turn_page_switch])
    renew_page.click(lambda s: -s, inputs=[turn_page_switch], outputs=[turn_page_switch])

    turn_page_switch.change(
        fn=get_image_page, 
        inputs=[img_path, page_index, filenames, keyword, sort_by], 
        outputs=[filenames, page_index, history_gallery, img_file_name, img_file_time, img_file_info, visible_img_num, warning_box]
    )
    turn_page_switch.change(fn=None, inputs=[tabname_box], outputs=None, _js="images_history_turnpage")
    turn_page_switch.change(fn=lambda:(gr.update(visible=False), gr.update(visible=False)), inputs=None, outputs=[delete_panel, button_panel])

    # other funcitons
    set_index.click(show_image_info, _js="images_history_get_current_img", inputs=[tabname_box, image_index, page_index, filenames], outputs=[img_file_name, img_file_time, image_index, hidden])
    set_index.click(fn=lambda:(gr.update(visible=True), gr.update(visible=True)), inputs=None, outputs=[delete_panel, button_panel]) 
    img_file_name.change(fn=lambda : "", inputs=None, outputs=[collected_warning])  
   
    hidden.change(fn=modules.extras.run_pnginfo, inputs=[hidden], outputs=[info1, img_file_info, info2])


    modules.generation_parameters_copypaste.connect_paste(send_to_txt2img, modules.ui.txt2img_paste_fields, img_file_info, 'switch_to_txt2img')
    modules.generation_parameters_copypaste.connect_paste(send_to_img2img, modules.ui.img2img_paste_fields, img_file_info, 'switch_to_img2img_img2img')
    modules.generation_parameters_copypaste.connect_paste(send_to_inpaint, modules.ui.img2img_paste_fields, img_file_info, 'switch_to_img2img_inpaint')
    try:
        send_to_img2img.click(lambda x: x, inputs=[img_file_name], outputs=[modules.ui.init_img_components["img2img"]])
        send_to_inpaint.click(lambda x: x, inputs=[img_file_name], outputs=[modules.ui.init_img_components["inpaint"]])
        send_to_extras.click(lambda x: x, inputs=[img_file_name], outputs=[modules.ui.init_img_components["extras"]])
        send_to_extras.click(fn=None, _js="switch_to_extras", inputs=None, outputs=None)
    except:
        pass

def on_ui_tabs():
    global num_of_imgs_per_page
    global loads_files_num
    num_of_imgs_per_page = int(opts.images_history_page_columns * opts.images_history_page_rows)
    loads_files_num = int(opts.images_history_pages_perload * num_of_imgs_per_page)
    with gr.Blocks(analytics_enabled=False) as images_history:
        with gr.Tabs(elem_id="images_history_tab") as tabs:
            for tab in tabs_list:
                with gr.Tab(tab):
                    with gr.Blocks(analytics_enabled=False) :
                        create_tab(tab)
        gr.Checkbox(opts.images_history_preload, elem_id="images_history_preload", visible=False)         
        gr.Textbox(",".join(tabs_list), elem_id="images_history_tabnames_list", visible=False) 
    return (images_history , "Image Browser", "images_history"),

def on_ui_settings():
    section = ('images-history', "Images Browser")
    shared.opts.add_option("images_history_preload", shared.OptionInfo(False, "Preload images at startup", section=section))
    shared.opts.add_option("images_history_page_columns", shared.OptionInfo(6, "Number of columns on the page", section=section))
    shared.opts.add_option("images_history_page_rows", shared.OptionInfo(6, "Number of rows on the page", section=section))
    shared.opts.add_option("images_history_pages_perload", shared.OptionInfo(20, "Minimum number of pages per load", section=section))


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)

#TODO:
#send to remove seed
#generate info in txt