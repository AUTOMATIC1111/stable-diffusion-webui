import os
import shutil
import time
import hashlib
import gradio
show_max_dates_num = 3
system_bak_path = "webui_log_and_bak"
def is_valid_date(date):
    try:
        time.strptime(date, "%Y%m%d")
        return True
    except:
        return False

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

def traverse_all_files(curr_path, image_list, all_type=False):
    try:
        f_list = os.listdir(curr_path)
    except:
        if all_type or curr_path[-10:].rfind(".") > 0 and curr_path[-4:] != ".txt":
            image_list.append(curr_path)
        return image_list
    for file in f_list:
        file = os.path.join(curr_path, file)
        if (not all_type) and file[-4:] == ".txt":
            pass
        elif os.path.isfile(file) and file[-10:].rfind(".") > 0:
            image_list.append(file)
        else:
            image_list = traverse_all_files(file, image_list)
    return image_list

def get_recent_images(dir_name, page_index, step, image_index, tabname, date_from, date_to):
    #print(f"turn_page {page_index}",date_from)
    if date_from is None or date_from == "":
        return None, 1, None, ""
    image_list = []
    date_list = auto_sorting(dir_name)
    page_index = int(page_index)
    today = time.strftime("%Y%m%d",time.localtime(time.time()))
    for date in date_list:
        if date >= date_from and date <= date_to:
            path = os.path.join(dir_name, date)
            if date == today and not os.path.exists(path):
                continue
            image_list = traverse_all_files(path, image_list)

    image_list = sorted(image_list, key=lambda file: -os.path.getctime(file))
    num = 48 if tabname != "extras" else 12
    max_page_index = len(image_list) // num + 1
    page_index = max_page_index if page_index == -1 else page_index + step
    page_index = 1 if page_index < 1 else page_index
    page_index = max_page_index if page_index > max_page_index else page_index
    idx_frm = (page_index - 1) * num
    image_list = image_list[idx_frm:idx_frm + num]
    image_index = int(image_index)
    if image_index < 0 or image_index > len(image_list) - 1:
        current_file = None
    else:
        current_file = image_list[image_index]
    return image_list, page_index, image_list,  ""

def auto_sorting(dir_name):
    #print(f"auto sorting")
    bak_path = os.path.join(dir_name, system_bak_path)
    if not os.path.exists(bak_path):
        os.mkdir(bak_path)
    log_file = None 
    files_list = []    
    f_list = os.listdir(dir_name)
    for file in f_list:   
        if file == system_bak_path:
            continue     
        file_path = os.path.join(dir_name, file)
        if not is_valid_date(file):
            if file[-10:].rfind(".") > 0:
                files_list.append(file_path)
            else:
                files_list = traverse_all_files(file_path, files_list, all_type=True)

    for file in files_list:        
        date_str = time.strftime("%Y%m%d",time.localtime(os.path.getctime(file)))
        file_path = os.path.dirname(file)
        hash_path = hashlib.md5(file_path.encode()).hexdigest()
        path = os.path.join(dir_name, date_str, hash_path)
        if not os.path.exists(path):
            os.makedirs(path)
        if log_file is None:
            log_file = open(os.path.join(bak_path,"path_mapping.csv"),"a") 
        log_file.write(f"{hash_path},{file_path}\n")
        reduplicative_file_move(file, path)
       
    date_list = []
    f_list = os.listdir(dir_name)
    for f in f_list:
        if is_valid_date(f):
            date_list.append(f)
        elif f == system_bak_path:
            continue
        else:
            reduplicative_file_move(os.path.join(dir_name, f), bak_path)           
            
    today = time.strftime("%Y%m%d",time.localtime(time.time()))
    if today not in date_list:
        date_list.append(today)
    return sorted(date_list)



def archive_images(dir_name):
    date_list = auto_sorting(dir_name)
    date_from = date_list[-show_max_dates_num] if len(date_list) > show_max_dates_num else date_list[0]
    return (
        gradio.update(visible=False), 
        gradio.update(visible=True), 
        gradio.Dropdown.update(choices=date_list, value=date_list[-1]),
        gradio.Dropdown.update(choices=date_list, value=date_from)
    )

def date_to_change(dir_name, page_index, image_index, tabname, date_from, date_to):
    #print("date_to", date_to)
    date_list = auto_sorting(dir_name)
    date_from_list = [date for date in date_list if date <= date_to]
    date_from = date_from_list[0] if len(date_from_list) < show_max_dates_num else date_from_list[-show_max_dates_num]
    image_list, page_index, image_list, _  =get_recent_images(dir_name, 1, 0, image_index, tabname, date_from, date_to)
    return image_list, page_index, image_list, _, gradio.Dropdown.update(choices=date_from_list, value=date_from)

def first_page_click(dir_name, page_index, image_index, tabname, date_from, date_to):
    return get_recent_images(dir_name, 1, 0, image_index, tabname, date_from, date_to)


def end_page_click(dir_name, page_index, image_index, tabname, date_from, date_to):
    return get_recent_images(dir_name, -1, 0, image_index, tabname, date_from, date_to)


def prev_page_click(dir_name, page_index, image_index, tabname, date_from, date_to):
    return get_recent_images(dir_name, page_index, -1, image_index, tabname, date_from, date_to)


def next_page_click(dir_name, page_index, image_index, tabname, date_from, date_to):
    return get_recent_images(dir_name, page_index, 1, image_index, tabname, date_from, date_to)


def page_index_change(dir_name, page_index, image_index, tabname, date_from, date_to):
    return get_recent_images(dir_name, page_index, 0, image_index, tabname, date_from, date_to)


def show_image_info(tabname_box, num, filenames):
    # #print(f"select image {num}")
    file = filenames[int(num)]
    return file, num, file

def delete_image(delete_num, tabname, name, page_index, filenames, image_index):
    if name == "":
        return filenames, delete_num
    else:
        delete_num = int(delete_num)
        index = list(filenames).index(name)
        i = 0
        new_file_list = []
        for name in filenames:
            if i >= index and i < index + delete_num:
                if os.path.exists(name):
                    #print(f"Delete file {name}")
                    os.remove(name)
                    txt_file = os.path.splitext(name)[0] + ".txt"
                    if os.path.exists(txt_file):
                        os.remove(txt_file)
                else:
                    #print(f"Not exists file {name}")
            else:
                new_file_list.append(name)
            i += 1
    return new_file_list, 1

def show_images_history(gr, opts, tabname, run_pnginfo, switch_dict):
    if tabname == "txt2img":
        dir_name = opts.outdir_txt2img_samples
    elif tabname == "img2img":
        dir_name = opts.outdir_img2img_samples
    elif tabname == "extras":
        dir_name = opts.outdir_extras_samples
    d = dir_name.split("/")
    dir_name = d[0]
    for p in d[1:]:
        dir_name = os.path.join(dir_name, p)

    f_list = os.listdir(dir_name)
    sorted_flag = os.path.exists(os.path.join(dir_name, system_bak_path)) or len(f_list) == 0 
    date_list, date_from, date_to = None, None, None
    if sorted_flag:
        #print(sorted_flag)
        date_list = auto_sorting(dir_name)
        date_to = date_list[-1]
        date_from = date_list[-show_max_dates_num] if len(date_list) > show_max_dates_num else date_list[0] 

    with gr.Column(visible=sorted_flag) as page_panel:
        with gr.Row():
            renew_page = gr.Button('Refresh', elem_id=tabname + "_images_history_renew_page", interactive=sorted_flag)
            first_page = gr.Button('First Page')
            prev_page = gr.Button('Prev Page')
            page_index = gr.Number(value=1, label="Page Index")
            next_page = gr.Button('Next Page')
            end_page = gr.Button('End Page')

        with gr.Row(elem_id=tabname + "_images_history"):
            with gr.Column(scale=2):
                with gr.Row():
                    newest = gr.Button('Newest')
                    date_to = gr.Dropdown(choices=date_list, value=date_to, label="Date to")
                    date_from = gr.Dropdown(choices=date_list, value=date_from, label="Date from")                    

                history_gallery = gr.Gallery(show_label=False, elem_id=tabname + "_images_history_gallery").style(grid=6)
                with gr.Row():
                    delete_num = gr.Number(value=1, interactive=True, label="number of images to delete consecutively next")
                    delete = gr.Button('Delete', elem_id=tabname + "_images_history_del_button")
            with gr.Column():
                with gr.Row():
                    pnginfo_send_to_txt2img = gr.Button('Send to txt2img')
                    pnginfo_send_to_img2img = gr.Button('Send to img2img')
                with gr.Row():
                    with gr.Column():
                        img_file_info = gr.Textbox(label="Generate Info", interactive=False)
                        img_file_name = gr.Textbox(value="", label="File Name", interactive=False)
                    # hiden items
                    with gr.Row(visible=False):
                        img_path = gr.Textbox(dir_name)
                        tabname_box = gr.Textbox(tabname)
                        image_index = gr.Textbox(value=-1)
                        set_index = gr.Button('set_index', elem_id=tabname + "_images_history_set_index")
                        filenames = gr.State()
                        hidden = gr.Image(type="pil")
                        info1 = gr.Textbox()
                        info2 = gr.Textbox()
    with gr.Column(visible=not sorted_flag) as init_warning:
        with gr.Row():
            gr.Textbox("The system needs to archive the files according to the date. This requires changing the directory structure of the files",
             label="Waring",
             css="")
        with gr.Row():
            sorted_button = gr.Button('Confirme')

                  
           
   
    # turn pages
    gallery_inputs = [img_path, page_index, image_index, tabname_box, date_from, date_to]
    gallery_outputs = [history_gallery, page_index, filenames, img_file_name]

    first_page.click(first_page_click, _js="images_history_turnpage", inputs=gallery_inputs, outputs=gallery_outputs)
    next_page.click(next_page_click, _js="images_history_turnpage", inputs=gallery_inputs, outputs=gallery_outputs)
    prev_page.click(prev_page_click, _js="images_history_turnpage", inputs=gallery_inputs, outputs=gallery_outputs)
    end_page.click(end_page_click, _js="images_history_turnpage", inputs=gallery_inputs, outputs=gallery_outputs)
    page_index.submit(page_index_change, _js="images_history_turnpage", inputs=gallery_inputs, outputs=gallery_outputs)
    renew_page.click(page_index_change, _js="images_history_turnpage", inputs=gallery_inputs, outputs=gallery_outputs)
    # page_index.change(page_index_change, inputs=[tabname_box, img_path,  page_index], outputs=[history_gallery, page_index])

    # other funcitons
    set_index.click(show_image_info, _js="images_history_get_current_img", inputs=[tabname_box, image_index, filenames], outputs=[img_file_name, image_index, hidden])
    img_file_name.change(fn=None, _js="images_history_enable_del_buttons", inputs=None, outputs=None)
    delete.click(delete_image, _js="images_history_delete", inputs=[delete_num, tabname_box, img_file_name, page_index, filenames, image_index], outputs=[filenames, delete_num])
    hidden.change(fn=run_pnginfo, inputs=[hidden], outputs=[info1, img_file_info, info2])
    date_to.change(date_to_change, _js="images_history_turnpage", inputs=gallery_inputs, outputs=gallery_outputs + [date_from])
    # pnginfo.click(fn=run_pnginfo, inputs=[hidden], outputs=[info1, img_file_info, info2])
    switch_dict["fn"](pnginfo_send_to_txt2img, switch_dict["t2i"], img_file_info, 'switch_to_txt2img')
    switch_dict["fn"](pnginfo_send_to_img2img, switch_dict["i2i"], img_file_info, 'switch_to_img2img_img2img')

    sorted_button.click(archive_images, inputs=[img_path], outputs=[init_warning, page_panel, date_to, date_from])
    newest.click(archive_images, inputs=[img_path], outputs=[init_warning, page_panel, date_to, date_from])
   
    
    


def create_history_tabs(gr, opts, run_pnginfo, switch_dict):
    with gr.Blocks(analytics_enabled=False) as images_history:
        with gr.Tabs() as tabs:
            with gr.Tab("txt2img history"):
                with gr.Blocks(analytics_enabled=False) as images_history_txt2img:
                    show_images_history(gr, opts, "txt2img", run_pnginfo, switch_dict)
            with gr.Tab("img2img history"):
                with gr.Blocks(analytics_enabled=False) as images_history_img2img:
                    show_images_history(gr, opts, "img2img", run_pnginfo, switch_dict)
            with gr.Tab("extras history"):
                with gr.Blocks(analytics_enabled=False) as images_history_img2img:
                    show_images_history(gr, opts, "extras", run_pnginfo, switch_dict)
    return images_history
