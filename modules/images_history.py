import os
import shutil
import time
import hashlib
import gradio
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
        if all_type or (curr_path[-10:].rfind(".") > 0 and curr_path[-4:] != ".txt" and curr_path[-4:] != ".csv"):
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

def auto_sorting(dir_name):    
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
        date_str = time.strftime("%Y%m%d",time.localtime(os.path.getmtime(file)))
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
            try:
                reduplicative_file_move(os.path.join(dir_name, f), bak_path)  
            except:
                pass         
            
    today = time.strftime("%Y%m%d",time.localtime(time.time()))
    if today not in date_list:
        date_list.append(today)
    return sorted(date_list, reverse=True)

def archive_images(dir_name, date_to): 
    filenames = []   
    loads_num =int(opts.images_history_num_per_page * opts.images_history_pages_num)
    if opts.images_history_reconstruct_directory:        
        date_list = auto_sorting(dir_name)
        today = time.strftime("%Y%m%d",time.localtime(time.time()))
        date_to = today if date_to is None  or date_to == "" else date_to        
        for date in date_list:
            if date <= date_to:
                path = os.path.join(dir_name, date)
                if date == today and not os.path.exists(path):
                    continue
                filenames = traverse_all_files(path, filenames)
            if len(filenames) > loads_num:            
                break
        filenames = sorted(filenames, key=lambda file: -os.path.getmtime(file))
    else:
        filenames = traverse_all_files(dir_name, filenames)       
        tmparray = [(os.path.getmtime(file), file) for file in filenames ]
        date_stamp = time.mktime(time.strptime(date_to, "%Y%m%d")) + 86400      
        filenames = []
        date_list = {}
        date = time.strftime("%Y%m%d",time.localtime(time.time()))
        for t, f in tmparray:
            date = time.strftime("%Y%m%d",time.localtime(t))
            date_list[date] = None
            if t <= date_stamp:
                filenames.append((t, f ,date))
        date_list = sorted(list(date_list.keys()), reverse=True)
        sort_array = sorted(filenames, key=lambda x:-x[0])
        if len(sort_array) > loads_num:
            date = sort_array[loads_num][2]
            filenames = [x[1] for x in sort_array]
        else:
            date =  sort_array[-1][2]
            filenames = [x[1] for x in sort_array]
        filenames = [x[1] for x in sort_array if x[2]>= date]
    _, image_list, _, visible_num = get_recent_images(1, 0, filenames)
    return (
        gradio.Dropdown.update(choices=date_list, value=date_to),
        date,
        filenames,        
        1,
        image_list,
        "",
        visible_num
    )

def newest_click(dir_name,  date_to):
        return archive_images(dir_name, time.strftime("%Y%m%d",time.localtime(time.time())))

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

def save_image(file_name):
    if file_name is not None and os.path.exists(file_name):
        shutil.copy2(file_name, opts.outdir_save)

def get_recent_images(page_index, step, filenames):
    page_index = int(page_index)
    num_of_imgs_per_page = int(opts.images_history_num_per_page)
    max_page_index = len(filenames) // num_of_imgs_per_page + 1
    page_index = max_page_index if page_index == -1 else page_index + step
    page_index = 1 if page_index < 1 else page_index
    page_index = max_page_index if page_index > max_page_index else page_index
    idx_frm = (page_index - 1) * num_of_imgs_per_page
    image_list = filenames[idx_frm:idx_frm + num_of_imgs_per_page]
    length = len(filenames)
    visible_num = num_of_imgs_per_page if  idx_frm + num_of_imgs_per_page <= length else length % num_of_imgs_per_page 
    visible_num = num_of_imgs_per_page if visible_num == 0 else visible_num
    return page_index, image_list,  "", visible_num

def first_page_click(page_index, filenames):
    return get_recent_images(1, 0, filenames)

def end_page_click(page_index, filenames):
    return get_recent_images(-1, 0, filenames)

def prev_page_click(page_index, filenames):
    return get_recent_images(page_index, -1, filenames)

def next_page_click(page_index, filenames):
    return get_recent_images(page_index, 1, filenames)

def page_index_change(page_index, filenames):
    return get_recent_images(page_index, 0, filenames)

def show_image_info(tabname_box, num, page_index, filenames):
    file = filenames[int(num) + int((page_index - 1) * int(opts.images_history_num_per_page))]     
    return file, num, file

def enable_page_buttons():
    return gradio.update(visible=True)

def show_images_history(gr, opts, tabname, run_pnginfo, switch_dict):
    if tabname == "txt2img":
        dir_name = opts.outdir_txt2img_samples
    elif tabname == "img2img":
        dir_name = opts.outdir_img2img_samples
    elif tabname == "extras":
        dir_name = opts.outdir_extras_samples
    elif tabname == "saved":
        dir_name = opts.outdir_save

    d = dir_name.split("/")
    dir_name = d[0]
    for p in d[1:]:
        dir_name = os.path.join(dir_name, p)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with gr.Column() as page_panel:
        with gr.Row(visible=False) as turn_page_buttons:
            renew_page = gr.Button('Refresh page', elem_id=tabname + "_images_history_renew_page")
            first_page = gr.Button('First Page')
            prev_page = gr.Button('Prev Page')
            page_index = gr.Number(value=1, label="Page Index")
            next_page = gr.Button('Next Page')
            end_page = gr.Button('End Page')

        with gr.Row(elem_id=tabname + "_images_history"):
            with gr.Column(scale=2):
                with gr.Row():
                    newest = gr.Button('Reload', elem_id=tabname + "_images_history_start")                    
                    date_from = gr.Textbox(label="Date from", interactive=False)  
                    date_to = gr.Dropdown(label="Date to")                  

                history_gallery = gr.Gallery(show_label=False, elem_id=tabname + "_images_history_gallery").style(grid=6)
                with gr.Row():
                    delete_num = gr.Number(value=1, interactive=True, label="number of images to delete consecutively next")
                    delete = gr.Button('Delete', elem_id=tabname + "_images_history_del_button")
                    
            with gr.Column():
                with gr.Row():
                    if tabname != "saved":
                        save_btn = gr.Button('Save')
                    pnginfo_send_to_txt2img = gr.Button('Send to txt2img')
                    pnginfo_send_to_img2img = gr.Button('Send to img2img')
                with gr.Row():
                    with gr.Column():
                        img_file_info = gr.Textbox(label="Generate Info", interactive=False)
                        img_file_name = gr.Textbox(value="", label="File Name", interactive=False)

                        
                    # hiden items
                    with gr.Row(visible=False):   
                        visible_img_num = gr.Number()                     
                        img_path = gr.Textbox(dir_name)
                        tabname_box = gr.Textbox(tabname)
                        image_index = gr.Textbox(value=-1)
                        set_index = gr.Button('set_index', elem_id=tabname + "_images_history_set_index")
                        filenames = gr.State()
                        all_images_list = gr.State()
                        hidden = gr.Image(type="pil")
                        info1 = gr.Textbox()
                        info2 = gr.Textbox()

    
    #change date
    change_date_output = [date_to, date_from, filenames, page_index, history_gallery, img_file_name, visible_img_num]  
    newest.click(newest_click, inputs=[img_path, date_to], outputs=change_date_output)
    date_to.change(archive_images, inputs=[img_path, date_to], outputs=change_date_output)     
    newest.click(fn=None, inputs=[tabname_box], outputs=None, _js="images_history_turnpage")
    date_to.change(fn=None, inputs=[tabname_box], outputs=None, _js="images_history_turnpage")
    date_to.change(enable_page_buttons, inputs=None, outputs=[turn_page_buttons])
    newest.click(enable_page_buttons, inputs=None, outputs=[turn_page_buttons])

    #delete
    delete.click(delete_image, inputs=[delete_num, img_file_name, filenames, image_index, visible_img_num], outputs=[filenames, delete_num, visible_img_num])
    delete.click(fn=None, _js="images_history_delete", inputs=[delete_num, tabname_box, image_index], outputs=None) 
    if tabname != "saved": 
        save_btn.click(save_image, inputs=[img_file_name], outputs=None)     

    #turn page
    gallery_inputs = [page_index, filenames]
    gallery_outputs = [page_index, history_gallery, img_file_name, visible_img_num]
    first_page.click(first_page_click, inputs=gallery_inputs, outputs=gallery_outputs)
    next_page.click(next_page_click, inputs=gallery_inputs, outputs=gallery_outputs)
    prev_page.click(prev_page_click, inputs=gallery_inputs, outputs=gallery_outputs)
    end_page.click(end_page_click, inputs=gallery_inputs, outputs=gallery_outputs)
    page_index.submit(page_index_change, inputs=gallery_inputs, outputs=gallery_outputs)
    renew_page.click(page_index_change, inputs=gallery_inputs, outputs=gallery_outputs)

    first_page.click(fn=None, inputs=[tabname_box], outputs=None, _js="images_history_turnpage")
    next_page.click(fn=None, inputs=[tabname_box], outputs=None, _js="images_history_turnpage")
    prev_page.click(fn=None, inputs=[tabname_box], outputs=None, _js="images_history_turnpage")
    end_page.click(fn=None, inputs=[tabname_box], outputs=None, _js="images_history_turnpage")
    page_index.submit(fn=None, inputs=[tabname_box], outputs=None, _js="images_history_turnpage")
    renew_page.click(fn=None, inputs=[tabname_box], outputs=None, _js="images_history_turnpage")

    # other funcitons
    set_index.click(show_image_info, _js="images_history_get_current_img", inputs=[tabname_box, image_index, page_index, filenames], outputs=[img_file_name, image_index, hidden])
    img_file_name.change(fn=None, _js="images_history_enable_del_buttons", inputs=None, outputs=None)   
    hidden.change(fn=run_pnginfo, inputs=[hidden], outputs=[info1, img_file_info, info2])
    switch_dict["fn"](pnginfo_send_to_txt2img, switch_dict["t2i"], img_file_info, 'switch_to_txt2img')
    switch_dict["fn"](pnginfo_send_to_img2img, switch_dict["i2i"], img_file_info, 'switch_to_img2img_img2img')

def create_history_tabs(gr, sys_opts, run_pnginfo, switch_dict):
    global opts;
    opts = sys_opts
    loads_files_num = int(opts.images_history_num_per_page)
    num_of_imgs_per_page = int(opts.images_history_num_per_page * opts.images_history_pages_num)
    with gr.Blocks(analytics_enabled=False) as images_history:
        with gr.Tabs() as tabs:
            for tab in ["txt2img", "img2img", "extras", "saved"]:
                with gr.Tab(tab):
                    with gr.Blocks(analytics_enabled=False) as images_history_img2img:
                        show_images_history(gr, opts, tab, run_pnginfo, switch_dict) 
        gradio.Checkbox(opts.images_history_reconstruct_directory, elem_id="images_history_reconstruct_directory", visible=False)
    
    return images_history
