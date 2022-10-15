import os
import shutil


def traverse_all_files(output_dir, image_list, curr_dir=None):
    curr_path = output_dir if curr_dir is None else os.path.join(output_dir, curr_dir)
    try:
        f_list = os.listdir(curr_path)
    except:
        if curr_dir[-10:].rfind(".") > 0 and curr_dir[-4:] != ".txt":
            image_list.append(curr_dir)
        return image_list
    for file in f_list:
        file = file if curr_dir is None else os.path.join(curr_dir, file)
        file_path = os.path.join(curr_path, file)
        if file[-4:] == ".txt":
            pass
        elif os.path.isfile(file_path) and file[-10:].rfind(".") > 0:
            image_list.append(file)
        else:
            image_list = traverse_all_files(output_dir, image_list, file)
    return image_list


def get_recent_images(dir_name, page_index, step, image_index, tabname):
    page_index = int(page_index)
    f_list = os.listdir(dir_name)
    image_list = []
    image_list = traverse_all_files(dir_name, image_list)
    image_list = sorted(image_list, key=lambda file: -os.path.getctime(os.path.join(dir_name, file)))
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
        hidden = None
    else:
        current_file = image_list[int(image_index)]
        hidden = os.path.join(dir_name, current_file)
    return [os.path.join(dir_name, file) for file in image_list], page_index, image_list, current_file, hidden, ""


def first_page_click(dir_name, page_index, image_index, tabname):
    return get_recent_images(dir_name, 1, 0, image_index, tabname)


def end_page_click(dir_name, page_index, image_index, tabname):
    return get_recent_images(dir_name, -1, 0, image_index, tabname)


def prev_page_click(dir_name, page_index, image_index, tabname):
    return get_recent_images(dir_name, page_index, -1, image_index, tabname)


def next_page_click(dir_name, page_index, image_index, tabname):
    return get_recent_images(dir_name, page_index, 1, image_index, tabname)


def page_index_change(dir_name, page_index, image_index, tabname):
    return get_recent_images(dir_name, page_index, 0, image_index, tabname)


def show_image_info(num, image_path, filenames):
    # print(f"select image {num}")
    file = filenames[int(num)]
    return file, num, os.path.join(image_path, file)


def delete_image(delete_num, tabname, dir_name, name, page_index, filenames, image_index):
    if name == "":
        return filenames, delete_num
    else:
        delete_num = int(delete_num)
        index = list(filenames).index(name)
        i = 0
        new_file_list = []
        for name in filenames:
            if i >= index and i < index + delete_num:
                path = os.path.join(dir_name, name)
                if os.path.exists(path):
                    print(f"Delete file {path}")
                    os.remove(path)
                    txt_file = os.path.splitext(path)[0] + ".txt"
                    if os.path.exists(txt_file):
                        os.remove(txt_file)
                else:
                    print(f"Not exists file {path}")
            else:
                new_file_list.append(name)
            i += 1
    return new_file_list, 1


def show_images_history(gr, opts, tabname, run_pnginfo, switch_dict):
    if opts.outdir_samples != "":
        dir_name = opts.outdir_samples
    elif tabname == "txt2img":
        dir_name = opts.outdir_txt2img_samples
    elif tabname == "img2img":
        dir_name = opts.outdir_img2img_samples
    elif tabname == "extras":
        dir_name = opts.outdir_extras_samples
    d = dir_name.split("/")
    dir_name = "/" if dir_name.startswith("/") else d[0]
    for p in d[1:]:
        dir_name = os.path.join(dir_name, p)
    with gr.Row():
        renew_page = gr.Button('Renew Page', elem_id=tabname + "_images_history_renew_page")
        first_page = gr.Button('First Page')
        prev_page = gr.Button('Prev Page')
        page_index = gr.Number(value=1, label="Page Index")
        next_page = gr.Button('Next Page')
        end_page = gr.Button('End Page')
    with gr.Row(elem_id=tabname + "_images_history"):
        with gr.Row():
            with gr.Column(scale=2):
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
                        img_file_name = gr.Textbox(label="File Name", interactive=False)
                with gr.Row():
                    # hiden items

                    img_path = gr.Textbox(dir_name.rstrip("/"), visible=False)
                    tabname_box = gr.Textbox(tabname, visible=False)
                    image_index = gr.Textbox(value=-1, visible=False)
                    set_index = gr.Button('set_index', elem_id=tabname + "_images_history_set_index", visible=False)
                    filenames = gr.State()
                    hidden = gr.Image(type="pil", visible=False)
                    info1 = gr.Textbox(visible=False)
                    info2 = gr.Textbox(visible=False)

    # turn pages
    gallery_inputs = [img_path, page_index, image_index, tabname_box]
    gallery_outputs = [history_gallery, page_index, filenames, img_file_name, hidden, img_file_name]

    first_page.click(first_page_click, _js="images_history_turnpage", inputs=gallery_inputs, outputs=gallery_outputs)
    next_page.click(next_page_click, _js="images_history_turnpage", inputs=gallery_inputs, outputs=gallery_outputs)
    prev_page.click(prev_page_click, _js="images_history_turnpage", inputs=gallery_inputs, outputs=gallery_outputs)
    end_page.click(end_page_click, _js="images_history_turnpage", inputs=gallery_inputs, outputs=gallery_outputs)
    page_index.submit(page_index_change, _js="images_history_turnpage", inputs=gallery_inputs, outputs=gallery_outputs)
    renew_page.click(page_index_change, _js="images_history_turnpage", inputs=gallery_inputs, outputs=gallery_outputs)
    # page_index.change(page_index_change, inputs=[tabname_box, img_path,  page_index], outputs=[history_gallery, page_index])

    # other funcitons
    set_index.click(show_image_info, _js="images_history_get_current_img", inputs=[tabname_box, img_path, filenames], outputs=[img_file_name, image_index, hidden])
    img_file_name.change(fn=None, _js="images_history_enable_del_buttons", inputs=None, outputs=None)
    delete.click(delete_image, _js="images_history_delete", inputs=[delete_num, tabname_box, img_path, img_file_name, page_index, filenames, image_index], outputs=[filenames, delete_num])
    hidden.change(fn=run_pnginfo, inputs=[hidden], outputs=[info1, img_file_info, info2])

    # pnginfo.click(fn=run_pnginfo, inputs=[hidden], outputs=[info1, img_file_info, info2])
    switch_dict["fn"](pnginfo_send_to_txt2img, switch_dict["t2i"], img_file_info, 'switch_to_txt2img')
    switch_dict["fn"](pnginfo_send_to_img2img, switch_dict["i2i"], img_file_info, 'switch_to_img2img_img2img')


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
