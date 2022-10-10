import os
def get_recent_images(is_img2img, dir_name, page_index, step):
	page_index = int(page_index)
	f_list = os.listdir(dir_name)
	file_list = []
	for file in f_list:
		if file[-4:] == ".txt":
			continue
		file_list.append(file)	
	file_list = sorted(file_list, key=lambda file: -os.path.getctime(os.path.join(dir_name, file)))
	num = 24
	max_page_index = len(file_list) // num + 1	
	page_index = max_page_index if page_index == -1 else page_index + step
	page_index = 1 if page_index < 1 else page_index	
	page_index = max_page_index if page_index > max_page_index else page_index
	idx_frm = (page_index - 1) * num
	file_list = file_list[idx_frm:idx_frm + num]
	print(f"Loading history page {page_index}")	
	return [os.path.join(dir_name, file) for file in file_list], page_index, file_list
def first_page_click(is_img2img, dir_name):
	return get_recent_images(is_img2img, dir_name, 1, 0)
def end_page_click(is_img2img, dir_name):
	return get_recent_images(is_img2img, dir_name, -1, 0)
def prev_page_click(is_img2img, dir_name, page_index):
	return get_recent_images(is_img2img, dir_name, page_index, -1)
def next_page_click(is_img2img, dir_name, page_index):	
	return get_recent_images(is_img2img, dir_name, page_index, 1)
def page_index_change(is_img2img, dir_name, page_index):	
	return get_recent_images(is_img2img, dir_name, page_index, 0)
def show_image_info(num, filenames):
	return filenames[int(num)]
def delete_image(is_img2img, dir_name, name, page_index, filenames):
	path = os.path.join(dir_name, name)		
	if os.path.exists(path):
		print(f"Delete file {path}")
		os.remove(path)	
	i = 0
	for f in filenames:
		if f == name:
			break
		i += 1
	images, page_index, file_list = get_recent_images(is_img2img, dir_name, page_index, 0)
	current_file =  file_list[i] if i < len(file_list) else None
	return images, page_index, file_list, current_file


def show_images_history(gr, opts, is_img2img):
		def id_name(is_img2img, name):	
			return ("img2img" if is_img2img else "txt2img") + "_" + name
		with gr.Row():				
			if is_img2img:
				dir_name = opts.outdir_img2img_samples
			else:
				dir_name = opts.outdir_txt2img_samples							
			first_page = gr.Button('First Page', elem_id=id_name(is_img2img,"images_history_first_page"))
			prev_page = gr.Button('Prev Page') 
			page_index = gr.Number(value=1)
			next_page = gr.Button('Next Page') 
			end_page = gr.Button('End Page')  
		with gr.Row():	      
			delete = gr.Button('Delete')
			Send = gr.Button('Send')
		with gr.Row():	
			with gr.Column(elem_id=id_name(is_img2img,"images_history")):	
				history_gallery = gr.Gallery(label="Images history").style(grid=6)
				img_file_name = gr.Textbox()	
				img_file_info = gr.Textbox(dir_name)
				img_path = gr.Textbox(dir_name, visible=False)					
				set_index = gr.Button('set_index',  elem_id=id_name(is_img2img,"images_history_set_index"))
				is_img2img_flag = gr.Checkbox(is_img2img, visible=False)
		filenames = gr.State()
		first_page.click(first_page_click, inputs=[is_img2img_flag, img_path], outputs=[history_gallery, page_index, filenames])
		next_page.click(next_page_click, inputs=[is_img2img_flag, img_path, page_index], outputs=[history_gallery, page_index, filenames])
		prev_page.click(prev_page_click, inputs=[is_img2img_flag, img_path,  page_index], outputs=[history_gallery, page_index, filenames])		
		end_page.click(end_page_click, inputs=[is_img2img_flag, img_path], outputs=[history_gallery, page_index, filenames])
		page_index.submit(page_index_change, inputs=[is_img2img_flag, img_path,  page_index], outputs=[history_gallery, page_index, filenames])
		set_index.click(show_image_info, _js="images_history_get_current_img",inputs=[is_img2img_flag, filenames], outputs=img_file_name)	
		delete.click(delete_image, inputs=[is_img2img_flag, img_path, img_file_name, page_index, filenames], outputs=[history_gallery, page_index, filenames,img_file_name]) 
		#page_index.change(page_index_change, inputs=[is_img2img_flag, img_path,  page_index], outputs=[history_gallery, page_index])
		
def create_history_tabs(gr, opts):
	with gr.Blocks(analytics_enabled=False) as images_history:
		with gr.Tabs() as tabs:			
			with gr.Tab("txt2img history", id="images_history_txt2img"):
				with gr.Blocks(analytics_enabled=False) as images_history_txt2img:   
					show_images_history(gr, opts, is_img2img=False) 
			with gr.Tab("img2img history", id="images_history_img2img"):
				with gr.Blocks(analytics_enabled=False) as images_history_img2img:
					show_images_history(gr, opts, is_img2img=True)
	return images_history
