import os
def get_recent_images(dir_name, page_index, step, image_index):
	print(image_index)
	page_index = int(page_index)
	f_list = os.listdir(dir_name)
	file_list = []
	for file in f_list:
		if file[-4:] == ".txt":
			continue
		file_list.append(file)	
	file_list = sorted(file_list, key=lambda file: -os.path.getctime(os.path.join(dir_name, file)))
	num = 48
	max_page_index = len(file_list) // num + 1	
	page_index = max_page_index if page_index == -1 else page_index + step
	page_index = 1 if page_index < 1 else page_index	
	page_index = max_page_index if page_index > max_page_index else page_index
	idx_frm = (page_index - 1) * num
	file_list = file_list[idx_frm:idx_frm + num]
	print(f"Loading history page {page_index}")	
	image_index = int(image_index)
	if image_index < 0 or image_index > len(file_list)  - 1:
		current_file = None 
		hide_image = None
	else:
		current_file =  file_list[int(image_index)]
		hide_image = os.path.join(dir_name, current_file)
	return [os.path.join(dir_name, file) for file in file_list], page_index, file_list, current_file, hide_image
def first_page_click(dir_name, page_index, image_index):
	return get_recent_images(dir_name, 1, 0, image_index)
def end_page_click(dir_name, page_index, image_index):
	return get_recent_images(dir_name, -1, 0, image_index)
def prev_page_click(dir_name, page_index, image_index):
	return get_recent_images(dir_name, page_index, -1, image_index)
def next_page_click(dir_name, page_index, image_index):	
	return get_recent_images(dir_name, page_index, 1, image_index)
def page_index_change(dir_name, page_index, image_index):	
	return get_recent_images(dir_name, page_index, 0, image_index)

def show_image_info(num, image_path, filenames):
	file = filenames[int(num)]
	return file, num, os.path.join(image_path, file)
def delete_image(is_img2img, dir_name, name, page_index, filenames, image_index):
	print("filename", name)
	path = os.path.join(dir_name, name)		
	if os.path.exists(path):
		print(f"Delete file {path}")
		os.remove(path)	
	images, page_index, file_list, current_file, hide_image = get_recent_images(dir_name, page_index, 0, image_index)
	return images, page_index, file_list, current_file, hide_image


def show_images_history(gr, opts, is_img2img, run_pnginfo, switch_dict):
		def id_name(is_img2img, name):	
			return ("img2img" if is_img2img else "txt2img") + "_" + name
		if is_img2img:
			dir_name = opts.outdir_img2img_samples
		else:
			dir_name = opts.outdir_txt2img_samples
		with gr.Row():	 
				first_page = gr.Button('First', elem_id=id_name(is_img2img,"images_history_first_page"))
				prev_page = gr.Button('Prev') 
				page_index = gr.Number(value=1, label="Page Index")
				next_page = gr.Button('Next') 
				end_page = gr.Button('End')	
		with gr.Row(elem_id=id_name(is_img2img,"images_history")):			
			with gr.Row():	 
				with gr.Column():
					history_gallery = gr.Gallery(show_label=False).style(grid=6)
				with gr.Column():	 
					with gr.Row():	      
						delete = gr.Button('Delete')
						pnginfo_send_to_txt2img = gr.Button('Send to txt2img')
						pnginfo_send_to_img2img = gr.Button('Send to img2img')
					with gr.Row():
						with gr.Column():
							img_file_info = gr.Textbox(dir_name, label="Generate Info")
							img_file_name = gr.Textbox(label="File Name")	
					with gr.Row():	
							# hiden items
							img_path = gr.Textbox(dir_name, visible=False)				
							is_img2img_flag = gr.Checkbox(is_img2img, visible=False)	
							image_index = gr.Textbox(value=-1, visible=False)						
							set_index = gr.Button('set_index',  elem_id=id_name(is_img2img,"images_history_set_index"))
							filenames = gr.State()
							hide_image = gr.Image(visible=False, type="pil")
							info1 = gr.Textbox(visible=False)
							info2 = gr.Textbox(visible=False)

				
		# turn pages
		gallery_inputs = [img_path, page_index, image_index]
		gallery_outputs = [history_gallery, page_index, filenames, img_file_name, hide_image]
		first_page.click(first_page_click, inputs=gallery_inputs, outputs=gallery_outputs)
		next_page.click(next_page_click, inputs=gallery_inputs, outputs=gallery_outputs)
		prev_page.click(prev_page_click, inputs=gallery_inputs, outputs=gallery_outputs)		
		end_page.click(end_page_click, inputs=gallery_inputs, outputs=gallery_outputs)
		page_index.submit(page_index_change, inputs=gallery_inputs, outputs=gallery_outputs)
		#page_index.change(page_index_change, inputs=[is_img2img_flag, img_path,  page_index], outputs=[history_gallery, page_index])

		#other funcitons
		set_index.click(show_image_info, _js="images_history_get_current_img", inputs=[is_img2img_flag,  img_path, filenames], outputs=[img_file_name, image_index, hide_image])
		delete.click(delete_image, inputs=[is_img2img_flag, img_path, img_file_name, page_index, filenames, image_index], outputs=gallery_outputs) 
		hide_image.change(fn=run_pnginfo, inputs=[hide_image], outputs=[info1, img_file_info, info2])
		switch_dict["fn"](pnginfo_send_to_txt2img, switch_dict["t2i"], img_file_info, 'switch_to_txt2img')
		switch_dict["fn"](pnginfo_send_to_img2img, switch_dict["i2i"], img_file_info, 'switch_to_img2img_img2img')
	
		
def create_history_tabs(gr, opts, run_pnginfo, switch_dict):
	with gr.Blocks(analytics_enabled=False) as images_history:
		with gr.Tabs() as tabs:			
			with gr.Tab("txt2img history", id="images_history_txt2img"):
				with gr.Blocks(analytics_enabled=False) as images_history_txt2img:   
					show_images_history(gr, opts, False, run_pnginfo, switch_dict) 
			with gr.Tab("img2img history", id="images_history_img2img"):
				with gr.Blocks(analytics_enabled=False) as images_history_img2img:
					show_images_history(gr, opts, True, run_pnginfo, switch_dict)
	return images_history
