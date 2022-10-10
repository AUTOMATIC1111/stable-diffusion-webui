function init_images_history(){ 
	if (gradioApp().getElementById('txt2img_images_history_first_page') == null) {
		setTimeout(init_images_history, 1000)
	} else {
		tab_list = ["txt2img", "img2img"]
		for (i in tab_list){
			tab = tab_list[i]
			gradioApp().getElementById(tab + "_images_history_first_page").click()
			$(gradioApp().getElementById(tab + '_images_history')).addClass("images_history_gallery")
			item = $(gradioApp().getElementById(tab + '_images_history_set_index'))
			item.addClass("images_history_set_index")
			item.hide()
		}		
	}
	
}
setTimeout(init_images_history, 1000)
onUiUpdate(function(){
    fullImg_preview = gradioApp().querySelectorAll('#txt2img_images_history img.w-full')
        if(fullImg_preview.length > 0){
	    	fullImg_preview.forEach(set_history_index_from_img);
    }
    fullImg_preview = gradioApp().querySelectorAll('#img2img_images_history img.w-full')
        if(fullImg_preview.length > 0){
	    	fullImg_preview.forEach(set_history_index_from_img);
    }
})

function set_history_gallery_index(item){
	buttons = item.find(".gallery-item")
	// alert(item.attr("id") + " " + buttons.length)
	index = -1
	i = 0
	buttons.each(function(){
		if($(this).hasClass("!ring-2")){ index = i }
		i += 1
	})
	if (index == -1){
		setTimeout(set_history_gallery_index, 10, item)     
	} else {
		item = item.find(".images_history_set_index").first()
    	item.attr("img_index", index)
    	item.click()
	}
}
function set_history_index_from_img(e){
    if(e && e.parentElement.tagName == 'BUTTON'){
    	bnt = $(e).parent()
    	if (bnt.hasClass("transform")){
    		bnt.off("click").on("click",function(){
    		set_history_gallery_index($(this).parents(".images_history_gallery").first())  
     		})
    	} else {
    		bnt.off("mousedown").on("mousedown", function(){
    		set_history_gallery_index($(this).parents(".images_history_gallery").first())  
    		})

		}  		
    }
}
function images_history_get_current_img(is_image2image, image_path, files){
	head = is_image2image?"img2img":"txt2img"
	s =  $(gradioApp().getElementById(head + '_images_history_set_index')).attr("img_index")
	return [s, image_path, files]
}

