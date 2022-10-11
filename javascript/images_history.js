images_history_tab_list = ["txt2img", "img2img", "extras"]
function images_history_init(){ 
    if (gradioApp().getElementById('txt2img_images_history_first_page') == null) {
        setTimeout(images_history_init, 500)
    } else {        
        for (i in images_history_tab_list ){
            tab = images_history_tab_list[i]
            gradioApp().getElementById(tab + '_images_history').classList.add("images_history_gallery")
            gradioApp().getElementById(tab + '_images_history_set_index').classList.add("images_history_set_index")      
            
        }
        gradioApp().getElementById("txt2img_images_history_first_page").click()     
    }    
}
setTimeout(images_history_init, 500)
var images_history_button_actions = function(){
    if (!this.classList.contains("transform")){        
        gallery = this.parentElement
        while(!gallery.classList.contains("images_history_gallery")){gallery = gallery.parentElement}        
        buttons = gallery.querySelectorAll(".gallery-item")
        i = 0
        hidden_list = []
        buttons.forEach(function(e){
            if (e.style.display == "none"){
                hidden_list.push(i)
            }
            i += 1
        })
        if (hidden_list.length > 0){
            setTimeout(images_history_hide_buttons, 10, hidden_list, gallery)
        }
        
    }    
    images_history_set_image_info(this) 

}
onUiUpdate(function(){
    for (i in images_history_tab_list ){
        tab = images_history_tab_list[i]
        buttons = gradioApp().querySelectorAll('#' + tab + '_images_history .gallery-item')
        buttons.forEach(function(bnt){    
            bnt.addEventListener('click', images_history_button_actions, true)
         });
    }
})
function images_history_hide_buttons(hidden_list, gallery){
    buttons = gallery.querySelectorAll(".gallery-item")
    num = 0
    buttons.forEach(function(e){
        if (e.style.display == "none"){
            num += 1
        }
    })
    if (num == hidden_list.length){
        setTimeout(images_history_hide_buttons, 10, hidden_list, gallery)
    } 
    for( i in hidden_list){
        buttons[hidden_list[i]].style.display = "none"
    }    
}

function images_history_set_image_info(button){
    item = button.parentElement
    while(item.tagName != "DIV"){item = item.parentElement}    
    buttons = item.querySelectorAll(".gallery-item")
    index = -1
    i = 0
    buttons.forEach(function(e){
        if(e==button){index = i}
        if(e.style.display != "none"){
            i += 1
        }        
    })
    gallery = button.parentElement
    while(!gallery.classList.contains("images_history_gallery")){gallery = gallery.parentElement}    
    set_btn = gallery.querySelector(".images_history_set_index")
    set_btn.setAttribute("img_index", index)
    set_btn.click()
}

function images_history_get_current_img(tabname, image_path, files){
    s =  gradioApp().getElementById(tabname + '_images_history_set_index').getAttribute("img_index")
    return [s, image_path, files]
}

function images_history_delete(tabname, img_path, img_file_name, page_index, filenames, image_index){
    image_index = parseInt(image_index)
    tab = gradioApp().getElementById(tabname + '_images_history')
    set_btn = tab.querySelector(".images_history_set_index")
    buttons = []
    tab.querySelectorAll(".gallery-item").forEach(function(e){
        if (e.style.display != 'none'){
            buttons.push(e)
        }
    })

    
    img_num = buttons.length / 2
    if (img_num == 1){
            setTimeout(function(tabname){
                gradioApp().getElementById(tabname + '_images_history_renew_page').click()
            }, 30, tabname) 
    } else {
        buttons[image_index].style.display = 'none'
        buttons[image_index + img_num].style.display = 'none'
        if (image_index  >= img_num - 1){
            console.log(buttons.length, img_num)
            btn = buttons[img_num - 2]
        } else {
            btn = buttons[image_index + 1]           
        }    
        setTimeout(function(btn){btn.click()}, 30, btn)
    }  
    
    return [tabname, img_path, img_file_name, page_index, filenames, image_index]
}

function images_history_turnpage(img_path, page_index, image_index, tabname){
    buttons = gradioApp().getElementById(tabname + '_images_history').querySelectorAll(".gallery-item")
    buttons.forEach(function(elem) {
         elem.style.display = 'block'
    })
    return [img_path, page_index, image_index, tabname]
}
