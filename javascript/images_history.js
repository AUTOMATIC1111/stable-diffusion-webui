var images_history_click_image = function(){
    if (!this.classList.contains("transform")){        
        var gallery = images_history_get_parent_by_class(this, "images_history_cantainor");
        var buttons = gallery.querySelectorAll(".gallery-item");
        var i = 0;
        var hidden_list = [];
        buttons.forEach(function(e){
            if (e.style.display == "none"){
                hidden_list.push(i);
            }
            i += 1;
        })
        if (hidden_list.length > 0){
            setTimeout(images_history_hide_buttons, 10, hidden_list, gallery);
        }        
    }    
    images_history_set_image_info(this); 
}

function images_history_disabled_del(){
    gradioApp().querySelectorAll(".images_history_del_button").forEach(function(btn){
        btn.setAttribute('disabled','disabled');
    }); 
}

function images_history_get_parent_by_class(item, class_name){
    var parent = item.parentElement;
    while(!parent.classList.contains(class_name)){
        parent = parent.parentElement;
    }
    return parent;  
}

function images_history_get_parent_by_tagname(item, tagname){
    var parent = item.parentElement;
    tagname = tagname.toUpperCase()
    while(parent.tagName != tagname){
        parent = parent.parentElement;
    }  
    return parent;
}

function images_history_hide_buttons(hidden_list, gallery){
    var buttons = gallery.querySelectorAll(".gallery-item");
    var num = 0;
    buttons.forEach(function(e){
        if (e.style.display == "none"){
            num += 1;
        }
    });
    if (num == hidden_list.length){
        setTimeout(images_history_hide_buttons, 10, hidden_list, gallery);
    } 
    for( i in hidden_list){
        buttons[hidden_list[i]].style.display = "none";
    }    
}

function images_history_set_image_info(button){
    var buttons = images_history_get_parent_by_tagname(button, "DIV").querySelectorAll(".gallery-item");
    var index = -1;
    var i = 0;
    buttons.forEach(function(e){
        if(e == button){
            index = i;
        }
        if(e.style.display != "none"){
            i += 1;
        }        
    });
    var gallery = images_history_get_parent_by_class(button, "images_history_cantainor");
    var set_btn = gallery.querySelector(".images_history_set_index");
    var curr_idx = set_btn.getAttribute("img_index", index);  
    if (curr_idx != index) {
        set_btn.setAttribute("img_index", index);        
        images_history_disabled_del();
    }
    set_btn.click();
    
}

function images_history_get_current_img(tabname, img_index, files){
    return [
        tabname, 
        gradioApp().getElementById(tabname + '_images_history_set_index').getAttribute("img_index"),         
        files
    ];
}

function images_history_delete(del_num, tabname, image_index){
    image_index = parseInt(image_index);
    var tab = gradioApp().getElementById(tabname + '_images_history');
    var set_btn = tab.querySelector(".images_history_set_index");
    var buttons = [];
    tab.querySelectorAll(".gallery-item").forEach(function(e){
        if (e.style.display != 'none'){
            buttons.push(e);
        }
    });    
    var img_num = buttons.length / 2;
    del_num = Math.min(img_num - image_index, del_num)    
    if (img_num <= del_num){
        setTimeout(function(tabname){
            gradioApp().getElementById(tabname + '_images_history_renew_page').click();
        }, 30, tabname); 
    } else {
        var next_img  
        for (var i = 0; i < del_num; i++){
            buttons[image_index + i].style.display = 'none';
            buttons[image_index + i + img_num].style.display = 'none';
            next_img = image_index + i + 1
        }
        var bnt;
        if (next_img  >= img_num){
            btn = buttons[image_index - 1];
        } else {            
            btn = buttons[next_img];          
        } 
        setTimeout(function(btn){btn.click()}, 30, btn);
    }
    images_history_disabled_del();  
   
}

function images_history_turnpage(tabname){
    gradioApp().getElementById(tabname + '_images_history_del_button').setAttribute('disabled','disabled');
    var buttons = gradioApp().getElementById(tabname + '_images_history').querySelectorAll(".gallery-item");
    buttons.forEach(function(elem) {
        elem.style.display = 'block';
    })   
}

function images_history_enable_del_buttons(){
    gradioApp().querySelectorAll(".images_history_del_button").forEach(function(btn){
        btn.removeAttribute('disabled');
    })
}

function images_history_init(){ 
    var tabnames = gradioApp().getElementById("images_history_tabnames_list")   
    if (tabnames){  
        images_history_tab_list = tabnames.querySelector("textarea").value.split(",")    
        for (var i in images_history_tab_list ){
            var tab = images_history_tab_list[i];
            gradioApp().getElementById(tab + '_images_history').classList.add("images_history_cantainor");
            gradioApp().getElementById(tab + '_images_history_set_index').classList.add("images_history_set_index");
            gradioApp().getElementById(tab + '_images_history_del_button').classList.add("images_history_del_button");
            gradioApp().getElementById(tab + '_images_history_gallery').classList.add("images_history_gallery");  
             gradioApp().getElementById(tab + "_images_history_start").setAttribute("style","padding:20px;font-size:25px");           
        }

        //preload
        if (gradioApp().getElementById("images_history_preload").querySelector("input").checked ){
            var tabs_box = gradioApp().getElementById("tab_images_history").querySelector("div").querySelector("div").querySelector("div");
            tabs_box.setAttribute("id", "images_history_tab");        
            var tab_btns = tabs_box.querySelectorAll("button"); 
            for (var i in images_history_tab_list){               
                var tabname = images_history_tab_list[i]
                tab_btns[i].setAttribute("tabname", tabname);
                tab_btns[i].addEventListener('click', function(){
                    var tabs_box = gradioApp().getElementById("images_history_tab");
                    if (!tabs_box.classList.contains(this.getAttribute("tabname"))) {
                        gradioApp().getElementById(this.getAttribute("tabname") + "_images_history_start").click();
                        tabs_box.classList.add(this.getAttribute("tabname"))
                    }                
                });
            }            
            tab_btns[0].click()
        }
    } else {
        setTimeout(images_history_init, 500);
    } 
}

var images_history_tab_list = "";
setTimeout(images_history_init, 500);
document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m){
        if (images_history_tab_list != ""){
            for (var i in images_history_tab_list ){
                let tabname = images_history_tab_list[i]
                var buttons = gradioApp().querySelectorAll('#' + tabname + '_images_history .gallery-item');
                buttons.forEach(function(bnt){    
                    bnt.addEventListener('click', images_history_click_image, true);
                });

                var cls_btn = gradioApp().getElementById(tabname + '_images_history_gallery').querySelector("svg");
                if (cls_btn){
                    cls_btn.addEventListener('click', function(){
                        gradioApp().getElementById(tabname + '_images_history_renew_page').click();
                    }, false);
                }

            }     
        }
    });
    mutationObserver.observe(gradioApp(), { childList:true, subtree:true });
});


