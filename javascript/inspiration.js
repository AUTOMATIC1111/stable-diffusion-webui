function public_image_index_in_gallery(item, gallery){
    var index;
    var i = 0;
    gallery.querySelectorAll("img").forEach(function(e){
        if (e == item)
            index = i;
        i += 1;
    });
    return index;
}

function inspiration_selected(name, types, name_list){
    var btn = gradioApp().getElementById("inspiration_select_button")
    return [gradioApp().getElementById("inspiration_select_button").getAttribute("img-index"), types];
}    
var inspiration_image_click = function(){
    var index =  public_image_index_in_gallery(this, gradioApp().getElementById("inspiration_gallery"));
    var btn = gradioApp().getElementById("inspiration_select_button")
    btn.setAttribute("img-index", index)
    setTimeout(function(btn){btn.click();}, 10, btn)
}

document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m){
        var gallery = gradioApp().getElementById("inspiration_gallery")
        if (gallery) {
            var node = gallery.querySelector(".absolute.backdrop-blur.h-full")
            if (node) {
                node.style.display = "None"; //parentNode.removeChild(node) 
            }
            
            gallery.querySelectorAll('img').forEach(function(e){    
                e.onclick = inspiration_image_click
        }) 

        }


    });
    mutationObserver.observe( gradioApp(), { childList:true, subtree:true });

});
