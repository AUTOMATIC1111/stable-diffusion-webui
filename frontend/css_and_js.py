def css(opt):
    css_hide_progressbar = """
    .wrap .m-12 svg { display:none!important; }
    .wrap .m-12::before { content:"Loading..." }
    .progress-bar { display:none!important; }
    .meta-text { display:none!important; }
    """
    styling = """
    
    [data-testid="image"] {min-height: 512px !important}
    * #body>.col:nth-child(2){width:250%;max-width:89vw}

    #prompt_input, #img2img_prompt_input { 
        padding: 0px;
        border: none;
    }

    #prompt_row input,
    #prompt_row textarea {
        font-size: 1.2rem;
        line-height: 1.6rem;
    }

    #edit_mode_select{width:auto !important}

    input[type=number]:disabled { -moz-appearance: textfield;+ }

    #generate, #img2img_mask_btn, #img2img_edit_btn {
        align-self: stretch;
    }
    """
    return styling if opt.no_progressbar_hiding else styling + css_hide_progressbar

# This is the code that finds which selected item the user has in the gallery
js_part_getindex_txt2img="""
const root = document.querySelector('gradio-app').shadowRoot;
const getIndex = function(){
const selected = root.querySelector('#txt2img_gallery_output .\\\\!ring-2');
return selected ? [...selected.parentNode.children].indexOf(selected) : 0;
};"""
js_part_getindex_img2img="""
const root = document.querySelector('gradio-app').shadowRoot;
const getIndex = function(){
    const selected = root.querySelector('#img2img_gallery_output .\\\\!ring-2');
    return selected ? [...selected.parentNode.children].indexOf(selected) : 0;
};"""
js_part_clear_img2img="""
root.querySelector('#img2img_editor .modify-upload button:last-child')?.click();


"""
js_return_selected_txt2img = "(x) => {" + js_part_getindex_txt2img + js_part_clear_img2img + """
return [x[getIndex()].replace('data:;','data:image/png;')];
}"""
js_return_selected_img2img = "(x) => {" + js_part_getindex_img2img + js_part_clear_img2img + """
return [x[getIndex()].replace('data:;','data:image/png;')];
}"""

js_part_copy_to_clipboard="""
const data = x[getIndex()];
const blob = await (await fetch(data.replace('data:;','data:image/png;'))).blob(); 
const item = new ClipboardItem({'image/png': blob});
navigator.clipboard.write([item]);
return x;
}"""
js_copy_selected_txt2img = "async (x) => {" + js_part_getindex_txt2img + js_part_copy_to_clipboard
js_copy_selected_img2img = "async (x) => {" + js_part_getindex_img2img + js_part_copy_to_clipboard
