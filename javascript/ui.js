// various functions for interation with ui.py not large enough to warrant putting them in separate files

function set_theme(theme){
    gradioURL = window.location.href
    if (!gradioURL.includes('?__theme=')) {
      window.location.replace(gradioURL + '?__theme=' + theme);
    }
}

function selected_gallery_index(){
    var buttons = gradioApp().querySelectorAll('[style="display: block;"].tabitem div[id$=_gallery] .gallery-item')
    var button = gradioApp().querySelector('[style="display: block;"].tabitem div[id$=_gallery] .gallery-item.\\!ring-2')

    var result = -1
    buttons.forEach(function(v, i){ if(v==button) { result = i } })

    return result
}

function extract_image_from_gallery(gallery){
    if(gallery.length == 1){
        return gallery[0]
    }

    index = selected_gallery_index()

    if (index < 0 || index >= gallery.length){
        return [null]
    }

    return gallery[index];
}

function args_to_array(args){
    res = []
    for(var i=0;i<args.length;i++){
        res.push(args[i])
    }
    return res
}

function switch_to_txt2img(){
    gradioApp().querySelector('#tabs').querySelectorAll('button')[0].click();

    return args_to_array(arguments);
}

function switch_to_img2img(){
    gradioApp().querySelector('#tabs').querySelectorAll('button')[1].click();
    gradioApp().getElementById('mode_img2img').querySelectorAll('button')[0].click();

    return args_to_array(arguments);
}

function switch_to_inpaint(){
    gradioApp().querySelector('#tabs').querySelectorAll('button')[1].click();
    gradioApp().getElementById('mode_img2img').querySelectorAll('button')[1].click();

    return args_to_array(arguments);
}

function switch_to_extras(){
    gradioApp().querySelector('#tabs').querySelectorAll('button')[2].click();

    return args_to_array(arguments);
}

function get_tab_index(tabId){
    var res = 0

    gradioApp().getElementById(tabId).querySelector('div').querySelectorAll('button').forEach(function(button, i){
        if(button.className.indexOf('bg-white') != -1)
            res = i
    })

    return res
}

function create_tab_index_args(tabId, args){
    var res = []
    for(var i=0; i<args.length; i++){
        res.push(args[i])
    }

    res[0] = get_tab_index(tabId)

    return res
}

function get_extras_tab_index(){
    const [,,...args] = [...arguments]
    return [get_tab_index('mode_extras'), get_tab_index('extras_resize_mode'), ...args]
}

function create_submit_args(args){
    res = []
    for(var i=0;i<args.length;i++){
        res.push(args[i])
    }

    // As it is currently, txt2img and img2img send back the previous output args (txt2img_gallery, generation_info, html_info) whenever you generate a new image.
    // This can lead to uploading a huge gallery of previously generated images, which leads to an unnecessary delay between submitting and beginning to generate.
    // I don't know why gradio is sending outputs along with inputs, but we can prevent sending the image gallery here, which seems to be an issue for some.
    // If gradio at some point stops sending outputs, this may break something
    if(Array.isArray(res[res.length - 3])){
        res[res.length - 3] = null
    }

    return res
}

function submit(){
    requestProgress('txt2img')

    return create_submit_args(arguments)
}

function submit_img2img(){
    requestProgress('img2img')

    res = create_submit_args(arguments)

    res[0] = get_tab_index('mode_img2img')

    return res
}


function ask_for_style_name(_, prompt_text, negative_prompt_text) {
    name_ = prompt('Style name:')
    return [name_, prompt_text, negative_prompt_text]
}

function confirm_clear_prompt(prompt, negative_prompt) {
    if(confirm("Delete prompt?")) {
        prompt = ""
        negative_prompt = ""
    }

    return [prompt, negative_prompt]
}



opts = {}
function apply_settings(jsdata){
    console.log(jsdata)

    opts = JSON.parse(jsdata)

    return jsdata
}

onUiUpdate(function(){
	if(Object.keys(opts).length != 0) return;

	json_elem = gradioApp().getElementById('settings_json')
	if(json_elem == null) return;

    textarea = json_elem.querySelector('textarea')
    jsdata = textarea.value
    opts = JSON.parse(jsdata)


    Object.defineProperty(textarea, 'value', {
        set: function(newValue) {
            var valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            var oldValue = valueProp.get.call(textarea);
            valueProp.set.call(textarea, newValue);

            if (oldValue != newValue) {
                opts = JSON.parse(textarea.value)
            }
        },
        get: function() {
            var valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            return valueProp.get.call(textarea);
        }
    });

    json_elem.parentElement.style.display="none"

	if (!txt2img_textarea) {
		txt2img_textarea = gradioApp().querySelector("#txt2img_prompt > label > textarea");
		txt2img_textarea?.addEventListener("input", () => update_token_counter("txt2img_token_button"));
	}
	if (!img2img_textarea) {
		img2img_textarea = gradioApp().querySelector("#img2img_prompt > label > textarea");
		img2img_textarea?.addEventListener("input", () => update_token_counter("img2img_token_button"));
	}
})

let txt2img_textarea, img2img_textarea = undefined;
let wait_time = 800
let token_timeout;

function update_txt2img_tokens(...args) {
	update_token_counter("txt2img_token_button")
	if (args.length == 2)
		return args[0]
	return args;
}

function update_img2img_tokens(...args) {
	update_token_counter("img2img_token_button")
	if (args.length == 2)
		return args[0]
	return args;
}

function update_token_counter(button_id) {
	if (token_timeout)
		clearTimeout(token_timeout);
	token_timeout = setTimeout(() => gradioApp().getElementById(button_id)?.click(), wait_time);
}

function restart_reload(){
    document.body.innerHTML='<h1 style="font-family:monospace;margin-top:20%;color:lightgray;text-align:center;">Reloading...</h1>';
    setTimeout(function(){location.reload()},2000)

    return []
}
