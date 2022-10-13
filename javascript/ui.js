// various functions for interation with ui.py not large enough to warrant putting them in separate files

var major_tab;

function selected_gallery_index(){
    var buttons = gradioApp().querySelectorAll('[style="display: block;"].tabitem .gallery-item')
    var button = gradioApp().querySelector('[style="display: block;"].tabitem .gallery-item.\\!ring-2')

    var result = -1
    buttons.forEach(function(v, i){ if(v==button) { result = i } })

    return result
}

function extract_image_from_gallery(gallery){
    if(gallery.length === 1) return gallery[0]

    index = selected_gallery_index()

    if (index < 0 || index >= gallery.length){
        return [null]
    }

    return gallery[index];
}

function switch_to_txt2img(){
    major_tab[0].click();

    return Array.from(arguments);
}

function switch_to_img2img_img2img(){
    major_tab[1].click();
    gradioApp().getElementById('mode_img2img').querySelectorAll('button')[0].click();

    return Array.from(arguments);
}

function switch_to_img2img_inpaint(){
    major_tab[1].click();
    gradioApp().getElementById('mode_img2img').querySelectorAll('button')[1].click();

    return Array.from(arguments);
}

function switch_to_extras(){
    major_tab[2].click();

    return Array.from(arguments);
}

function extract_image_from_gallery_txt2img(gallery){
    switch_to_txt2img()
    return extract_image_from_gallery(gallery);
}

function extract_image_from_gallery_img2img(gallery){
    switch_to_img2img_img2img()
    return extract_image_from_gallery(gallery);
}

function extract_image_from_gallery_inpaint(gallery){
    switch_to_img2img_inpaint()
    return extract_image_from_gallery(gallery);
}

function extract_image_from_gallery_extras(gallery){
    switch_to_extras()
    return extract_image_from_gallery(gallery);
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
    return create_tab_index_args('mode_extras', arguments)
}

function create_submit_args(args) {
    args = Array.from(args);

    // As it is currently, txt2img and img2img send back the previous output args (txt2img_gallery, generation_info, html_info) whenever you generate a new image.
    // This can lead to uploading a huge gallery of previously generated images, which leads to an unnecessary delay between submitting and beginning to generate.
    // I don't know why gradio is seding outputs along with inputs, but we can prevent sending the image gallery here, which seems to be an issue for some.
    // If gradio at some point stops sending outputs, this may break something
    if(Array.isArray(args[args.length - 3])) {
        args[args.length - 3] = null
    }

    return args;
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
    name_ = prompt(I18N('Style name:'));
    return name_ === null ? [null, null, null]: [name_, prompt_text, negative_prompt_text]
}



opts = {}
function apply_settings(jsdata){
    console.log(jsdata)

    opts = JSON.parse(jsdata)

    return jsdata
}

onLoad(function(){
    major_tab = gradioApp().querySelectorAll('.tabs:nth-child(1) button');

    var json_elem = gradioApp().getElementById('settings_json');
    if(json_elem == null) return;
    json_elem.parentElement.remove();

    var textarea = json_elem.querySelector('textarea');
    opts = JSON.parse(textarea.value);

    var realVal = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
    Object.defineProperty(textarea, 'value', {
        set: function(v) {
            var oldValue = this.value;
            if (oldValue != v) {
                realVal.set.call(this, v);
                opts = JSON.parse(v);
            }
        },
        get: realVal.get
    });

    txt2img_textarea = gradioApp().querySelector("#txt2img_prompt > label > textarea");
    txt2img_textarea.addEventListener("input", () => update_token_counter("txt2img_token_button"));
    txt2img_textarea.addEventListener("keyup", (e) => submit_prompt(e, "txt2img_generate"));

    img2img_textarea = gradioApp().querySelector("#img2img_prompt > label > textarea");
    img2img_textarea.addEventListener("input", () => update_token_counter("img2img_token_button"));
    img2img_textarea.addEventListener("keyup", (e) => submit_prompt(e, "img2img_generate"));
})

let txt2img_textarea, img2img_textarea;
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

function submit_prompt(event, generate_button_id) {
    if (event.altKey && event.keyCode === 13) {
        event.preventDefault();
        gradioApp().getElementById(generate_button_id).click();
    }
}

function restart_reload(){
    document.body.innerHTML='<h1 style="font-family:monospace;margin-top:20%;color:lightgray;text-align:center;">'+I18N('Reloading...')+'</h1>';
    setTimeout(function(){location.reload()},2000)
}
