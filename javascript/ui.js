// various functions for interation with ui.py not large enough to warrant putting them in separate files

function selected_gallery_index(){
    var buttons = gradioApp().querySelectorAll('[style="display: block;"].tabitem .gallery-item')
    var button = gradioApp().querySelector('[style="display: block;"].tabitem .gallery-item.\\!ring-2')

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
    gradioApp().querySelectorAll('button')[0].click();

    return args_to_array(arguments);
}

function switch_to_img2img_img2img(){
    gradioApp().querySelectorAll('button')[1].click();
    gradioApp().getElementById('mode_img2img').querySelectorAll('button')[0].click();

    return args_to_array(arguments);
}

function switch_to_img2img_inpaint(){
    gradioApp().querySelectorAll('button')[1].click();
    gradioApp().getElementById('mode_img2img').querySelectorAll('button')[1].click();

    return args_to_array(arguments);
}

function switch_to_extras(){
    gradioApp().querySelectorAll('button')[2].click();

    return args_to_array(arguments);
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

function create_submit_args(args){
    res = []
    for(var i=0;i<args.length;i++){
        res.push(args[i])
    }

    // As it is currently, txt2img and img2img send back the previous output args (txt2img_gallery, generation_info, html_info) whenever you generate a new image.
    // This can lead to uploading a huge gallery of previously generated images, which leads to an unnecessary delay between submitting and beginning to generate.
    // I don't know why gradio is seding outputs along with inputs, but we can prevent sending the image gallery here, which seems to be an issue for some.
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
    return name_ === null ? [null, null, null]: [name_, prompt_text, negative_prompt_text]
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

	let debounce_time = 800
	if (!txt2img_textarea) {
		txt2img_textarea = gradioApp().querySelector("#txt2img_prompt > label > textarea")
		txt2img_textarea?.addEventListener("input", debounce(submit_prompt_text.bind(null, "txt2img"), debounce_time))
	}
	if (!img2img_textarea) {
		img2img_textarea = gradioApp().querySelector("#img2img_prompt > label > textarea")
		img2img_textarea?.addEventListener("input", debounce(submit_prompt_text.bind(null, "img2img"), debounce_time))
    }
})


let txt2img_textarea, img2img_textarea = undefined;
function submit_prompt_text(source, e) {
	let prompt_text;
	if (source == "txt2img")
		prompt_text = txt2img_textarea.value;
	else if (source == "img2img")
		prompt_text = img2img_textarea.value;
	if (!prompt_text)
		return;
	params = {
		method: "POST",
		headers: {
			"Accept": "application/json",
			"Content-type": "application/json"
		},
		body: JSON.stringify({data:[prompt_text]})
	}
	fetch('http://127.0.0.1:7860/api/tokenize/', params)
	.then((response) => response.json())
	.then((data) => {
		if (data?.data.length) {
			let response_json = data.data[0]
			if (elem = gradioApp().getElementById(source+"_token_counter")) {
				if (response_json.token_count > response_json.max_length)
					elem.classList.add("red");
				else
					elem.classList.remove("red");
				elem.innerText = response_json.token_count + "/" + response_json.max_length;
			}
		}
	})
	.catch((error) => {
		console.error('Error:', error);
	});
}