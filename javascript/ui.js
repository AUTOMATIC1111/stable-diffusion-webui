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
})

/**
 * Implement script-dependent UI restraints, e.g. forcing a specific sampling method
 */
let prev_ui_states = {};
function updateScriptRestraints() {
    const currentTab = get_uiCurrentTab()?.textContent.trim();
    const restraintsField = Array.from(gradioApp().querySelectorAll(`#${currentTab}_script_restraints_json textarea`))
        .filter(el => uiElementIsVisible(el.closest('.gr-form')))?.[0];
    
    if ( ! restraintsField ) {
        return;
    }

    if ( typeof prev_ui_states[currentTab] === 'undefined' ) {
        prev_ui_states[currentTab] = {};
    }

    window.requestAnimationFrame(() => {
        const restraints = JSON.parse(restraintsField.value);
        // const scriptSelect = gradioApp().querySelector(`#${currentTab}_scripts select`);
        const methodRadios = gradioApp().querySelectorAll(`[name="radio-${currentTab}_sampling"]`);

        if( restraints?.methods?.length ) {
            prev_ui_states[currentTab].sampling_method = gradioApp().querySelector(`[name="radio-${currentTab}_sampling"]:checked`);
            methodRadios.forEach(radio => {
                const isAllowed = restraints.methods.includes(radio.value);
                const label = radio.closest('label');
                radio.disabled = !isAllowed;
                radio.checked = isAllowed;
                label.classList[isAllowed ? 'remove' : 'add']('!cursor-not-allowed','disabled');
                label.title = !isAllowed ? `The selected script does not work with this method` : '';
            });
        } else {
            // reset to previously selected method
            methodRadios.forEach(radio => {
                const label = radio.closest('label');
                radio.disabled = false;
                radio.checked = radio === prev_ui_states[currentTab].sampling_method;
                label.classList.remove('!cursor-not-allowed','disabled');
                label.title = '';
            });
        }
    })
}