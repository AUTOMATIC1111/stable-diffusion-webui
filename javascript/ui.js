// various functions for interaction with ui.py not large enough to warrant putting them in separate files

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
        return [gallery[0]]
    }

    index = selected_gallery_index()

    if (index < 0 || index >= gallery.length){
        return [null]
    }

    return [gallery[index]];
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

function switch_to_img2img_tab(no){
    gradioApp().querySelector('#tabs').querySelectorAll('button')[1].click();
    gradioApp().getElementById('mode_img2img').querySelectorAll('button')[no].click();
}
function switch_to_img2img(){
    switch_to_img2img_tab(0);
    return args_to_array(arguments);
}

function switch_to_sketch(){
    switch_to_img2img_tab(1);
    return args_to_array(arguments);
}

function switch_to_inpaint(){
    switch_to_img2img_tab(2);
    return args_to_array(arguments);
}

function switch_to_inpaint_sketch(){
    switch_to_img2img_tab(3);
    return args_to_array(arguments);
}

function switch_to_inpaint(){
    gradioApp().querySelector('#tabs').querySelectorAll('button')[1].click();
    gradioApp().getElementById('mode_img2img').querySelectorAll('button')[2].click();

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

function get_img2img_tab_index() {
    let res = args_to_array(arguments)
    res.splice(-2)
    res[0] = get_tab_index('mode_img2img')
    return res
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

function showSubmitButtons(tabname, show){
    gradioApp().getElementById(tabname+'_interrupt').style.display = show ? "none" : "block"
    gradioApp().getElementById(tabname+'_skip').style.display = show ? "none" : "block"
}

function submit(){
    rememberGallerySelection('txt2img_gallery')
    showSubmitButtons('txt2img', false)

    var id = randomId()
    requestProgress(id, gradioApp().getElementById('txt2img_gallery_container'), gradioApp().getElementById('txt2img_gallery'), function(){
        showSubmitButtons('txt2img', true)

    })

    var res = create_submit_args(arguments)

    res[0] = id

    return res
}

function submit_img2img(){
    rememberGallerySelection('img2img_gallery')
    showSubmitButtons('img2img', false)

    var id = randomId()
    requestProgress(id, gradioApp().getElementById('img2img_gallery_container'), gradioApp().getElementById('img2img_gallery'), function(){
        showSubmitButtons('img2img', true)
    })

    var res = create_submit_args(arguments)

    res[0] = id
    res[1] = get_tab_index('mode_img2img')

    return res
}

function modelmerger(){
    var id = randomId()
    requestProgress(id, gradioApp().getElementById('modelmerger_results_panel'), null, function(){})

    var res = create_submit_args(arguments)
    res[0] = id
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


promptTokecountUpdateFuncs = {}

function recalculatePromptTokens(name){
    if(promptTokecountUpdateFuncs[name]){
        promptTokecountUpdateFuncs[name]()
    }
}

function recalculate_prompts_txt2img(){
    recalculatePromptTokens('txt2img_prompt')
    recalculatePromptTokens('txt2img_neg_prompt')
    return args_to_array(arguments);
}

function recalculate_prompts_img2img(){
    recalculatePromptTokens('img2img_prompt')
    recalculatePromptTokens('img2img_neg_prompt')
    return args_to_array(arguments);
}


opts = {}
onUiUpdate(function(){
	if(Object.keys(opts).length != 0) return;

	json_elem = gradioApp().getElementById('settings_json')
	if(json_elem == null) return;

    var textarea = json_elem.querySelector('textarea')
    var jsdata = textarea.value
    opts = JSON.parse(jsdata)
    executeCallbacks(optionsChangedCallbacks);

    Object.defineProperty(textarea, 'value', {
        set: function(newValue) {
            var valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            var oldValue = valueProp.get.call(textarea);
            valueProp.set.call(textarea, newValue);

            if (oldValue != newValue) {
                opts = JSON.parse(textarea.value)
            }

            executeCallbacks(optionsChangedCallbacks);
        },
        get: function() {
            var valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            return valueProp.get.call(textarea);
        }
    });

    json_elem.parentElement.style.display="none"

    function registerTextarea(id, id_counter, id_button){
        var prompt = gradioApp().getElementById(id)
        var counter = gradioApp().getElementById(id_counter)
        var textarea = gradioApp().querySelector("#" + id + " > label > textarea");

        if(counter.parentElement == prompt.parentElement){
            return
        }

        prompt.parentElement.insertBefore(counter, prompt)
        counter.classList.add("token-counter")
        prompt.parentElement.style.position = "relative"

		promptTokecountUpdateFuncs[id] = function(){ update_token_counter(id_button); }
		textarea.addEventListener("input", promptTokecountUpdateFuncs[id]);
    }

    registerTextarea('txt2img_prompt', 'txt2img_token_counter', 'txt2img_token_button')
    registerTextarea('txt2img_neg_prompt', 'txt2img_negative_token_counter', 'txt2img_negative_token_button')
    registerTextarea('img2img_prompt', 'img2img_token_counter', 'img2img_token_button')
    registerTextarea('img2img_neg_prompt', 'img2img_negative_token_counter', 'img2img_negative_token_button')

    show_all_pages = gradioApp().getElementById('settings_show_all_pages')
    settings_tabs = gradioApp().querySelector('#settings div')
    if(show_all_pages && settings_tabs){
        settings_tabs.appendChild(show_all_pages)
        show_all_pages.onclick = function(){
            gradioApp().querySelectorAll('#settings > div').forEach(function(elem){
                elem.style.display = "block";
            })
        }
    }
	
	
	
	/* 	
	^ matches the start
	* matches any position
	$ matches the end
	*/
	
	/* auto grow textarea */
	gradioApp().querySelectorAll('[id $= "_prompt"] textarea').forEach(function (elem) {
		elem.style.boxSizing = 'border-box';
		var offset = elem.offsetHeight - elem.clientHeight;
		elem.addEventListener('input', function (e) {
			e.target.style.minHeight = 'auto';
			e.target.style.minHeight = e.target.scrollHeight + offset + 'px';
		});
		
		elem.addEventListener('focus', function (e) {
			e.target.style.minHeight = 'auto';
			e.target.style.minHeight = e.target.scrollHeight + offset + 'px';
		});
	});
	
	
	/* resizable split view */			
	
	const resizeEvent = window.document.createEvent('UIEvents'); 
	resizeEvent.initUIEvent('resize', true, false, window, 0); 	

	gradioApp().querySelectorAll('[id $="2img_splitter"]').forEach((elem) => {
		
		elem.addEventListener("mousedown", function(e) {	

			e.preventDefault();
			
			let resizer = e.currentTarget;
			let container = resizer.parentElement;
			
			let flexDir = window.getComputedStyle(container).getPropertyValue('flex-direction');
		
			let leftSide = resizer.previousElementSibling;
			let rightSide = resizer.nextElementSibling;
			let dir = 1.0;
			
			if(flexDir == "row-reverse"){
				dir = -1.0;				
			}

			let x = e.clientX;
			let y = e.clientY;
			let leftWidth = leftSide.getBoundingClientRect().width;		
			

			
			function mouseMoveHandler(e) {		
				resizer.style.cursor = 'col-resize';
				container.style.cursor = 'col-resize';

				const dx = (e.clientX - x)*dir;
				const dy = (e.clientY - y)*dir;

				const newLeftWidth = ((leftWidth + dx) * 100) / container.getBoundingClientRect().width;
				leftSide.style.flexBasis  = `${newLeftWidth}%`;
				leftSide.style.userSelect = 'none';
				leftSide.style.pointerEvents = 'none';
				rightSide.style.userSelect = 'none';
				rightSide.style.pointerEvents = 'none';
				window.dispatchEvent(resizeEvent);		
			}

			function mouseUpHandler() {
				resizer.style.removeProperty('cursor');
				container.style.removeProperty('cursor');
				leftSide.style.removeProperty('user-select');
				leftSide.style.removeProperty('pointer-events');
				rightSide.style.removeProperty('user-select');
				rightSide.style.removeProperty('pointer-events');
				container.removeEventListener('mousemove', mouseMoveHandler);
				container.removeEventListener('mouseup', mouseUpHandler);
				window.dispatchEvent(resizeEvent);				
			}
			
			container.addEventListener('mousemove', mouseMoveHandler);
			container.addEventListener('mouseup', mouseUpHandler);		
	
		})
		
		let flex_reverse = false;
		elem.addEventListener("dblclick", function(e) {	
			flex_reverse = !flex_reverse;	
			e.preventDefault();
			
			let resizer = e.currentTarget;
			let container = resizer.parentElement;			
			//let flexDir = window.getComputedStyle(container).getPropertyValue('flex-direction');

			if(flex_reverse){
				container.style.flexDirection = 'row-reverse';			
			}else{
				container.style.flexDirection = 'row';	
			}
		
		})
		
		

	})
	
	// mobile nav menu
	const tabs_menu = gradioApp().querySelector('#tabs > div:first-child');
	const nav_menu = gradioApp().querySelector('#nav_menu');
	const gcontainer = gradioApp().querySelector('.mx-auto.container');
	
	let menu_open = false;
	function toggleNavMenu(e) {
		menu_open = !menu_open;	
		e.preventDefault();
        e.stopPropagation();
		
		if(menu_open){
			tabs_menu.classList.add("open");
			nav_menu.classList.add("fixed");
			gcontainer.addEventListener('click', toggleNavMenu);
		

		}else{
			tabs_menu.classList.remove("open");
			nav_menu.classList.remove("fixed");
			gcontainer.removeEventListener('click', toggleNavMenu);
			
		}
		
		
	}
	

    nav_menu.addEventListener('click', toggleNavMenu);
	
	//const doc = document.getElementsByTagName('gradio-app')[0].shadowRoot;
	


	
})

onOptionsChanged(function(){
    elem = gradioApp().getElementById('sd_checkpoint_hash')
    sd_checkpoint_hash = opts.sd_checkpoint_hash || ""
    shorthash = sd_checkpoint_hash.substr(0,10)

	if(elem && elem.textContent != shorthash){
	    elem.textContent = shorthash
	    elem.title = sd_checkpoint_hash
	    elem.href = "https://google.com/search?q=" + sd_checkpoint_hash
	}
})

let txt2img_textarea, img2img_textarea = undefined;
let wait_time = 800
let token_timeouts = {};

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
	if (token_timeouts[button_id])
		clearTimeout(token_timeouts[button_id]);
	token_timeouts[button_id] = setTimeout(() => gradioApp().getElementById(button_id)?.click(), wait_time);
}

function restart_reload(){
	document.body.style.backgroundColor = "#1a1a1a";
    document.body.innerHTML='<h1 style="font-family:monospace;margin-top:20%;color:lightgray;text-align:center;">Reloading...</h1>';	
    setTimeout(function(){location.reload()},2000)

    return []
}

// Simulate an `input` DOM event for Gradio Textbox component. Needed after you edit its contents in javascript, otherwise your edits
// will only visible on web page and not sent to python.
function updateInput(target){
	let e = new Event("input", { bubbles: true })
	Object.defineProperty(e, "target", {value: target})
	target.dispatchEvent(e);
}


var desiredCheckpointName = null;
function selectCheckpoint(name){
    desiredCheckpointName = name;
    gradioApp().getElementById('change_checkpoint').click()
}


