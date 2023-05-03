// various functions for interaction with ui.py not large enough to warrant putting them in separate files

function set_theme(theme){
    gradioURL = window.location.href
    if (!gradioURL.includes('?__theme=')) window.location.replace(gradioURL + '?__theme=' + theme);
}

function all_gallery_buttons() {
    var allGalleryButtons = gradioApp().querySelectorAll('[style="display: block;"].tabitem div[id$=_gallery].gradio-gallery .thumbnails > .thumbnail-item.thumbnail-small');
    var visibleGalleryButtons = [];
    allGalleryButtons.forEach(function(elem) {
        if (elem.parentElement.offsetParent) visibleGalleryButtons.push(elem);
    })
    return visibleGalleryButtons;
}

function selected_gallery_button() {
    var allCurrentButtons = gradioApp().querySelectorAll('[style="display: block;"].tabitem div[id$=_gallery].gradio-gallery .thumbnail-item.thumbnail-small.selected');
    var visibleCurrentButton = null;
    allCurrentButtons.forEach(function(elem) {
        if (elem.parentElement.offsetParent) visibleCurrentButton = elem;
    })
    return visibleCurrentButton;
}

function selected_gallery_index(){
    var buttons = all_gallery_buttons();
    var button = selected_gallery_button();
    var result = -1
    buttons.forEach(function(v, i){ if(v==button) { result = i } })
    return result
}

function extract_image_from_gallery(gallery){
    if (gallery.length == 0) return [null];
    if (gallery.length == 1) return [gallery[0]];
    index = selected_gallery_index()
    if (index < 0 || index >= gallery.length) index = 0;
    return [gallery[index]];
}

function args_to_array(args){
    res = []
    for(var i=0;i<args.length;i++) res.push(args[i])
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
        if(button.className.indexOf('selected') != -1) res = i
    })
    return res
}

function create_tab_index_args(tabId, args){
    var res = []
    for(var i=0; i<args.length; i++) res.push(args[i])
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
    for(var i=0;i<args.length;i++) res.push(args[i])
    // As it is currently, txt2img and img2img send back the previous output args (txt2img_gallery, generation_info, html_info) whenever you generate a new image.
    // This can lead to uploading a huge gallery of previously generated images, which leads to an unnecessary delay between submitting and beginning to generate.
    // I don't know why gradio is sending outputs along with inputs, but we can prevent sending the image gallery here, which seems to be an issue for some.
    // If gradio at some point stops sending outputs, this may break something
    if(Array.isArray(res[res.length - 3])) res[res.length - 3] = null
    return res
}

function showSubmitButtons(tabname, show){
    gradioApp().getElementById(tabname+'_interrupt').style.display = show ? "none" : "block"
    gradioApp().getElementById(tabname+'_skip').style.display = show ? "none" : "block"
    // gradioApp().getElementById(tabname+'_interrupt').style.display = "block"
    // gradioApp().getElementById(tabname+'_skip').style.display = "block"
}

function submit(){
    rememberGallerySelection('txt2img_gallery')
    showSubmitButtons('txt2img', false)
    const id = randomId()
    const atEnd = () => showSubmitButtons('txt2img', true)
    requestProgress(id, gradioApp().getElementById('txt2img_gallery_container'), gradioApp().getElementById('txt2img_gallery'), atEnd)
    var res = create_submit_args(arguments)
    res[0] = id
    return res
}

function submit_img2img(){
    rememberGallerySelection('img2img_gallery')
    showSubmitButtons('img2img', false)
    const id = randomId()
    const atEnd = () => showSubmitButtons('img2img', true)
    requestProgress(id, gradioApp().getElementById('img2img_gallery_container'), gradioApp().getElementById('img2img_gallery'), atEnd)
    var res = create_submit_args(arguments)
    res[0] = id
    res[1] = get_tab_index('mode_img2img')
    return res
}

function modelmerger(){
    const id = randomId()
    requestProgress(id, gradioApp().getElementById('modelmerger_results_panel'), null)
    var res = create_submit_args(arguments)
    res[0] = id
    return res
}

function ask_for_style_name(_, prompt_text, negative_prompt_text) {
    name_ = prompt('Style name:')
    return [name_, prompt_text, negative_prompt_text]
}

function confirm_clear_prompt(prompt, negative_prompt) {
    prompt = ""
    negative_prompt = ""
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
            if (oldValue != newValue) opts = JSON.parse(textarea.value)
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
        if(counter.parentElement == prompt.parentElement) return
        prompt.parentElement.insertBefore(counter, prompt)
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

function getTranslation(...args) {
  return null
}

function update_token_counter(button_id) {
	if (token_timeouts[button_id])
		clearTimeout(token_timeouts[button_id]);
	token_timeouts[button_id] = setTimeout(() => gradioApp().getElementById(button_id)?.click(), wait_time);
}

function monitor_server_status() {
  document.open();
  document.write(`
    <html>
      <head><title>SD.Next</title></head>
      <body style="background: #222222; font-size: 1rem; font-family:monospace; margin-top:20%; color:lightgray; text-align:center">
        <h1>Waiting for server...</h1>
        <script>
          function monitor_server_status() {
            fetch('http://127.0.0.1:7860/sdapi/v1/progress')
              .then((res) => { !res?.ok ? setTimeout(monitor_server_status, 1000) : location.reload(); })
              .catch((e) => setTimeout(monitor_server_status, 1000))
          }
          window.onload = () => monitor_server_status();
        </script>
      </body>
    </html>
  `);
  document.close();
}

function restart_reload(){
    document.body.style = "background: #222222; font-size: 1rem; font-family:monospace; margin-top:20%; color:lightgray; text-align:center"
    document.body.innerHTML = "<h1>Server shutdown in progress...</h1>"
    fetch('http://127.0.0.1:7860/sdapi/v1/progress')
      .then((res) => setTimeout(restart_reload, 1000))
      .catch((e) => setTimeout(monitor_server_status, 500))
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

function create_theme_element() {
  el = document.createElement('img');
  el.id = 'theme-preview';
  el.className = 'theme-preview';
  el.onclick = () => el.style.display = 'none';
  document.body.appendChild(el);
  return el;
}

function preview_theme() {
  const name = gradioApp().getElementById('setting_gradio_theme').querySelectorAll('input')?.[0].value || '';
  if (name === 'black-orange' || name.startsWith('gradio/')) {
    el = document.getElementById('theme-preview') || create_theme_element();
    el.style.display = el.style.display === 'block' ? 'none' : 'block';
    if (name === 'black-orange') el.src = '/file=javascript/black-orange.jpg';
    else el.src = `/file=javascript/${name.replace('/', '-')}.jpg`;
  } else {
    fetch('/file=javascript/themes.json')
      .then((r) => r.json())
        .then(themes => {
          theme = themes.find((t)=> t.id === name);
          window.open(theme.subdomain, '_blank');
      });
  }
}

function reconnect_ui() {
  const el1 = gradioApp().getElementById('txt2img_gallery_container')
  const el2 = gradioApp().getElementById('txt2img_gallery')
  const task_id = localStorage.getItem('task')
  if (!el1 || !el2) return
  else clearInterval(start_check)
  if (task_id) {
    console.debug('task check:', task_id)
    rememberGallerySelection('txt2img_gallery')
    showSubmitButtons('txt2img', false)
    const atEnd = () => showSubmitButtons('txt2img', true)
    requestProgress(task_id, el1, el2, atEnd, null, true)
  }

  sd_model = gradioApp().getElementById("setting_sd_model_checkpoint")
  let loadingStarted = 0;
  let loadingMonitor = 0;
  const sd_model_callback = () => {
    loading = sd_model.querySelector(".eta-bar")
    if (!loading) {
      loadingStarted = 0
      clearInterval(loadingMonitor)
    } else {
      if (loadingStarted === 0) {
        loadingStarted = Date.now();
        loadingMonitor = setInterval(() => {
          elapsed = Date.now() - loadingStarted;
          console.log('Loading', elapsed)
          if (elapsed > 3000 && loading) loading.style.display = 'none';
        }, 5000);
      }
    }
  };
  const sd_model_observer = new MutationObserver(sd_model_callback);
  sd_model_observer.observe(sd_model, { attributes: true, childList: true, subtree: true });
}

var start_check = setInterval(reconnect_ui, 50);
