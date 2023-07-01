window.opts = {};
window.localization = {};
window.titles = {};
let tabSelected = '';

function set_theme(theme) {
  const gradioURL = window.location.href;
  if (!gradioURL.includes('?__theme=')) window.location.replace(`${gradioURL}?__theme=${theme}`);
}

function clip_gallery_urls(gallery) {
  const files = gallery.map((v) => v.data);
  navigator.clipboard.writeText(JSON.stringify(files)).then(
    () => console.log('clipboard set:', files),
    (err) => console.log('clipboard error:', files, err)
  );
}

function all_gallery_buttons() {
  const allGalleryButtons = gradioApp().querySelectorAll('[style="display: block;"].tabitem div[id$=_gallery].gradio-gallery .thumbnails > .thumbnail-item.thumbnail-small');
  const visibleGalleryButtons = [];
  allGalleryButtons.forEach((elem) => {
    if (elem.parentElement.offsetParent) visibleGalleryButtons.push(elem);
  });
  return visibleGalleryButtons;
}

function selected_gallery_button() {
  const allCurrentButtons = gradioApp().querySelectorAll('[style="display: block;"].tabitem div[id$=_gallery].gradio-gallery .thumbnail-item.thumbnail-small.selected');
  let visibleCurrentButton = null;
  allCurrentButtons.forEach((elem) => {
    if (elem.parentElement.offsetParent) visibleCurrentButton = elem;
  });
  return visibleCurrentButton;
}

function selected_gallery_index() {
  const buttons = all_gallery_buttons();
  const button = selected_gallery_button();
  let result = -1;
  buttons.forEach((v, i) => { if (v === button) { result = i; } });
  return result;
}

function extract_image_from_gallery(gallery) {
  if (gallery.length === 0) return [null];
  if (gallery.length === 1) return [gallery[0]];
  let index = selected_gallery_index();
  if (index < 0 || index >= gallery.length) index = 0;
  return [gallery[index]];
}

window.args_to_array = Array.from; // Compatibility with e.g. extensions that may expect this to be around

function switch_to_txt2img(...args) {
  gradioApp().querySelector('#tabs').querySelectorAll('button')[0].click();
  return Array.from(arguments);
}

function switch_to_img2img_tab(no) {
  gradioApp().querySelector('#tabs').querySelectorAll('button')[1].click();
  gradioApp().getElementById('mode_img2img').querySelectorAll('button')[no].click();
}

function switch_to_img2img(...args) {
  switch_to_img2img_tab(0);
  return Array.from(arguments);
}

function switch_to_sketch(...args) {
  switch_to_img2img_tab(1);
  return Array.from(arguments);
}

function switch_to_inpaint(...args) {
  switch_to_img2img_tab(2);
  return Array.from(arguments);
}

function switch_to_inpaint_sketch(...args) {
  switch_to_img2img_tab(3);
  return Array.from(arguments);
}

function switch_to_extras(...args) {
  gradioApp().querySelector('#tabs').querySelectorAll('button')[2].click();
  return Array.from(arguments);
}

function get_tab_index(tabId) {
  let res = 0;
  gradioApp().getElementById(tabId).querySelector('div').querySelectorAll('button')
    .forEach((button, i) => {
      if (button.className.indexOf('selected') !== -1) res = i;
    });
  return res;
}

function create_tab_index_args(tabId, args) {
  let res = Array.from(args);
  res[0] = get_tab_index(tabId);
  return res;
}

function get_img2img_tab_index(...args) {
  let res = Array.from(arguments);
  res.splice(-2);
  res[0] = get_tab_index('mode_img2img');
  return res;
}

function create_submit_args(args) {
  var res = Array.from(args);
  // As it is currently, txt2img and img2img send back the previous output args (txt2img_gallery, generation_info, html_info) whenever you generate a new image.
  // This can lead to uploading a huge gallery of previously generated images, which leads to an unnecessary delay between submitting and beginning to generate.
  // I don't know why gradio is sending outputs along with inputs, but we can prevent sending the image gallery here, which seems to be an issue for some.
  // If gradio at some point stops sending outputs, this may break something
  if (Array.isArray(res[res.length - 3])) res[res.length - 3] = null;
  return res;
}

function showSubmitButtons(tabname, show) {}

function clearGallery(tabname) {
  const gallery = gradioApp().getElementById(`${tabname}_gallery`)
  gallery.classList.remove('logo');
  // gallery.style.height = window.innerHeight - gallery.getBoundingClientRect().top - 200 + 'px'
  const footer = gradioApp().getElementById(`${tabname}_footer`)
  footer.style.display = 'flex';
}

function submit(...args) {
  console.log('Submit txt2img');
  rememberGallerySelection('txt2img_gallery');
  clearGallery('txt2img');
  const id = randomId();
  requestProgress(id, null, gradioApp().getElementById('txt2img_gallery'));
  const res = create_submit_args(args);
  res[0] = id;
  return res;
}

function submit_img2img(...args) {
  console.log('Submit img2img');
  rememberGallerySelection('img2img_gallery');
  clearGallery('img2img');
  const id = randomId();
  requestProgress(id, null, gradioApp().getElementById('img2img_gallery'));
  const res = create_submit_args(args);
  res[0] = id;
  res[1] = get_tab_index('mode_img2img');
  return res;
}

function submit_postprocessing(...args) {
  console.log('Submit extras');
  clearGallery('extras');
  return args
}


function modelmerger(...args) {
  const id = randomId();
  const res = create_submit_args(args);
  res[0] = id;
  return res;
}

function ask_for_style_name(_, prompt_text, negative_prompt_text) {
  const name = prompt('Style name:');
  return [name, prompt_text, negative_prompt_text];
}

function confirm_clear_prompt(prompt, negative_prompt) {
  prompt = '';
  negative_prompt = '';
  return [prompt, negative_prompt];
}

const promptTokecountUpdateFuncs = {};

function recalculatePromptTokens(name) {
  if (promptTokecountUpdateFuncs[name]) {
    promptTokecountUpdateFuncs[name]();
  }
}

function recalculate_prompts_txt2img(...args) {
  recalculatePromptTokens('txt2img_prompt');
  recalculatePromptTokens('txt2img_neg_prompt');
  return Array.from(arguments);
}

function recalculate_prompts_img2img(...args) {
  recalculatePromptTokens('img2img_prompt');
  recalculatePromptTokens('img2img_neg_prompt');
  return Array.from(arguments);
}

function recalculate_prompts_inpaint(...args) {
  recalculatePromptTokens('img2img_prompt');
  recalculatePromptTokens('img2img_neg_prompt');
  return Array.from(arguments);
}

function register_drag_drop() {
  const qs = gradioApp().getElementById('quicksettings');
  if (!qs) return;
  qs.addEventListener('dragover', (evt) => {
    evt.preventDefault();
    evt.dataTransfer.dropEffect = 'copy';
  });
  qs.addEventListener('drop', (evt) => {
    evt.preventDefault();
    evt.dataTransfer.dropEffect = 'copy';
    for (const f of evt.dataTransfer.files) {
      console.log('QuickSettingsDrop', f);
    }
  });
}

opts = {}
opts_metadata = {}
function updateOpts(json_string){
    let settings_data = JSON.parse(json_string)
    opts = settings_data.values
    opts_metadata = settings_data.metadata
    opts_tabs = {}
    Object.entries(opts_metadata).forEach(([opt, meta]) => {
        opts_tabs[meta.tab_name] ||= {}
        let unsaved = (opts_tabs[meta.tab_name].unsaved_keys ||= new Set())
        if (!meta.is_stored) unsaved.add(opt)
    })
}

function showAllSettings() {
  // Try to ensure that the show all settings tab is opened by clicking on its tab button
  let tab_dirty_indicator = gradioApp().getElementById('modification_indicator_show_all_pages');
  if (tab_dirty_indicator && tab_dirty_indicator.nextSibling) {
    tab_dirty_indicator.nextSibling.click();
  }
  gradioApp().querySelectorAll('#settings > .tab-content > .tabitem').forEach((elem) => {
    if (elem.id === 'settings_tab_licenses' || elem.id === 'settings_show_all_pages') return;
    elem.style.display = 'block';
  });
}

onAfterUiUpdate(() => {
  sort_ui_elements();
  if (Object.keys(opts).length !== 0) return;
  const json_elem = gradioApp().getElementById('settings_json');
  if (!json_elem) return;
  json_elem.parentElement.style.display = 'none';
  const textarea = json_elem.querySelector('textarea');
  const jsdata = textarea.value;
  updateOpts(jsdata);
  executeCallbacks(optionsChangedCallbacks);
  register_drag_drop();

  Object.defineProperty(textarea, 'value', {
    set(newValue) {
      const valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
      const oldValue = valueProp.get.call(textarea);
      valueProp.set.call(textarea, newValue);
      if (oldValue !== newValue) updateOpts(textarea.value);
      executeCallbacks(optionsChangedCallbacks);
    },
    get() {
      const valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
      return valueProp.get.call(textarea);
    },
  });

  function registerTextarea(id, id_counter, id_button) {
    const prompt = gradioApp().getElementById(id);
    const counter = gradioApp().getElementById(id_counter);
    const localTextarea = gradioApp().querySelector(`#${id} > label > textarea`);
    if (counter.parentElement === prompt.parentElement) return;
    prompt.parentElement.insertBefore(counter, prompt);
    prompt.parentElement.style.position = 'relative';
    promptTokecountUpdateFuncs[id] = () => { update_token_counter(id_button); };
    localTextarea.addEventListener('input', promptTokecountUpdateFuncs[id]);
  }

  registerTextarea('txt2img_prompt', 'txt2img_token_counter', 'txt2img_token_button');
  registerTextarea('txt2img_neg_prompt', 'txt2img_negative_token_counter', 'txt2img_negative_token_button');
  registerTextarea('img2img_prompt', 'img2img_token_counter', 'img2img_token_button');
  registerTextarea('img2img_neg_prompt', 'img2img_negative_token_counter', 'img2img_negative_token_button');

  const settings_search = gradioApp().querySelectorAll('#settings_search > label > textarea')[0];
  settings_search.oninput = (e) => {
    setTimeout(() => {
      showAllSettings();
      gradioApp().querySelectorAll('#tab_settings .tabitem').forEach((section) => {
        section.querySelectorAll('.dirtyable').forEach((setting) => {
          const visible = setting.innerText.toLowerCase().includes(e.target.value.toLowerCase()) || setting.id.toLowerCase().includes(e.target.value.toLowerCase());
          if (!visible) {
            setting.style.display = 'none'
          } else {
            setting.style.removeProperty('display')
          }
        });
      });
    }, 50);
  };
});

onOptionsChanged(() => {
  const elem = gradioApp().getElementById('sd_checkpoint_hash');
  const sd_checkpoint_hash = opts.sd_checkpoint_hash || '';
  const shorthash = sd_checkpoint_hash.substring(0, 10);

  if (elem && elem.textContent !== shorthash) {
    elem.textContent = shorthash;
    elem.title = sd_checkpoint_hash;
    elem.href = `https://google.com/search?q=${sd_checkpoint_hash}`;
  }
});

onOptionsChanged(function(){
    let setting_elems = gradioApp().querySelectorAll('#settings [id^="setting_"]')
    setting_elems.forEach(function(elem){
        setting_name = elem.id.replace("setting_", "")
        markIfModified(setting_name, opts[setting_name])
    })
})

onUiLoaded(function(){
    let tab_nav_element = gradioApp().querySelector('#settings > .tab-nav')
    let tab_nav_buttons = gradioApp().querySelectorAll('#settings > .tab-nav > button')
    let tab_elements = gradioApp().querySelectorAll('#settings > div:not(.tab-nav)')

    // HACK Add mutation observer to keep gradio from closing setting tabs when showing all pages
    const observer = new MutationObserver(function(mutations) {
        const show_all_pages_dummy = gradioApp().getElementById('settings_show_all_pages')
        if (show_all_pages_dummy.style.display == "none") 
            return;
        function mutation_on_style(mut) {
            return mut.type === 'attributes' && mut.attributeName === 'style'
        }
        if (mutations.some(mutation_on_style)) {
            showAllSettings();
        }
    })

    // Add a wrapper for the tab content (everything but the tab nav)
    const tab_content_wrapper = document.createElement('div')
    tab_content_wrapper.className = "tab-content"
    tab_nav_element.parentElement.insertBefore(tab_content_wrapper, tab_nav_element.nextSibling)

    tab_elements.forEach(function(elem, index){
        // Move the modification indicator to the toplevel tab button
        let tab_name = elem.id.replace("settings_", "")
        let indicator = gradioApp().getElementById("modification_indicator_"+tab_name)
        tab_nav_element.insertBefore(indicator, tab_nav_buttons[index])

        // Add the tab content to the wrapper
        tab_content_wrapper.appendChild(elem)

        // Add the mutation observer to the tab element
        observer.observe(elem, { attributes: true, attributeFilter: ['style'] })
    })
})

function markIfModified(setting_name, value) {
    let elem = gradioApp().getElementById("modification_indicator_"+setting_name)
    if(elem == null) return;
    // Use JSON.stringify to compare nested objects (e.g. arrays for checkbox-groups)
    let previous_value_json = JSON.stringify(opts[setting_name])
    let changed_value = JSON.stringify(value) != previous_value_json
    if (changed_value) {
        elem.title = `Click to revert to previous value: ${previous_value_json}`
    }

    is_unsaved = !opts_metadata[setting_name].is_stored
    if (is_unsaved) {
        elem.title = 'Default value (not saved to config)';
    }
    elem.disabled = !(is_unsaved || changed_value)
    elem.classList.toggle('changed', changed_value)
    elem.classList.toggle('unsaved', is_unsaved)

    let tab_name = opts_metadata[setting_name].tab_name
    let changed_items = (opts_tabs[tab_name].changed ||= new Set())
    changed_value ? changed_items.add(setting_name) : changed_items.delete(setting_name)
    let unsaved = opts_tabs[tab_name].unsaved_keys

    // Set the indicator on the tab nav element
    let tab_nav_indicator = gradioApp().getElementById('modification_indicator_'+tab_name)
    tab_nav_indicator.disabled = (changed_items.size == 0) && (unsaved.size == 0)
    tab_nav_indicator.title = '';
    tab_nav_indicator.classList.toggle('changed', changed_items.size > 0)
    tab_nav_indicator.classList.toggle('unsaved', unsaved.size > 0)
    if (changed_items.size > 0)
        tab_nav_indicator.title += `Click to reset ${changed_items.size} unapplied change${changed_items.size > 1 ? 's': ''} in this tab.\n`
    if (unsaved.size > 0)
        tab_nav_indicator.title += `${unsaved.size} new default value${unsaved.size > 1 ? 's':''} (not yet saved).`;
}

let txt2img_textarea;
let img2img_textarea;
const wait_time = 800;
const token_timeouts = {};

function update_txt2img_tokens(...args) {
  update_token_counter('txt2img_token_button');
  if (args.length === 2) return args[0];
  return args;
}

function update_img2img_tokens(...args) {
  update_token_counter('img2img_token_button');
  if (args.length === 2) return args[0];
  return args;
}

function getTranslation(...args) {
  return null;
}

function update_token_counter(button_id) {
  if (token_timeouts[button_id]) clearTimeout(token_timeouts[button_id]);
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
            fetch('/sdapi/v1/progress')
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

function restart_reload() {
  document.body.style = 'background: #222222; font-size: 1rem; font-family:monospace; margin-top:20%; color:lightgray; text-align:center';
  document.body.innerHTML = '<h1>Server shutdown in progress...</h1>';
  fetch('/sdapi/v1/progress')
    .then((res) => setTimeout(restart_reload, 1000))
    .catch((e) => setTimeout(monitor_server_status, 500));
  return [];
}

function updateInput(target) {
  const e = new Event('input', { bubbles: true });
  Object.defineProperty(e, 'target', { value: target });
  target.dispatchEvent(e);
}

let desiredCheckpointName = null;
function selectCheckpoint(name) {
  desiredCheckpointName = name;
  gradioApp().getElementById('change_checkpoint').click();
}

function currentImg2imgSourceResolution(_a, _b, scaleBy) {
  const img = gradioApp().querySelector('#mode_img2img > div[style="display: block;"] img');
  return img ? [img.naturalWidth, img.naturalHeight, scaleBy] : [0, 0, scaleBy];
}

function updateImg2imgResizeToTextAfterChangingImage() {
  setTimeout(() => gradioApp().getElementById('img2img_update_resize_to').click(), 500);
  return [];
}

function create_theme_element() {
  const el = document.createElement('img');
  el.id = 'theme-preview';
  el.className = 'theme-preview';
  el.onclick = () => { el.style.display = 'none'; };
  document.body.appendChild(el);
  return el;
}

function sort_ui_elements() {
  // sort top-level tabs
  const currSelected = gradioApp()?.querySelector('.tab-nav > .selected')?.innerText;
  if (currSelected === tabSelected || !opts.ui_tab_reorder) return;
  tabSelected = currSelected;
  const tabs = gradioApp().getElementById('tabs')?.children[0];
  if (!tabs) return;
  let tabsOrder = opts.ui_tab_reorder?.split(',').map((el) => el.trim().toLowerCase()) || [];
  for (const el of Array.from(tabs.children)) {
    const elIndex = tabsOrder.indexOf(el.innerText.toLowerCase());
    if (elIndex > -1) el.style.order = elIndex - 50; // default is 0 so setting to negative values
  }
  // sort always-on scripts
  const find = (el, ordered) => {
    for (const i in ordered) {
      if (el.innerText.toLowerCase().startsWith(ordered[i])) return i;
    }
    return 99;
  };

  tabsOrder = opts.ui_scripts_reorder?.split(',').map((el) => el.trim().toLowerCase()) || [];

  const scriptsTxt = gradioApp().getElementById('scripts_alwayson_txt2img').children;
  for (const el of Array.from(scriptsTxt)) el.style.order = find(el, tabsOrder);

  const scriptsImg = gradioApp().getElementById('scripts_alwayson_img2img').children;
  for (const el of Array.from(scriptsImg)) el.style.order = find(el, tabsOrder);
}

function preview_theme() {
  const name = gradioApp().getElementById('setting_gradio_theme').querySelectorAll('input')?.[0].value || '';
  if (name === 'black-orange' || name.startsWith('gradio/')) {
    const el = document.getElementById('theme-preview') || create_theme_element();
    el.style.display = el.style.display === 'block' ? 'none' : 'block';
    if (name === 'black-orange') el.src = '/file=html/black-orange.jpg';
    else el.src = `/file=html/${name.replace('/', '-')}.jpg`;
  } else {
    fetch('/file=html/themes.json')
      .then((r) => r.json())
      .then((themes) => {
        const theme = themes.find((t) => t.id === name);
        window.open(theme.subdomain, '_blank');
      });
  }
}

let uiLoaded = false;

function reconnect_ui() {
  const api_logo = Array.from(gradioApp().querySelectorAll('img')).filter((el) => el?.src?.endsWith('api-logo.svg'));
  if (api_logo.length > 0) api_logo[0].remove();

  const gallery = gradioApp().getElementById('txt2img_gallery');
  const task_id = localStorage.getItem('task');
  if (!gallery) return;
  clearInterval(start_check);
  if (task_id) {
    console.debug('task check:', task_id);
    rememberGallerySelection('txt2img_gallery');
    requestProgress(task_id, null, gallery, null, null, true);
  }
  uiLoaded = true;

  const sd_model = gradioApp().getElementById('setting_sd_model_checkpoint');
  let loadingStarted = 0;
  let loadingMonitor = 0;
  const sd_model_callback = () => {
    const loading = sd_model.querySelector('.eta-bar');
    if (!loading) {
      loadingStarted = 0;
      clearInterval(loadingMonitor);
    } else if (loadingStarted === 0) {
      loadingStarted = Date.now();
      loadingMonitor = setInterval(() => {
        const elapsed = Date.now() - loadingStarted;
        if (elapsed > 3000 && loading) loading.style.display = 'none';
      }, 5000);
    }
  };
  const sd_model_observer = new MutationObserver(sd_model_callback);
  sd_model_observer.observe(sd_model, { attributes: true, childList: true, subtree: true });
}

const start_check = setInterval(reconnect_ui, 50);
