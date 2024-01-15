let settingsInitialized = false;
let opts_metadata = {};
const opts_tabs = {};

const monitoredOpts = [
  { sd_model_checkpoint: null },
  { sd_backend: () => gradioApp().getElementById('refresh_sd_model_checkpoint')?.click() },
];

const AppyOpts = [
  { compact_view: (val) => toggleCompact(val) },
  { gradio_theme: (val, old) => setTheme(val, old) },
  { font_size: (val) => setFontSize(val) },
];

async function updateOpts(json_string) {
  const settings_data = JSON.parse(json_string);
  const new_opts = settings_data.values;
  opts_metadata = settings_data.metadata;

  for (const op of monitoredOpts) {
    const key = Object.keys(op)[0];
    const callback = op[key];
    if (opts[key] && opts[key] !== settings_data.values[key]) {
      log('updateOpts', key, opts[key], settings_data.values[key]);
      if (callback) callback(new_opts[key], opts[key]);
    }
  }

  for (const op of AppyOpts) {
    const key = Object.keys(op)[0];
    const callback = op[key];
    if (callback) callback(new_opts[key], opts[key]);
  }

  opts = new_opts;

  Object.entries(opts_metadata).forEach(([opt, meta]) => {
    if (!opts_tabs[meta.tab_name]) opts_tabs[meta.tab_name] = {};
    if (!opts_tabs[meta.tab_name].unsaved_keys) opts_tabs[meta.tab_name].unsaved_keys = new Set();
    if (!opts_tabs[meta.tab_name].saved_keys) opts_tabs[meta.tab_name].saved_keys = new Set();
    if (!meta.is_stored) opts_tabs[meta.tab_name].unsaved_keys.add(opt);
    else opts_tabs[meta.tab_name].saved_keys.add(opt);
  });
}

function showAllSettings() {
  // Try to ensure that the show all settings tab is opened by clicking on its tab button
  const tab_dirty_indicator = gradioApp().getElementById('modification_indicator_show_all_pages');
  if (tab_dirty_indicator && tab_dirty_indicator.nextSibling) tab_dirty_indicator.nextSibling.click();
  gradioApp().querySelectorAll('#settings > .tab-content > .tabitem').forEach((elem) => {
    if (elem.id === 'settings_tab_licenses' || elem.id === 'settings_show_all_pages') return;
    elem.style.display = 'block';
  });
}

function markIfModified(setting_name, value) {
  if (!opts_metadata[setting_name]) return;
  const elem = gradioApp().getElementById(`modification_indicator_${setting_name}`);
  if (!elem) return;
  const previous_value = JSON.stringify(opts[setting_name]);
  const current_value = JSON.stringify(value);
  const changed_value = previous_value !== current_value;
  if (changed_value) elem.title = `click to revert to previous value: ${previous_value}`;
  const is_stored = opts_metadata[setting_name].is_stored;
  if (is_stored) elem.title = 'custom value';
  elem.disabled = !changed_value && !is_stored;
  elem.classList.toggle('changed', changed_value);
  elem.classList.toggle('saved', is_stored);

  const { tab_name } = opts_metadata[setting_name];
  if (!opts_tabs[tab_name].changed) opts_tabs[tab_name].changed = new Set();
  const changed_items = opts_tabs[tab_name].changed;
  if (changed_value) changed_items.add(setting_name);
  else changed_items.delete(setting_name);
  const unsaved = opts_tabs[tab_name].unsaved_keys;
  const saved = opts_tabs[tab_name].saved_keys;

  // Set the indicator on the tab nav element
  const tab_nav_indicator = gradioApp().getElementById(`modification_indicator_${tab_name}`);
  tab_nav_indicator.disabled = (changed_items.size === 0) && (unsaved.size === 0);
  tab_nav_indicator.title = '';
  tab_nav_indicator.classList.toggle('changed', changed_items.size > 0);
  tab_nav_indicator.classList.toggle('saved', saved.size > 0);
  if (changed_items.size > 0) tab_nav_indicator.title += `click to reset ${changed_items.size} unapplied changes in this tab\n`;
  if (saved.size > 0) tab_nav_indicator.title += `${saved.size} custom values\n${unsaved.size} default values}`;
  // elem.scrollIntoView({ behavior: 'smooth', block: 'center' }); // TODO why is scroll happening on every change if all pages are visible?
}

onAfterUiUpdate(async () => {
  if (Object.keys(opts).length !== 0) return;
  const json_elem = gradioApp().getElementById('settings_json');
  if (!json_elem) return;
  json_elem.parentElement.style.display = 'none';
  const textarea = json_elem.querySelector('textarea');
  const jsdata = textarea.value;
  updateOpts(jsdata);
  executeCallbacks(optionsChangedCallbacks);
  registerDragDrop();

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

  const settingsSearch = gradioApp().querySelectorAll('#settings_search > label > textarea')[0];
  settingsSearch.oninput = (e) => {
    setTimeout(() => {
      log('settingsSearch', e.target.value);
      showAllSettings();
      gradioApp().querySelectorAll('#tab_settings .tabitem').forEach((section) => {
        section.querySelectorAll('.dirtyable').forEach((setting) => {
          const visible = setting.innerText.toLowerCase().includes(e.target.value.toLowerCase()) || setting.id.toLowerCase().includes(e.target.value.toLowerCase());
          if (!visible) setting.style.display = 'none';
          else setting.style.removeProperty('display');
        });
      });
    }, 50);
  };
});

onOptionsChanged(() => {
  const setting_elems = gradioApp().querySelectorAll('#settings [id^="setting_"]');
  setting_elems.forEach((elem) => {
    const setting_name = elem.id.replace('setting_', '');
    markIfModified(setting_name, opts[setting_name]);
  });
});

async function initModels() {
  const warn = () => `
    <p style='color: white'>No models available</p>
    - Select a model from reference list to download or<br>
    - Set model path to a folder containing your models<br>
    Current model path: ${opts.ckpt_dir}<br>
  `;
  const el = gradioApp().getElementById('main_info');
  const en = gradioApp().getElementById('txt2img_extra_networks');
  if (!el || !en) return;
  const req = await fetch('/sdapi/v1/sd-models');
  const res = req.ok ? await req.json() : [];
  log('initModels', res.length);
  const ready = () => `
    <p style='color: white'>Ready</p>
    ${res.length} models available<br>
  `;
  el.innerHTML = res.length > 0 ? ready() : warn();
  el.style.display = 'block';
  setTimeout(() => el.style.display = 'none', res.length === 0 ? 30000 : 1500);
  if (res.length === 0) {
    if (en.classList.contains('hide')) gradioApp().getElementById('txt2img_extra_networks_btn').click();
    const repeat = setInterval(() => {
      const buttons = Array.from(gradioApp().querySelectorAll('#txt2img_model_subdirs > button')) || [];
      const reference = buttons.find((b) => b.innerText === 'Reference');
      if (reference) {
        clearInterval(repeat);
        reference.click();
        log('enReferenceSelect');
      }
    }, 100);
  }
}

async function initSettings() {
  if (settingsInitialized) return;
  settingsInitialized = true;
  const tabNavElements = gradioApp().querySelector('#settings > .tab-nav');
  const tabNavButtons = gradioApp().querySelectorAll('#settings > .tab-nav > button');
  const tabElements = gradioApp().querySelectorAll('#settings > div:not(.tab-nav)');
  const observer = new MutationObserver((mutations) => {
    const showAllPages = gradioApp().getElementById('settings_show_all_pages');
    if (showAllPages.style.display === 'none') return;
    const mutation = (mut) => mut.type === 'attributes' && mut.attributeName === 'style';
    if (mutations.some(mutation)) showAllSettings();
  });
  const tabContentWrapper = document.createElement('div');
  tabContentWrapper.className = 'tab-content';
  tabNavElements.parentElement.insertBefore(tabContentWrapper, tabNavElements.nextSibling);
  tabElements.forEach((elem, index) => {
    const tabName = elem.id.replace('settings_', '');
    const indicator = gradioApp().getElementById(`modification_indicator_${tabName}`);
    tabNavElements.insertBefore(indicator, tabNavButtons[index]);
    tabContentWrapper.appendChild(elem);
    observer.observe(elem, { attributes: true, attributeFilter: ['style'] });
  });
  log('initSettings');
}

onUiLoaded(initSettings);
onUiLoaded(initModels);
