const monitoredOpts = [
  { sd_model_checkpoint: null },
  {
    sd_backend: () => {
      gradioApp().getElementById('refresh_sd_model_checkpoint')?.click();
    },
  },
];

function updateOpts(json_string) {
  const settings_data = JSON.parse(json_string);
  for (const op of monitoredOpts) {
    const key = Object.keys(op)[0];
    const callback = op[key];
    if (opts[key] && opts[key] !== settings_data.values[key]) {
      console.log('updateOpts', key, opts[key], settings_data.values[key]);
      if (callback) callback();
    }
  }
  opts = settings_data.values;
  opts_metadata = settings_data.metadata;
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
  if (tab_dirty_indicator && tab_dirty_indicator.nextSibling) {
    tab_dirty_indicator.nextSibling.click();
  }
  gradioApp().querySelectorAll('#settings > .tab-content > .tabitem').forEach((elem) => {
    if (elem.id === 'settings_tab_licenses' || elem.id === 'settings_show_all_pages') return;
    elem.style.display = 'block';
  });
}

function markIfModified(setting_name, value) {
  const elem = gradioApp().getElementById(`modification_indicator_${setting_name}`);
  if (elem == null) return;
  // Use JSON.stringify to compare nested objects (e.g. arrays for checkbox-groups)
  const previous_value_json = JSON.stringify(opts[setting_name]);
  const changed_value = JSON.stringify(value) !== previous_value_json;
  if (changed_value) elem.title = `click to revert to previous value: ${previous_value_json}`;

  const is_unsaved = opts_metadata[setting_name].is_stored;
  if (is_unsaved) elem.title = 'custom value';
  elem.disabled = !(is_unsaved || changed_value);
  elem.classList.toggle('changed', changed_value);
  elem.classList.toggle('unsaved', is_unsaved);

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
  tab_nav_indicator.classList.toggle('unsaved', saved.size > 0);
  if (changed_items.size > 0) tab_nav_indicator.title += `click to reset ${changed_items.size} unapplied changes in this tab\n`;
  if (saved.size > 0) tab_nav_indicator.title += `${saved.size} custom values\n${unsaved.size} default values}`;
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

  const settings_search = gradioApp().querySelectorAll('#settings_search > label > textarea')[0];
  settings_search.oninput = (e) => {
    setTimeout(() => {
      showAllSettings();
      gradioApp().querySelectorAll('#tab_settings .tabitem').forEach((section) => {
        section.querySelectorAll('.dirtyable').forEach((setting) => {
          const visible = setting.innerText.toLowerCase().includes(e.target.value.toLowerCase()) || setting.id.toLowerCase().includes(e.target.value.toLowerCase());
          if (!visible) {
            setting.style.display = 'none';
          } else {
            setting.style.removeProperty('display');
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

onOptionsChanged(() => {
  const setting_elems = gradioApp().querySelectorAll('#settings [id^="setting_"]');
  setting_elems.forEach((elem) => {
    const setting_name = elem.id.replace('setting_', '');
    markIfModified(setting_name, opts[setting_name]);
  });
});

onUiLoaded(() => {
  const tab_nav_element = gradioApp().querySelector('#settings > .tab-nav');
  const tab_nav_buttons = gradioApp().querySelectorAll('#settings > .tab-nav > button');
  const tab_elements = gradioApp().querySelectorAll('#settings > div:not(.tab-nav)');
  const observer = new MutationObserver((mutations) => {
    const show_all_pages_dummy = gradioApp().getElementById('settings_show_all_pages');
    if (show_all_pages_dummy.style.display === 'none') { return; }
    function mutation_on_style(mut) {
      return mut.type === 'attributes' && mut.attributeName === 'style';
    }
    if (mutations.some(mutation_on_style)) {
      showAllSettings();
    }
  });
  const tab_content_wrapper = document.createElement('div');
  tab_content_wrapper.className = 'tab-content';
  tab_nav_element.parentElement.insertBefore(tab_content_wrapper, tab_nav_element.nextSibling);

  tab_elements.forEach((elem, index) => {
    // Move the modification indicator to the toplevel tab button
    const tab_name = elem.id.replace('settings_', '');
    const indicator = gradioApp().getElementById(`modification_indicator_${tab_name}`);
    tab_nav_element.insertBefore(indicator, tab_nav_buttons[index]);

    // Add the tab content to the wrapper
    tab_content_wrapper.appendChild(elem);

    // Add the mutation observer to the tab element
    observer.observe(elem, { attributes: true, attributeFilter: ['style'] });
  });
});
