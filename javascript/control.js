function controlInputMode(inputMode, ...args) {
  const tab = gradioApp().querySelector('#control-tab-input button.selected');
  if (!tab) return ['Select', ...args];
  inputMode = tab.innerText;
  if (inputMode === 'Image') {
    if (!gradioApp().getElementById('control_input_select').classList.contains('hidden')) inputMode = 'Select';
    else if (!gradioApp().getElementById('control_input_resize').classList.contains('hidden')) inputMode = 'Outpaint';
    else if (!gradioApp().getElementById('control_input_inpaint').classList.contains('hidden')) inputMode = 'Inpaint';
  }
  return [inputMode, ...args];
}

async function setupControlUI() {
  const tabs = ['input', 'output', 'preview'];
  for (const tab of tabs) {
    const btn = gradioApp().getElementById(`control-${tab}-button`);
    if (!btn) continue; // eslint-disable-line no-continue
    btn.style.cursor = 'pointer';
    btn.onclick = () => {
      const t = gradioApp().getElementById(`control-tab-${tab}`);
      t.style.display = t.style.display === 'none' ? 'block' : 'none';
      const c = gradioApp().getElementById(`control-${tab}-column`);
      c.style.flexGrow = c.style.flexGrow === '0' ? '9' : '0';
    };
  }

  const el = gradioApp().getElementById('control-input-column');
  if (!el) return;
  const intersectionObserver = new IntersectionObserver((entries) => {
    if (entries[0].intersectionRatio > 0) {
      const tab = gradioApp().querySelector('#control-tabs > .tab-nav > .selected')?.innerText.toLowerCase() || ''; // selected tab name
      const btn = gradioApp().getElementById(`refresh_${tab}_models`);
      if (btn) btn.click();
    }
  });
  intersectionObserver.observe(el); // monitor visibility of tab

  log('initControlUI');
}

onUiLoaded(setupControlUI);
