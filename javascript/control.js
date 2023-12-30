function setupControlUI() {
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
