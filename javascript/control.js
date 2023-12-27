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
  log('initControlUI');
}

onUiLoaded(setupControlUI);
