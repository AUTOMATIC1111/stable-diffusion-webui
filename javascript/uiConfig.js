function uiOpenSubmenus() {
  const accordions = Array.from(gradioApp().querySelectorAll('.gradio-accordion'));
  const states = {};
  accordions.forEach((el) => {
    const name = el.querySelector('.label-wrap > span:not(.icon)').innerText.trim();
    const children = Array.from(el.childNodes);
    const open = children.filter((c) => c.style?.display === 'block');
    if (states[name] === undefined) states[name] = open.length > 0;
  });
  return states;
}

function getUIDefaults() {
  const btn = gradioApp().getElementById('ui_defaults_view');
  if (!btn) return;
  const intersectionObserver = new IntersectionObserver((entries) => {
    if (entries[0].intersectionRatio <= 0) { }
    if (entries[0].intersectionRatio > 0) btn.click();
  });
  intersectionObserver.observe(btn); // monitor visibility of tab
}

onUiLoaded(getUIDefaults);
