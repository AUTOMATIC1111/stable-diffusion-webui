// this would not be needed if automatic run gradio with loop enabled

let loaded = false;
let interval;

function refresh() {
  const btn = gradioApp().getElementById('system_info_tab_refresh_btn') // we could cache this dom element
  if (!btn) return // but ui may get destroyed
  btn.click() // actual refresh is done from python code we just trigger it but simulating button click
}

function onHidden() { // stop refresh interval when tab is not visible
  if (!interval) return
  clearInterval(interval);
  interval = undefined;
}

function onVisible() { // start refresh interval tab is when visible
  if (interval) return // interval already started so dont start it again
  interval = setInterval(refresh, 1000);
}

function initLoading() { // triggered on gradio change to monitor when ui gets sufficiently constructed
  if (loaded) return
  const block = gradioApp().getElementById('system_info_tab');
  if (!block) return
  intersectionObserver = new IntersectionObserver((entries) => {
    if (entries[0].intersectionRatio <= 0) onHidden();
    if (entries[0].intersectionRatio > 0) onVisible();
  });
  intersectionObserver.observe(block); // monitor visibility of tab
}

function initInitial() { // just setup monitor for gradio events
  const mutationObserver = new MutationObserver(initLoading)
  mutationObserver.observe(gradioApp(), { childList: true, subtree: true }); // monitor changes to gradio
}

document.addEventListener('DOMContentLoaded', initInitial);
