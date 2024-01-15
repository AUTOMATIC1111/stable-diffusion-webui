const localeData = {
  data: [],
  timeout: null,
  finished: false,
  type: 2,
  el: null,
};

async function tooltipCreate() {
  localeData.el = document.createElement('div');
  localeData.el.className = 'tooltip';
  localeData.el.id = 'tooltip-container';
  localeData.el.innerText = 'this is a hint';
  gradioApp().appendChild(localeData.el);
  if (window.opts.tooltips === 'None') localeData.type = 0;
  if (window.opts.tooltips === 'Browser default') localeData.type = 1;
  if (window.opts.tooltips === 'UI tooltips') localeData.type = 2;
}

async function tooltipShow(e) {
  if (e.target.dataset.hint) {
    localeData.el.classList.add('tooltip-show');
    localeData.el.innerHTML = `<b>${e.target.textContent}</b><br>${e.target.dataset.hint}`;
  }
}

async function tooltipHide(e) {
  localeData.el.classList.remove('tooltip-show');
}

async function validateHints(elements, data) {
  let original = elements.map((e) => e.textContent.toLowerCase().trim()).sort((a, b) => a > b);
  original = [...new Set(original)]; // remove duplicates
  const current = data.map((e) => e.label.toLowerCase().trim()).sort((a, b) => a > b);
  log('all elements:', original);
  log('all hints:', current);
  log('hints-differences', { elements: original.length, hints: current.length });
  const missingLocale = original.filter((e) => !current.includes(e));
  log('missing in locale:', missingLocale);
  const missingUI = current.filter((e) => !original.includes(e));
  log('in locale but not ui:', missingUI);
}

async function setHints() {
  if (localeData.finished) return;
  if (localeData.data.length === 0) {
    const res = await fetch('/file=html/locale_en.json');
    const json = await res.json();
    localeData.data = Object.values(json).flat().filter((e) => e.hint.length > 0);
    for (const e of localeData.data) e.label = e.label.toLowerCase().trim();
  }
  const elements = [
    ...Array.from(gradioApp().querySelectorAll('button')),
    ...Array.from(gradioApp().querySelectorAll('label > span')),
  ];
  if (elements.length === 0) return;
  if (Object.keys(opts).length === 0) return;
  if (!localeData.el) tooltipCreate();
  let localized = 0;
  let hints = 0;
  localeData.finished = true;
  const t0 = performance.now();
  for (const el of elements) {
    const found = localeData.data.find((l) => l.label === el.textContent.toLowerCase().trim());
    if (found?.localized?.length > 0) {
      localized++;
      el.textContent = found.localized;
    }
    if (found?.hint?.length > 0) {
      hints++;
      if (localeData.type === 1) {
        el.title = found.hint;
      } else if (localeData.type === 2) {
        el.dataset.hint = found.hint;
        el.addEventListener('mouseover', tooltipShow);
        el.addEventListener('mouseout', tooltipHide);
      } else {
        // tooltips disabled
      }
    }
  }
  const t1 = performance.now();
  log('setHints', { type: localeData.type, elements: elements.length, localized, hints, data: localeData.data.length, time: t1 - t0 });
  // sortUIElements();
  removeSplash();
  // validateHints(elements, localeData.data);
}

onAfterUiUpdate(async () => {
  if (localeData.timeout) clearTimeout(localeData.timeout);
  localeData.timeout = setTimeout(setHints, 250);
});
