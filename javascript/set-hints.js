let locale = {
  data: [],
  timeout: null,
  finished: false,
}

async function setLocale() {
  if (locale.finished) return;
  if (locale.data.length === 0) {
    const res = await fetch('/file=html/locale_en.json');
    const json = await res.json(); 
    locale.data = Object.values(json).flat();
  }
  const elements = [
    ...Array.from(gradioApp().querySelectorAll('button')),
    ...Array.from(gradioApp().querySelectorAll('label > span')),
  ];
  if (elements.length === 0) return;
  let localized = 0;
  let hints = 0;
  for (el of elements) {
    const found = locale.data.find(l => l.label === el.textContent);
    if (found?.localized?.length > 0) {
      localized++;
      el.textContent = found.localized;
    }
    if (found?.hint?.length > 0) {
      hints++;
      el.title = found.hint;
    }
  }
  console.log('setLocale', { elements: elements.length, localized, hints, data: locale.data });
  locale.finished = true;
}

onAfterUiUpdate(async () => {
  if (locale.timeout) clearTimeout(locale.timeout);
  locale.timeout = setTimeout(setLocale, 250)
});
