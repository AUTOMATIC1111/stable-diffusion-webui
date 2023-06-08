let locale = {
  data: [],
  timeout: null,
  finished: false,
}

async function setLocale() {
  if (locale.finished) return;
  console.log('setLocale');
  if (locale.data.length === 0) {
    const res = await fetch('/file=html/locale_en.json');
    const json = await res.json(); 
    locale.data = Object.values(json).flat();
  }
  const elements = [
    ...Array.from(gradioApp().querySelectorAll('button')),
    ...Array.from(gradioApp().querySelectorAll('label > span')),
  ];
  for (el of elements) {
    const found = locale.data.find(l => l.label === el.textContent);
    if (found?.localized?.length > 0) el.textContent = found.localized;
    if (found?.hint?.length > 0) el.title = found.hint;
  }
  locale.finished = true;
}

onAfterUiUpdate(async () => {
  if (locale.timeout) clearTimeout(locale.timeout);
  locale.timeout = setTimeout(setLocale, 250)
});
