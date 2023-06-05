onUiUpdate(() => {
  gradioApp().querySelectorAll('span, button, select, p').forEach((span) => {
    tooltip = titles[span.textContent];
    if (!tooltip) tooltip = titles[span.value];
    if (!tooltip) {
      for (const c of span.classList) {
        if (c in titles) {
          tooltip = titles[c];
          break;
        }
      }
    }
    if (tooltip) span.title = tooltip;
  });

  gradioApp().querySelectorAll('select').forEach((select) => {
	    if (select.onchange != null) return;
	    select.onchange = () => select.title = titles[select.value] || '';
  });
});
