addEventListener('keydown', (event) => {
	let target = event.originalTarget || event.composedPath()[0];
	if (!target.hasAttribute("placeholder")) return;
	if (!target.placeholder.toLowerCase().includes("prompt")) return;

	let plus = "ArrowUp"
	let minus = "ArrowDown"
	if (event.key != plus && event.key != minus) return;

	selectionStart = target.selectionStart;
	selectionEnd = target.selectionEnd;
	if(selectionStart == selectionEnd) return;

	event.preventDefault();

	if (selectionStart == 0 || target.value[selectionStart - 1] != "(") {
		target.value = target.value.slice(0, selectionStart) +
			"(" + target.value.slice(selectionStart, selectionEnd) + ":1.0)" +
			target.value.slice(selectionEnd);

		target.focus();
		target.selectionStart = selectionStart + 1;
		target.selectionEnd = selectionEnd + 1;

	} else {
		end = target.value.slice(selectionEnd + 1).indexOf(")") + 1;
		weight = parseFloat(target.value.slice(selectionEnd + 1, selectionEnd + 1 + end));
		if (isNaN(weight)) return;
		if (event.key == minus) weight -= 0.1;
		if (event.key == plus) weight += 0.1;

		weight = parseFloat(weight.toPrecision(12));

		target.value = target.value.slice(0, selectionEnd + 1) +
			weight +
			target.value.slice(selectionEnd + 1 + end - 1);

		target.focus();
		target.selectionStart = selectionStart;
		target.selectionEnd = selectionEnd;
	}
	// Since we've modified a Gradio Textbox component manually, we need to simulate an `input` DOM event to ensure its
	// internal Svelte data binding remains in sync.
	target.dispatchEvent(new Event("input", { bubbles: true }));
});
