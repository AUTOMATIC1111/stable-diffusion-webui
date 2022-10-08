onUiUpdate(function(){
	if (!txt2img_textarea) {
		txt2img_textarea = gradioApp().querySelector("#txt2img_prompt > label > textarea");
		txt2img_textarea?.addEventListener("keydown", (event) => edit_attention(txt2img_textarea, event));
	}
	if (!img2img_textarea) {
		img2img_textarea = gradioApp().querySelector("#img2img_prompt > label > textarea");
		img2img_textarea?.addEventListener("keydown", (event) => edit_attention(img2img_textarea, event));
	}
})

function edit_attention(el, event) {
	const {key, ctrlKey} = event;
	if (!ctrlKey) return;

	let plus = "ArrowUp"
	let minus = "ArrowDown"
	if (key != plus && key != minus) return;

	selectionStart = el.selectionStart;
	selectionEnd = el.selectionEnd;
	if(selectionStart == selectionEnd) return;

	event.preventDefault();

	if (selectionStart == 0 || el.value[selectionStart - 1] != "(") {
		el.value = el.value.slice(0, selectionStart) +
			"(" + el.value.slice(selectionStart, selectionEnd) + ":1.0)" +
			el.value.slice(selectionEnd);

		el.focus();
		el.selectionStart = selectionStart + 1;
		el.selectionEnd = selectionEnd + 1;

	} else {
		end = el.value.slice(selectionEnd + 1).indexOf(")") + 1;
		weight = parseFloat(el.value.slice(selectionEnd + 1, selectionEnd + 1 + end));
		if (key == minus) weight -= 0.1;
		if (key == plus) weight += 0.1;

		weight = parseFloat(weight.toPrecision(12));

		el.value = el.value.slice(0, selectionEnd + 1) +
			weight +
			el.value.slice(selectionEnd + 1 + end - 1);

		el.focus();
		el.selectionStart = selectionStart;
		el.selectionEnd = selectionEnd;
	}
}
