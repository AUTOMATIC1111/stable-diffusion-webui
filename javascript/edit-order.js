/* alt+left/right moves text in prompt */

function keyupEditOrder(event) {
    if (!opts.keyedit_move) return;

    let target = event.originalTarget || event.composedPath()[0];
    if (!target.matches("*:is([id*='_toprow'] [id*='_prompt'], .prompt) textarea")) return;
    if (!event.altKey) return;

    let isLeft = event.key == "ArrowLeft";
    let isRight = event.key == "ArrowRight";
    if (!isLeft && !isRight) return;
    event.preventDefault();

    let selectionStart = target.selectionStart;
    let selectionEnd = target.selectionEnd;
    let text = target.value;
    let items = text.split(",");
    let indexStart = (text.slice(0, selectionStart).match(/,/g) || []).length;
    let indexEnd = (text.slice(0, selectionEnd).match(/,/g) || []).length;
    let range = indexEnd - indexStart + 1;

    if (isLeft && indexStart > 0) {
        items.splice(indexStart - 1, 0, ...items.splice(indexStart, range));
        target.value = items.join();
        target.selectionStart = items.slice(0, indexStart - 1).join().length + (indexStart == 1 ? 0 : 1);
        target.selectionEnd = items.slice(0, indexEnd).join().length;
    } else if (isRight && indexEnd < items.length - 1) {
        items.splice(indexStart + 1, 0, ...items.splice(indexStart, range));
        target.value = items.join();
        target.selectionStart = items.slice(0, indexStart + 1).join().length + 1;
        target.selectionEnd = items.slice(0, indexEnd + 2).join().length;
    }

    event.preventDefault();
    updateInput(target);
}

addEventListener('keydown', (event) => {
    keyupEditOrder(event);
});
