function keyupEditAttention(event) {
    let target = event.originalTarget || event.composedPath()[0];
    if (!target.matches("*:is([id*='_toprow'] [id*='_prompt'], .prompt) textarea")) return;
    if (!(event.metaKey || event.ctrlKey)) return;

    let isPlus = event.key == "ArrowUp";
    let isMinus = event.key == "ArrowDown";
    if (!isPlus && !isMinus) return;

    let selectionStart = target.selectionStart;
    let selectionEnd = target.selectionEnd;
    let text = target.value;

    function selectCurrentParenthesisBlock(OPEN, CLOSE) {
        if (selectionStart !== selectionEnd) return false;

        // Find opening parenthesis around current cursor
        const before = text.substring(0, selectionStart);
        let beforeParen = before.lastIndexOf(OPEN);
        if (beforeParen == -1) return false;
        let beforeParenClose = before.lastIndexOf(CLOSE);
        while (beforeParenClose !== -1 && beforeParenClose > beforeParen) {
            beforeParen = before.lastIndexOf(OPEN, beforeParen - 1);
            beforeParenClose = before.lastIndexOf(CLOSE, beforeParenClose - 1);
        }

        // Find closing parenthesis around current cursor
        const after = text.substring(selectionStart);
        let afterParen = after.indexOf(CLOSE);
        if (afterParen == -1) return false;
        let afterParenOpen = after.indexOf(OPEN);
        while (afterParenOpen !== -1 && afterParen > afterParenOpen) {
            afterParen = after.indexOf(CLOSE, afterParen + 1);
            afterParenOpen = after.indexOf(OPEN, afterParenOpen + 1);
        }
        if (beforeParen === -1 || afterParen === -1) return false;

        // Set the selection to the text between the parenthesis
        const parenContent = text.substring(beforeParen + 1, selectionStart + afterParen);
        const lastColon = parenContent.lastIndexOf(":");
        selectionStart = beforeParen + 1;
        selectionEnd = selectionStart + lastColon;
        target.setSelectionRange(selectionStart, selectionEnd);
        return true;
    }

    function selectCurrentWord() {
        if (selectionStart !== selectionEnd) return false;
        const delimiters = opts.keyedit_delimiters + " \r\n\t";

        // seek backward until to find beggining
        while (!delimiters.includes(text[selectionStart - 1]) && selectionStart > 0) {
            selectionStart--;
        }

        // seek forward to find end
        while (!delimiters.includes(text[selectionEnd]) && selectionEnd < text.length) {
            selectionEnd++;
        }

        target.setSelectionRange(selectionStart, selectionEnd);
        return true;
    }

    // If the user hasn't selected anything, let's select their current parenthesis block or word
    if (!selectCurrentParenthesisBlock('<', '>') && !selectCurrentParenthesisBlock('(', ')')) {
        selectCurrentWord();
    }

    event.preventDefault();

    var closeCharacter = ')';
    var delta = opts.keyedit_precision_attention;

    if (selectionStart > 0 && text[selectionStart - 1] == '<') {
        closeCharacter = '>';
        delta = opts.keyedit_precision_extra;
    } else if (selectionStart == 0 || text[selectionStart - 1] != "(") {

        // do not include spaces at the end
        while (selectionEnd > selectionStart && text[selectionEnd - 1] == ' ') {
            selectionEnd -= 1;
        }
        if (selectionStart == selectionEnd) {
            return;
        }

        text = text.slice(0, selectionStart) + "(" + text.slice(selectionStart, selectionEnd) + ":1.0)" + text.slice(selectionEnd);

        selectionStart += 1;
        selectionEnd += 1;
    }

    var end = text.slice(selectionEnd + 1).indexOf(closeCharacter) + 1;
    var weight = parseFloat(text.slice(selectionEnd + 1, selectionEnd + 1 + end));
    if (isNaN(weight)) return;

    weight += isPlus ? delta : -delta;
    weight = parseFloat(weight.toPrecision(12));
    if (String(weight).length == 1) weight += ".0";

    if (closeCharacter == ')' && weight == 1) {
        text = text.slice(0, selectionStart - 1) + text.slice(selectionStart, selectionEnd) + text.slice(selectionEnd + 5);
        selectionStart--;
        selectionEnd--;
    } else {
        text = text.slice(0, selectionEnd + 1) + weight + text.slice(selectionEnd + 1 + end - 1);
    }

    target.focus();
    target.value = text;
    target.selectionStart = selectionStart;
    target.selectionEnd = selectionEnd;

    updateInput(target);
}

addEventListener('keydown', (event) => {
    keyupEditAttention(event);
});
