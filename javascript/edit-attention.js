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

        // Find closing parenthesis around current cursor
        const after = text.substring(selectionStart);
        let afterParen = after.indexOf(CLOSE);
        if (afterParen == -1) return false;

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
        const whitespace_delimiters = {"Tab": "\t", "Carriage Return": "\r", "Line Feed": "\n"};
        let delimiters = opts.keyedit_delimiters;

        for (let i of opts.keyedit_delimiters_whitespace) {
            delimiters += whitespace_delimiters[i];
        }

        // seek backward to find beginning
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
    var weight = parseFloat(text.slice(selectionEnd + 1, selectionEnd + end));
    if (isNaN(weight)) return;

    weight += isPlus ? delta : -delta;
    weight = parseFloat(weight.toPrecision(12));
    if (String(weight).length == 1) weight += ".0";

    if (closeCharacter == ')' && weight == 1) {
        var endParenPos = text.substring(selectionEnd).indexOf(')');
        text = text.slice(0, selectionStart - 1) + text.slice(selectionStart, selectionEnd) + text.slice(selectionEnd + endParenPos + 1);
        selectionStart--;
        selectionEnd--;
    } else {
        text = text.slice(0, selectionEnd + 1) + weight + text.slice(selectionEnd + end);
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
