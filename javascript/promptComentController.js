(function() {
    const COMMENT_REG = /^\s*(#|\/\/)\s*/;
    const pos = {};
    let isFocus = false;
    let changedFirstLineCharCount = 0;

    function getSelectionPosition(el) {
        isFocus = true;
        // the obtained pos data is old when mouseup event.
        setTimeout(() => {
            pos.start = el.selectionStart;
            pos.end = el.selectionEnd;
        }, 0);
    }

    function textereaEventHandler(e) {
        const el = e.target;
        // Determine whether it is a PROMPT textarea, div[id="*_prompt"]
        if (el.nodeName === 'TEXTAREA' && (el.closest('div').id || '').endsWith('_prompt')) {
            switch (e.type) {
            case 'mouseup':
                // left mouse button
                if (e.which === 1) {
                    getSelectionPosition(el);
                }
                break;
            case 'keyup':
            case 'select':
                getSelectionPosition(el);
                break;
            }
        } else {
            isFocus = false;
        }
    }

    document.addEventListener('select', textereaEventHandler);
    document.addEventListener('keyup', textereaEventHandler);
    document.addEventListener('mouseup', textereaEventHandler);

    document.addEventListener('keydown', (e) => {
        if (!isFocus) return;
        // Ctrl + /
        if (e.ctrlKey && e.key === '/') {
            changedFirstLineCharCount = 0;
            const el = e.target;
            const txt = el.value;
            const oldTxtCount = txt.length;

            let lines = txt.split('\n');
            const len = lines.length;
            if (len === 1) {
                el.value = toggleComment(txt);
            } else {
                let charCountStart = 0;
                let charCountEnd = 0;
                const newLines = [];
                let line;

                // Determine whether to execute comment or uncoment
                let isExecuteComment = false;
                for (let i = 0; i < len; i++) {
                    line = lines[i];
                    // +1 \n
                    charCountEnd += line.length + 1;
                    if (charCountEnd > pos.start && charCountStart <= pos.end) {
                        if (!COMMENT_REG.test(line)) {
                            isExecuteComment = true;
                            break;
                        }
                    }
                    charCountStart = charCountEnd;
                }

                charCountStart = 0;
                charCountEnd = 0;

                for (let i = 0; i < len; i++) {
                    line = lines[i];
                    // +1 \n
                    charCountEnd += line.length + 1;
                    if (charCountEnd > pos.start && charCountStart <= pos.end) {
                        newLines.push(isExecuteComment ? comment(line) : uncomment(line));
                    } else {
                        newLines.push(line);
                    }
                    charCountStart = charCountEnd;
                }
                el.value = newLines.join('\n');
            }
            setSelection(el.value.length - oldTxtCount, el);
        }
    });

    function comment(line) {
        if (changedFirstLineCharCount === 0) changedFirstLineCharCount = 3;
        return '// ' + line;
    }

    function uncomment(line) {
        const newLine = line.replace(COMMENT_REG, '');
        if (changedFirstLineCharCount === 0) {
            changedFirstLineCharCount = newLine.length - line.length;
        }
        return newLine;
    }

    function toggleComment(line) {
        return COMMENT_REG.test(line) ? uncomment(line) : comment(line);
    }

    function setSelection(changedCount, el) {
        const isNotSelect = pos.start === pos.end;
        pos.start += changedFirstLineCharCount;
        pos.end = isNotSelect ? pos.start : pos.end + changedCount;
        el.setSelectionRange(pos.start, pos.end);
        updateInput?.(el);
    }
})();
