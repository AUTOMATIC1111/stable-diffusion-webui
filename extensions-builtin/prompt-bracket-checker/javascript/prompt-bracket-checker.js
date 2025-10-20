// Stable Diffusion WebUI - Bracket Checker
// By @Bwin4L, @akx, @w-e-w, @Haoming02
// Counts open and closed brackets (round, square, curly) in the prompt and negative prompt text boxes in the txt2img and img2img tabs.
// If there's a mismatch, the keyword counter turns red, and if you hover on it, a tooltip tells you what's wrong.

function checkBrackets(textArea, counterElem) {
    const pairs = [
        ['(', ')', 'round brackets'],
        ['[', ']', 'square brackets'],
        ['{', '}', 'curly brackets']
    ];

    const counts = {};
    const errors = new Set();
    let i = 0;

    while (i < textArea.value.length) {
        let char = textArea.value[i];
        let escaped = false;
        while (char === '\\' && i + 1 < textArea.value.length) {
            escaped = !escaped;
            i++;
            char = textArea.value[i];
        }

        if (escaped) {
            i++;
            continue;
        }

        for (const [open, close, label] of pairs) {
            if (char === open) {
                counts[label] = (counts[label] || 0) + 1;
            } else if (char === close) {
                counts[label] = (counts[label] || 0) - 1;
                if (counts[label] < 0) {
                    errors.add(`Incorrect order of ${label}.`);
                }
            }
        }

        i++;
    }

    for (const [open, close, label] of pairs) {
        if (counts[label] == undefined) {
            continue;
        }

        if (counts[label] > 0) {
            errors.add(`${open} ... ${close} - Detected ${counts[label]} more opening than closing ${label}.`);
        } else if (counts[label] < 0) {
            errors.add(`${open} ... ${close} - Detected ${-counts[label]} more closing than opening ${label}.`);
        }
    }

    counterElem.title = [...errors].join('\n');
    counterElem.classList.toggle('error', errors.size !== 0);
}

function setupBracketChecking(id_prompt, id_counter) {
    const textarea = gradioApp().querySelector(`#${id_prompt} > label > textarea`);
    const counter = gradioApp().getElementById(id_counter);

    if (textarea && counter) {
        onEdit(`${id_prompt}_BracketChecking`, textarea, 400, () => checkBrackets(textarea, counter));
    }
}

onUiLoaded(function() {
    setupBracketChecking('txt2img_prompt', 'txt2img_token_counter');
    setupBracketChecking('txt2img_neg_prompt', 'txt2img_negative_token_counter');
    setupBracketChecking('img2img_prompt', 'img2img_token_counter');
    setupBracketChecking('img2img_neg_prompt', 'img2img_negative_token_counter');
});
