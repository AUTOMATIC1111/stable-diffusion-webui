let promptTokenCountDebounceTime = 800;
let promptTokenCountTimeouts = {};
var promptTokenCountUpdateFunctions = {};

function update_txt2img_tokens(...args) {
    // Called from Gradio
    update_token_counter("txt2img_token_button");
    if (args.length == 2) {
        return args[0];
    }
    return args;
}

function update_img2img_tokens(...args) {
    // Called from Gradio
    update_token_counter("img2img_token_button");
    if (args.length == 2) {
        return args[0];
    }
    return args;
}

function update_token_counter(button_id) {
    if (opts.disable_token_counters) {
        return;
    }
    if (promptTokenCountTimeouts[button_id]) {
        clearTimeout(promptTokenCountTimeouts[button_id]);
    }
    promptTokenCountTimeouts[button_id] = setTimeout(
        () => gradioApp().getElementById(button_id)?.click(),
        promptTokenCountDebounceTime,
    );
}


function recalculatePromptTokens(name) {
    promptTokenCountUpdateFunctions[name]?.();
}

function recalculate_prompts_txt2img() {
    // Called from Gradio
    recalculatePromptTokens('txt2img_prompt');
    recalculatePromptTokens('txt2img_neg_prompt');
    return Array.from(arguments);
}

function recalculate_prompts_img2img() {
    // Called from Gradio
    recalculatePromptTokens('img2img_prompt');
    recalculatePromptTokens('img2img_neg_prompt');
    return Array.from(arguments);
}

function setupTokenCounting(id, id_counter, id_button) {
    var prompt = gradioApp().getElementById(id);
    var counter = gradioApp().getElementById(id_counter);
    var textarea = gradioApp().querySelector(`#${id} > label > textarea`);

    if (opts.disable_token_counters) {
        counter.style.display = "none";
        return;
    }

    if (counter.parentElement == prompt.parentElement) {
        return;
    }

    prompt.parentElement.insertBefore(counter, prompt);
    prompt.parentElement.style.position = "relative";

    promptTokenCountUpdateFunctions[id] = function() {
        update_token_counter(id_button);
    };
    textarea.addEventListener("input", promptTokenCountUpdateFunctions[id]);
}

function setupTokenCounters() {
    setupTokenCounting('txt2img_prompt', 'txt2img_token_counter', 'txt2img_token_button');
    setupTokenCounting('txt2img_neg_prompt', 'txt2img_negative_token_counter', 'txt2img_negative_token_button');
    setupTokenCounting('img2img_prompt', 'img2img_token_counter', 'img2img_token_button');
    setupTokenCounting('img2img_neg_prompt', 'img2img_negative_token_counter', 'img2img_negative_token_button');
}
