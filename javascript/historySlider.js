function buildKey(a, b) {
    return a + "_" + b;
}

function capturePrompts(tabname) {
    const value = {};
    value.negative_prompt = gradioApp().querySelector(
        "#" + tabname + "_neg_prompt textarea"
    ).value;
    value.prompt = gradioApp().querySelector(
        "#" + tabname + "_prompt textarea"
    ).value;
    return value;
}
