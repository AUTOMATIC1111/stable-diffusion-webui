
// added to fix a weird error in gradio 4.19 at page load
Object.defineProperty(Array.prototype, 'toLowerCase', {
    value: function() {
        return this;
    }
});
