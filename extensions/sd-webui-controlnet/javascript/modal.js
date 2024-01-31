(function () {
    const cnetModalRegisteredElements = new Set();
    onUiUpdate(() => {
        // Get all the buttons that open a modal
        const btns = gradioApp().querySelectorAll(".cnet-modal-open");

        // Get all the <span> elements that close a modal
        const spans = document.querySelectorAll(".cnet-modal-close");

        // For each button, add a click event listener that opens the corresponding modal
        btns.forEach((btn) => {
            if (cnetModalRegisteredElements.has(btn)) return;
            cnetModalRegisteredElements.add(btn);

            const modalId = btn.id.replace('cnet-modal-open-', '');
            const modal = document.getElementById("cnet-modal-" + modalId);
            btn.addEventListener('click', () => {
                modal.style.display = "block";
            });
        });

        // For each <span> element, add a click event listener that closes the corresponding modal
        spans.forEach((span) => {
            if (cnetModalRegisteredElements.has(span)) return;
            cnetModalRegisteredElements.add(span);

            const modal = span.parentNode;
            span.addEventListener('click', () => {
                modal.style.display = "none";
            });
        });
    });
})();
