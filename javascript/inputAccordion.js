var observerAccordionOpen = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutationRecord) {
        var elem = mutationRecord.target;
        var open = elem.classList.contains('open');

        var accordion = elem.parentNode;
        accordion.classList.toggle('input-accordion-open', open);

        var checkbox = gradioApp().querySelector('#' + accordion.id + "-checkbox input");
        checkbox.checked = open;
        updateInput(checkbox);

        var extra = gradioApp().querySelector('#' + accordion.id + "-extra");
        if (extra) {
            extra.style.display = open ? "" : "none";
        }
    });
});

function inputAccordionChecked(id, checked) {
    var label = gradioApp().querySelector('#' + id + " .label-wrap");
    if (label.classList.contains('open') != checked) {
        label.click();
    }
}

onUiLoaded(function() {
    for (var accordion of gradioApp().querySelectorAll('.input-accordion')) {
        var labelWrap = accordion.querySelector('.label-wrap');
        observerAccordionOpen.observe(labelWrap, {attributes: true, attributeFilter: ['class']});

        var extra = gradioApp().querySelector('#' + accordion.id + "-extra");
        if (extra) {
            labelWrap.insertBefore(extra, labelWrap.lastElementChild);
        }
    }
});
