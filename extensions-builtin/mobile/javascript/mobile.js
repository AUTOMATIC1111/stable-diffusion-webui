var isSetupForMobile = false;

function isMobile() {
    for (var tab of ["txt2img", "img2img"]) {
        var imageTab = gradioApp().getElementById(tab + '_results');
        if (imageTab && imageTab.offsetParent && imageTab.offsetLeft == 0) {
            return true;
        }
    }

    return false;
}

function reportWindowSize() {
    if (gradioApp().querySelector('.toprow-compact-tools')) return; // not applicable for compact prompt layout

    var currentlyMobile = isMobile();
    if (currentlyMobile == isSetupForMobile) return;
    isSetupForMobile = currentlyMobile;

    for (var tab of ["txt2img", "img2img"]) {
        var button = gradioApp().getElementById(tab + '_generate_box');
        var target = gradioApp().getElementById(currentlyMobile ? tab + '_results' : tab + '_actions_column');
        target.insertBefore(button, target.firstElementChild);

        gradioApp().getElementById(tab + '_results').classList.toggle('mobile', currentlyMobile);
    }
}

window.addEventListener("resize", reportWindowSize);

onUiLoaded(function() {
    reportWindowSize();
});
