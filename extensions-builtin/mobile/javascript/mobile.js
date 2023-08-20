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
    var currentlyMobile = isMobile();
    if (currentlyMobile == isSetupForMobile) return;
    isSetupForMobile = currentlyMobile;

    for (var tab of ["txt2img", "img2img"]) {
        var button = gradioApp().getElementById(tab + '_generate_box');
        var target = null;
        if (currentlyMobile) {
            target = gradioApp().getElementById(tab + '_results');
        } else {
            target = gradioApp().getElementById(tab + '_actions_column') || gradioApp().getElementById(tab + '_generate_row');
        }

        target.insertBefore(button, target.firstElementChild);
    }
}

window.addEventListener("resize", reportWindowSize);
