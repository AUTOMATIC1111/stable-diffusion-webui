const delay = 250//ms
window.addEventListener('gamepadconnected', (e) => {
    setInterval(() => {
        if (!opts.js_modal_lightbox_gamepad) return;
        const gamepad = navigator.getGamepads()[0];
        const xValue = gamepad.axes[0];
        if (xValue < -0.3) {
            modalPrevImage(e);
        } else if (xValue > 0.3) {
            modalNextImage(e);
        }
    }, delay);
});

/*
Primarily for vr controller type pointer devices.
I use the wheel event because there's currently no way to do it properly with web xr.
 */
let isScrolling = false;
window.addEventListener('wheel', (e) => {
    if (!opts.js_modal_lightbox_gamepad || isScrolling) return;
    isScrolling = true;

    if (e.deltaX <= -0.6) {
        modalPrevImage(e);
    } else if (e.deltaX >= 0.6) {
        modalNextImage(e);
    }

    setTimeout(() => {
        isScrolling = false;
    }, delay);
});
