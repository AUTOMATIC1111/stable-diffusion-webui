let gamepads = [];

window.addEventListener('gamepadconnected', (e) => {
    const index = e.gamepad.index;
    let isWaiting = false;
    gamepads[index] = setInterval(async() => {
        if (!opts.js_modal_lightbox_gamepad || isWaiting) return;
        const gamepad = navigator.getGamepads()[index];
        const xValue = gamepad.axes[0];
        if (xValue <= -0.3) {
            modalPrevImage(e);
            isWaiting = true;
        } else if (xValue >= 0.3) {
            modalNextImage(e);
            isWaiting = true;
        }
        if (isWaiting) {
            await sleepUntil(() => {
                const xValue = navigator.getGamepads()[index].axes[0];
                if (xValue < 0.3 && xValue > -0.3) {
                    return true;
                }
            }, opts.js_modal_lightbox_gamepad_repeat);
            isWaiting = false;
        }
    }, 10);
});

window.addEventListener('gamepaddisconnected', (e) => {
    clearInterval(gamepads[e.gamepad.index]);
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
    }, opts.js_modal_lightbox_gamepad_repeat);
});

function sleepUntil(f, timeout) {
    return new Promise((resolve) => {
        const timeStart = new Date();
        const wait = setInterval(function() {
            if (f() || new Date() - timeStart > timeout) {
                clearInterval(wait);
                resolve();
            }
        }, 20);
    });
}
