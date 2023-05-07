const delay = 350//ms
let isWaiting = false;
window.addEventListener('gamepadconnected', (e) => {
    setInterval(async () => {
        if (!opts.js_modal_lightbox_gamepad || isWaiting) return;
        const gamepad = navigator.getGamepads()[0];
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
                const xValue = navigator.getGamepads()[0].axes[0]
                if (xValue < 0.3 && xValue > -0.3) {
                    return true;
                }
            }, delay);
            isWaiting = false;
        }
    }, 10);
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
