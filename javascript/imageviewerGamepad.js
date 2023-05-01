    let delay = 350//ms
    window.addEventListener('gamepadconnected', (e) => {
        console.log("Gamepad connected!")
        const gamepad = e.gamepad;
        setInterval(() => {
            const xValue = gamepad.axes[0].toFixed(2);
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
        if (isScrolling) return;
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