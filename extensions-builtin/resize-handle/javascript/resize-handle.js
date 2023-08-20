onUiLoaded(async() => {
    const GRADIO_MIN_WIDTH = 320;
    const GRID_TEMPLATE_COLUMNS = '1fr 16px 1fr';
    const PAD = 16;
    const DEBOUNCE_TIME = 100;

    const R = {
        tracking: false,
        parent: null,
        parentWidth: null,
        leftCol: null,
        leftColStartWidth: null,
        screenX: null,
    };

    let resizeTimer;

    const leftCols = [
        gradioApp().querySelector('#txt2img_settings'),
        gradioApp().querySelector('#img2img_settings'),
    ];

    function setLeftColGridTemplate(el, width) {
        el.style.gridTemplateColumns = `${width}px 16px 1fr`;
    }

    function displayResizeHandle(parent) {
        if (window.innerWidth < GRADIO_MIN_WIDTH * 2 + PAD * 4) {
            parent.style.display = 'flex';
            if (R.handle != null) {
                R.handle.style.opacity = '0';
            }
            return false;
        } else {
            parent.style.display = 'grid';
            if (R.handle != null) {
                R.handle.style.opacity = '100';
            }
            return true;
        }
    }

    function setup() {
        for (const leftCol of leftCols) {
            const parent = leftCol.parentElement;
            const rightCol = parent.lastElementChild;

            if (!displayResizeHandle(parent)) {
                return;
            }

            parent.style.display = 'grid';
            parent.style.gap = '0';
            parent.style.gridTemplateColumns = GRID_TEMPLATE_COLUMNS;

            const resizeHandle = document.createElement('div');
            resizeHandle.classList.add('resize-handle');
            parent.insertBefore(resizeHandle, rightCol);

            resizeHandle.addEventListener('mousedown', (evt) => {
                R.tracking = true;
                R.parent = parent;
                R.parentWidth = parent.offsetWidth;
                R.handle = resizeHandle;
                R.leftCol = leftCol;
                R.leftColStartWidth = leftCol.offsetWidth;
                R.screenX = evt.screenX;
            });
        }
    }

    window.addEventListener('mousemove', (evt) => {
        if (R.tracking) {
            const delta = R.screenX - evt.screenX;
            const leftColWidth = Math.max(Math.min(R.leftColStartWidth - delta, R.parent.offsetWidth - GRADIO_MIN_WIDTH - PAD), GRADIO_MIN_WIDTH);
            setLeftColGridTemplate(R.parent, leftColWidth);
        }
    });

    window.addEventListener('mouseup', () => R.tracking = false);

    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);

        resizeTimer = setTimeout(() => {
            for (const leftCol of leftCols) {
                const parent = leftCol.parentElement;

                if (displayResizeHandle(parent) && parent.style.gridTemplateColumns != GRID_TEMPLATE_COLUMNS) {
                    const oldParentWidth = R.parentWidth;
                    const newParentWidth = parent.offsetWidth;
                    const widthL = parseInt(parent.style.gridTemplateColumns.split(' ')[0]);

                    const ratio = newParentWidth / oldParentWidth;

                    const newWidthL = Math.max(Math.floor(ratio * widthL), GRADIO_MIN_WIDTH);
                    setLeftColGridTemplate(parent, newWidthL);

                    R.parentWidth = newParentWidth;
                }
            }
        }, DEBOUNCE_TIME);
    });

    setup();
});
