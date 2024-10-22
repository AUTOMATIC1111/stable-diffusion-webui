(function() {
    const GRADIO_MIN_WIDTH = 320;
    const PAD = 16;
    const DEBOUNCE_TIME = 100;
    const DOUBLE_TAP_DELAY = 200; //ms

    const R = {
        tracking: false,
        parent: null,
        parentWidth: null,
        leftCol: null,
        leftColStartWidth: null,
        screenX: null,
        lastTapTime: null,
    };

    let resizeTimer;
    let parents = [];

    function setLeftColGridTemplate(el, width) {
        el.style.gridTemplateColumns = `${width}px 16px 1fr`;
    }

    function displayResizeHandle(parent) {
        if (!parent.needHideOnMoblie) {
            return true;
        }
        if (window.innerWidth < GRADIO_MIN_WIDTH * 2 + PAD * 4) {
            parent.style.display = 'flex';
            parent.resizeHandle.style.display = "none";
            return false;
        } else {
            parent.style.display = 'grid';
            parent.resizeHandle.style.display = "block";
            return true;
        }
    }

    function afterResize(parent) {
        if (displayResizeHandle(parent) && parent.style.gridTemplateColumns != parent.style.originalGridTemplateColumns) {
            const oldParentWidth = R.parentWidth;
            const newParentWidth = parent.offsetWidth;
            const widthL = parseInt(parent.style.gridTemplateColumns.split(' ')[0]);

            const ratio = newParentWidth / oldParentWidth;

            const newWidthL = Math.max(Math.floor(ratio * widthL), parent.minLeftColWidth);
            setLeftColGridTemplate(parent, newWidthL);

            R.parentWidth = newParentWidth;
        }
    }

    function setup(parent) {

        function onDoubleClick(evt) {
            evt.preventDefault();
            evt.stopPropagation();

            parent.style.gridTemplateColumns = parent.style.originalGridTemplateColumns;
        }

        const leftCol = parent.firstElementChild;
        const rightCol = parent.lastElementChild;

        parents.push(parent);

        parent.style.display = 'grid';
        parent.style.gap = '0';
        let leftColTemplate = "";
        if (parent.children[0].style.flexGrow) {
            leftColTemplate = `${parent.children[0].style.flexGrow}fr`;
            parent.minLeftColWidth = GRADIO_MIN_WIDTH;
            parent.minRightColWidth = GRADIO_MIN_WIDTH;
            parent.needHideOnMoblie = true;
        } else {
            leftColTemplate = parent.children[0].style.flexBasis;
            parent.minLeftColWidth = parent.children[0].style.flexBasis.slice(0, -2) / 2;
            parent.minRightColWidth = 0;
            parent.needHideOnMoblie = false;
        }

        if (!leftColTemplate) {
            leftColTemplate = '1fr';
        }

        const gridTemplateColumns = `${leftColTemplate} ${PAD}px ${parent.children[1].style.flexGrow}fr`;
        parent.style.gridTemplateColumns = gridTemplateColumns;
        parent.style.originalGridTemplateColumns = gridTemplateColumns;

        const resizeHandle = document.createElement('div');
        resizeHandle.classList.add('resize-handle');
        parent.insertBefore(resizeHandle, rightCol);
        parent.resizeHandle = resizeHandle;

        ['mousedown', 'touchstart'].forEach((eventType) => {
            resizeHandle.addEventListener(eventType, (evt) => {
                if (eventType.startsWith('mouse')) {
                    if (evt.button !== 0) return;
                } else {
                    if (evt.changedTouches.length !== 1) return;

                    const currentTime = new Date().getTime();
                    if (R.lastTapTime && currentTime - R.lastTapTime <= DOUBLE_TAP_DELAY) {
                        onDoubleClick(evt);
                        return;
                    }

                    R.lastTapTime = currentTime;
                }

                evt.preventDefault();
                evt.stopPropagation();

                document.body.classList.add('resizing');

                R.tracking = true;
                R.parent = parent;
                R.parentWidth = parent.offsetWidth;
                R.leftCol = leftCol;
                R.leftColStartWidth = leftCol.offsetWidth;
                if (eventType.startsWith('mouse')) {
                    R.screenX = evt.screenX;
                } else {
                    R.screenX = evt.changedTouches[0].screenX;
                }
            }, {passive: false});
        });

        resizeHandle.addEventListener('dblclick', onDoubleClick);

        afterResize(parent);
    }

    ['mousemove', 'touchmove'].forEach((eventType) => {
        window.addEventListener(eventType, (evt) => {
            if (eventType.startsWith('mouse')) {
                if (evt.button !== 0) return;
            } else {
                if (evt.changedTouches.length !== 1) return;
            }

            if (R.tracking) {
                if (eventType.startsWith('mouse')) {
                    evt.preventDefault();
                }
                evt.stopPropagation();

                let delta = 0;
                if (eventType.startsWith('mouse')) {
                    delta = R.screenX - evt.screenX;
                } else {
                    delta = R.screenX - evt.changedTouches[0].screenX;
                }
                const leftColWidth = Math.max(Math.min(R.leftColStartWidth - delta, R.parent.offsetWidth - R.parent.minRightColWidth - PAD), R.parent.minLeftColWidth);
                setLeftColGridTemplate(R.parent, leftColWidth);
            }
        });
    });

    ['mouseup', 'touchend'].forEach((eventType) => {
        window.addEventListener(eventType, (evt) => {
            if (eventType.startsWith('mouse')) {
                if (evt.button !== 0) return;
            } else {
                if (evt.changedTouches.length !== 1) return;
            }

            if (R.tracking) {
                evt.preventDefault();
                evt.stopPropagation();

                R.tracking = false;

                document.body.classList.remove('resizing');
            }
        });
    });


    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);

        resizeTimer = setTimeout(function() {
            for (const parent of parents) {
                afterResize(parent);
            }
        }, DEBOUNCE_TIME);
    });

    setupResizeHandle = setup;
})();


function setupAllResizeHandles() {
    for (var elem of gradioApp().querySelectorAll('.resize-handle-row')) {
        if (!elem.querySelector('.resize-handle') && !elem.children[0].classList.contains("hidden")) {
            setupResizeHandle(elem);
        }
    }
}


onUiLoaded(setupAllResizeHandles);

