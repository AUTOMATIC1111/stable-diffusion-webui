onUiLoaded(async() => {
    const elementIDs = {
        img2imgTabs: "#mode_img2img .tab-nav",
        inpaint: "#img2maskimg",
        inpaintSketch: "#inpaint_sketch",
        rangeGroup: "#img2img_column_size",
        sketch: "#img2img_sketch",
    };
    const tabNameToElementId = {
        "Inpaint sketch": elementIDs.inpaintSketch,
        "Inpaint": elementIDs.inpaint,
        "Sketch": elementIDs.sketch,
    };

    // Helper functions
    // Get active tab
    function getActiveTab(elements, all = false) {
        const tabs = elements.img2imgTabs.querySelectorAll("button");

        if (all) return tabs;

        for (let tab of tabs) {
            if (tab.classList.contains("selected")) {
                return tab;
            }
        }
    }

    // Get tab ID
    function getTabId(elements) {
        const activeTab = getActiveTab(elements);
        return tabNameToElementId[activeTab.innerText];
    }

    // Wait until opts loaded
    async function waitForOpts() {
        for (;;) {
            if (window.opts && Object.keys(window.opts).length) {
                return window.opts;
            }
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }

    // Check is hotkey valid
    function isSingleLetter(value) {
        return (
            typeof value === "string" && value.length === 1 && /[a-z]/i.test(value)
        );
    }

    // Create hotkeyConfig from opts
    function createHotkeyConfig(defaultHotkeysConfig, hotkeysConfigOpts) {
        const result = {};
        const usedKeys = new Set();

        for (const key in defaultHotkeysConfig) {
            if (typeof hotkeysConfigOpts[key] === "boolean") {
                result[key] = hotkeysConfigOpts[key];
                continue;
            }
            if (
                hotkeysConfigOpts[key] &&
                isSingleLetter(hotkeysConfigOpts[key]) &&
                !usedKeys.has(hotkeysConfigOpts[key].toUpperCase())
            ) {
                // If the property passed the test and has not yet been used, add 'Key' before it and save it
                result[key] = "Key" + hotkeysConfigOpts[key].toUpperCase();
                usedKeys.add(hotkeysConfigOpts[key].toUpperCase());
            } else {
                // If the property does not pass the test or has already been used, we keep the default value
                console.error(
                    `Hotkey: ${hotkeysConfigOpts[key]} for ${key} is repeated and conflicts with another hotkey or is not 1 letter. The default hotkey is used: ${defaultHotkeysConfig[key][3]}`
                );
                result[key] = defaultHotkeysConfig[key];
            }
        }

        return result;
    }

    /**
     * The restoreImgRedMask function displays a red mask around an image to indicate the aspect ratio.
     * If the image display property is set to 'none', the mask breaks. To fix this, the function
     * temporarily sets the display property to 'block' and then hides the mask again after 300 milliseconds
     * to avoid breaking the canvas. Additionally, the function adjusts the mask to work correctly on
     * very long images.
     */
    function restoreImgRedMask(elements) {
        const mainTabId = getTabId(elements);

        if (!mainTabId) return;

        const mainTab = gradioApp().querySelector(mainTabId);
        const img = mainTab.querySelector("img");
        const imageARPreview = gradioApp().querySelector("#imageARPreview");

        if (!img || !imageARPreview) return;

        imageARPreview.style.transform = "";
        if (parseFloat(mainTab.style.width) > 865) {
            const transformString = mainTab.style.transform;
            const scaleMatch = transformString.match(/scale\(([-+]?[0-9]*\.?[0-9]+)\)/);
            let zoom = 1; // default zoom

            if (scaleMatch && scaleMatch[1]) {
                zoom = Number(scaleMatch[1]);
            }

            imageARPreview.style.transformOrigin = "0 0";
            imageARPreview.style.transform = `scale(${zoom})`;
        }

        if (img.style.display !== "none") return;

        img.style.display = "block";

        setTimeout(() => {
            img.style.display = "none";
        }, 400);
    }

    const hotkeysConfigOpts = await waitForOpts();

    // Default config
    const defaultHotkeysConfig = {
        canvas_hotkey_reset: "KeyR",
        canvas_hotkey_fullscreen: "KeyS",
        canvas_hotkey_move: "KeyF",
        canvas_hotkey_overlap: "KeyO",
        canvas_show_tooltip: true,
        canvas_swap_controls: false
    };
    // swap the actions for ctr + wheel and shift + wheel
    const hotkeysConfig = createHotkeyConfig(
        defaultHotkeysConfig,
        hotkeysConfigOpts
    );

    let isMoving = false;
    let mouseX, mouseY;
    let activeElement;

    const elements = Object.fromEntries(Object.keys(elementIDs).map((id) => [
        id,
        gradioApp().querySelector(elementIDs[id]),
    ]));
    const elemData = {};

    // Apply functionality to the range inputs. Restore redmask and correct for long images.
    const rangeInputs = elements.rangeGroup ? Array.from(elements.rangeGroup.querySelectorAll("input")) :
        [
            gradioApp().querySelector("#img2img_width input[type='range']"),
            gradioApp().querySelector("#img2img_height input[type='range']")
        ];

    for (const input of rangeInputs) {
        input?.addEventListener("input", () => restoreImgRedMask(elements));
    }

    function applyZoomAndPan(elemId) {
        const targetElement = gradioApp().querySelector(elemId);

        if (!targetElement) {
            console.log("Element not found");
            return;
        }

        targetElement.style.transformOrigin = "0 0";

        elemData[elemId] = {
            zoom: 1,
            panX: 0,
            panY: 0
        };
        let fullScreenMode = false;

        // Create tooltip
        function createTooltip() {
            const toolTipElemnt =
                targetElement.querySelector(".image-container");
            const tooltip = document.createElement("div");
            tooltip.className = "tooltip";

            // Creating an item of information
            const info = document.createElement("i");
            info.className = "tooltip-info";
            info.textContent = "";

            // Create a container for the contents of the tooltip
            const tooltipContent = document.createElement("div");
            tooltipContent.className = "tooltip-content";

            // Add info about hotkeys
            const zoomKey = hotkeysConfig.canvas_swap_controls ? "Ctrl" : "Shift";
            const adjustKey = hotkeysConfig.canvas_swap_controls ? "Shift" : "Ctrl";

            const hotkeys = [
                {key: `${zoomKey} + wheel`, action: "Zoom canvas"},
                {key: `${adjustKey} + wheel`, action: "Adjust brush size"},
                {
                    key: hotkeysConfig.canvas_hotkey_reset.charAt(hotkeysConfig.canvas_hotkey_reset.length - 1),
                    action: "Reset zoom"
                },
                {
                    key: hotkeysConfig.canvas_hotkey_fullscreen.charAt(hotkeysConfig.canvas_hotkey_fullscreen.length - 1),
                    action: "Fullscreen mode"
                },
                {
                    key: hotkeysConfig.canvas_hotkey_move.charAt(hotkeysConfig.canvas_hotkey_move.length - 1),
                    action: "Move canvas"
                }
            ];
            for (const hotkey of hotkeys) {
                const p = document.createElement("p");
                p.innerHTML = `<b>${hotkey.key}</b> - ${hotkey.action}`;
                tooltipContent.appendChild(p);
            }

            // Add information and content elements to the tooltip element
            tooltip.appendChild(info);
            tooltip.appendChild(tooltipContent);

            // Add a hint element to the target element
            toolTipElemnt.appendChild(tooltip);
        }

        //Show tool tip if setting enable
        if (hotkeysConfig.canvas_show_tooltip) {
            createTooltip();
        }

        // In the course of research, it was found that the tag img is very harmful when zooming and creates white canvases. This hack allows you to almost never think about this problem, it has no effect on webui.
        function fixCanvas() {
            const activeTab = getActiveTab(elements).textContent.trim();

            if (activeTab !== "img2img") {
                const img = targetElement.querySelector(`${elemId} img`);

                if (img && img.style.display !== "none") {
                    img.style.display = "none";
                    img.style.visibility = "hidden";
                }
            }
        }

        // Reset the zoom level and pan position of the target element to their initial values
        function resetZoom() {
            elemData[elemId] = {
                zoomLevel: 1,
                panX: 0,
                panY: 0
            };

            fixCanvas();
            targetElement.style.transform = `scale(${elemData[elemId].zoomLevel}) translate(${elemData[elemId].panX}px, ${elemData[elemId].panY}px)`;

            const canvas = gradioApp().querySelector(
                `${elemId} canvas[key="interface"]`
            );

            toggleOverlap("off");
            fullScreenMode = false;

            if (
                canvas &&
                parseFloat(canvas.style.width) > 865 &&
                parseFloat(targetElement.style.width) > 865
            ) {
                fitToElement();
                return;
            }

            targetElement.style.width = "";
            if (canvas) {
                targetElement.style.height = canvas.style.height;
            }
        }

        // Toggle the zIndex of the target element between two values, allowing it to overlap or be overlapped by other elements
        function toggleOverlap(forced = "") {
            const zIndex1 = "0";
            const zIndex2 = "998";

            targetElement.style.zIndex =
                targetElement.style.zIndex !== zIndex2 ? zIndex2 : zIndex1;

            if (forced === "off") {
                targetElement.style.zIndex = zIndex1;
            } else if (forced === "on") {
                targetElement.style.zIndex = zIndex2;
            }
        }

        // Adjust the brush size based on the deltaY value from a mouse wheel event
        function adjustBrushSize(
            elemId,
            deltaY,
            withoutValue = false,
            percentage = 5
        ) {
            const input =
                gradioApp().querySelector(
                    `${elemId} input[aria-label='Brush radius']`
                ) ||
                gradioApp().querySelector(
                    `${elemId} button[aria-label="Use brush"]`
                );

            if (input) {
                input.click();
                if (!withoutValue) {
                    const maxValue =
                        parseFloat(input.getAttribute("max")) || 100;
                    const changeAmount = maxValue * (percentage / 100);
                    const newValue =
                        parseFloat(input.value) +
                        (deltaY > 0 ? -changeAmount : changeAmount);
                    input.value = Math.min(Math.max(newValue, 0), maxValue);
                    input.dispatchEvent(new Event("change"));
                }
            }
        }

        // Reset zoom when uploading a new image
        const fileInput = gradioApp().querySelector(
            `${elemId} input[type="file"][accept="image/*"].svelte-116rqfv`
        );
        fileInput.addEventListener("click", resetZoom);

        // Update the zoom level and pan position of the target element based on the values of the zoomLevel, panX and panY variables
        function updateZoom(newZoomLevel, mouseX, mouseY) {
            newZoomLevel = Math.max(0.5, Math.min(newZoomLevel, 15));

            elemData[elemId].panX +=
                mouseX - (mouseX * newZoomLevel) / elemData[elemId].zoomLevel;
            elemData[elemId].panY +=
                mouseY - (mouseY * newZoomLevel) / elemData[elemId].zoomLevel;

            targetElement.style.transformOrigin = "0 0";
            targetElement.style.transform = `translate(${elemData[elemId].panX}px, ${elemData[elemId].panY}px) scale(${newZoomLevel})`;

            toggleOverlap("on");
            return newZoomLevel;
        }

        // Change the zoom level based on user interaction
        function changeZoomLevel(operation, e) {
            if (
                (!hotkeysConfig.canvas_swap_controls && e.shiftKey) ||
                (hotkeysConfig.canvas_swap_controls && e.ctrlKey)
            ) {
                e.preventDefault();

                let zoomPosX, zoomPosY;
                let delta = 0.2;
                if (elemData[elemId].zoomLevel > 7) {
                    delta = 0.9;
                } else if (elemData[elemId].zoomLevel > 2) {
                    delta = 0.6;
                }

                zoomPosX = e.clientX;
                zoomPosY = e.clientY;

                fullScreenMode = false;
                elemData[elemId].zoomLevel = updateZoom(
                    elemData[elemId].zoomLevel +
                        (operation === "+" ? delta : -delta),
                    zoomPosX - targetElement.getBoundingClientRect().left,
                    zoomPosY - targetElement.getBoundingClientRect().top
                );
            }
        }

        /**
         * This function fits the target element to the screen by calculating
         * the required scale and offsets. It also updates the global variables
         * zoomLevel, panX, and panY to reflect the new state.
         */

        function fitToElement() {
            //Reset Zoom
            targetElement.style.transform = `translate(${0}px, ${0}px) scale(${1})`;

            // Get element and screen dimensions
            const elementWidth = targetElement.offsetWidth;
            const elementHeight = targetElement.offsetHeight;
            const parentElement = targetElement.parentElement;
            const screenWidth = parentElement.clientWidth;
            const screenHeight = parentElement.clientHeight;

            // Get element's coordinates relative to the parent element
            const elementRect = targetElement.getBoundingClientRect();
            const parentRect = parentElement.getBoundingClientRect();
            const elementX = elementRect.x - parentRect.x;

            // Calculate scale and offsets
            const scaleX = screenWidth / elementWidth;
            const scaleY = screenHeight / elementHeight;
            const scale = Math.min(scaleX, scaleY);

            const transformOrigin =
                window.getComputedStyle(targetElement).transformOrigin;
            const [originX, originY] = transformOrigin.split(" ");
            const originXValue = parseFloat(originX);
            const originYValue = parseFloat(originY);

            const offsetX =
                (screenWidth - elementWidth * scale) / 2 -
                originXValue * (1 - scale);
            const offsetY =
                (screenHeight - elementHeight * scale) / 2.5 -
                originYValue * (1 - scale);

            // Apply scale and offsets to the element
            targetElement.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;

            // Update global variables
            elemData[elemId].zoomLevel = scale;
            elemData[elemId].panX = offsetX;
            elemData[elemId].panY = offsetY;

            fullScreenMode = false;
            toggleOverlap("off");
        }

        /**
         * This function fits the target element to the screen by calculating
         * the required scale and offsets. It also updates the global variables
         * zoomLevel, panX, and panY to reflect the new state.
         */

        // Fullscreen mode
        function fitToScreen() {
            const canvas = gradioApp().querySelector(
                `${elemId} canvas[key="interface"]`
            );

            if (!canvas) return;

            if (canvas.offsetWidth > 862) {
                targetElement.style.width = canvas.offsetWidth + "px";
            }

            if (fullScreenMode) {
                resetZoom();
                fullScreenMode = false;
                return;
            }

            //Reset Zoom
            targetElement.style.transform = `translate(${0}px, ${0}px) scale(${1})`;

            // Get scrollbar width to right-align the image
            const scrollbarWidth =
                window.innerWidth - document.documentElement.clientWidth;

            // Get element and screen dimensions
            const elementWidth = targetElement.offsetWidth;
            const elementHeight = targetElement.offsetHeight;
            const screenWidth = window.innerWidth - scrollbarWidth;
            const screenHeight = window.innerHeight;

            // Get element's coordinates relative to the page
            const elementRect = targetElement.getBoundingClientRect();
            const elementY = elementRect.y;
            const elementX = elementRect.x;

            // Calculate scale and offsets
            const scaleX = screenWidth / elementWidth;
            const scaleY = screenHeight / elementHeight;
            const scale = Math.min(scaleX, scaleY);

            // Get the current transformOrigin
            const computedStyle = window.getComputedStyle(targetElement);
            const transformOrigin = computedStyle.transformOrigin;
            const [originX, originY] = transformOrigin.split(" ");
            const originXValue = parseFloat(originX);
            const originYValue = parseFloat(originY);

            // Calculate offsets with respect to the transformOrigin
            const offsetX =
                (screenWidth - elementWidth * scale) / 2 -
                elementX -
                originXValue * (1 - scale);
            const offsetY =
                (screenHeight - elementHeight * scale) / 2 -
                elementY -
                originYValue * (1 - scale);

            // Apply scale and offsets to the element
            targetElement.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;

            // Update global variables
            elemData[elemId].zoomLevel = scale;
            elemData[elemId].panX = offsetX;
            elemData[elemId].panY = offsetY;

            fullScreenMode = true;
            toggleOverlap("on");
        }

        // Handle keydown events
        function handleKeyDown(event) {
            const hotkeyActions = {
                [hotkeysConfig.canvas_hotkey_reset]: resetZoom,
                [hotkeysConfig.canvas_hotkey_overlap]: toggleOverlap,
                [hotkeysConfig.canvas_hotkey_fullscreen]: fitToScreen
            };

            const action = hotkeyActions[event.code];
            if (action) {
                event.preventDefault();
                action(event);
            }
        }

        // Get Mouse position
        function getMousePosition(e) {
            mouseX = e.offsetX;
            mouseY = e.offsetY;
        }

        targetElement.addEventListener("mousemove", getMousePosition);

        // Handle events only inside the targetElement
        let isKeyDownHandlerAttached = false;

        function handleMouseMove() {
            if (!isKeyDownHandlerAttached) {
                document.addEventListener("keydown", handleKeyDown);
                isKeyDownHandlerAttached = true;

                activeElement = elemId;
            }
        }

        function handleMouseLeave() {
            if (isKeyDownHandlerAttached) {
                document.removeEventListener("keydown", handleKeyDown);
                isKeyDownHandlerAttached = false;

                activeElement = null;
            }
        }

        // Add mouse event handlers
        targetElement.addEventListener("mousemove", handleMouseMove);
        targetElement.addEventListener("mouseleave", handleMouseLeave);

        // Reset zoom when click on another tab
        elements.img2imgTabs.addEventListener("click", resetZoom);
        elements.img2imgTabs.addEventListener("click", () => {
            // targetElement.style.width = "";
            if (parseInt(targetElement.style.width) > 865) {
                setTimeout(fitToElement, 0);
            }
        });

        targetElement.addEventListener("wheel", e => {
            // change zoom level
            const operation = e.deltaY > 0 ? "-" : "+";
            changeZoomLevel(operation, e);

            // Handle brush size adjustment with ctrl key pressed
            if (
                (hotkeysConfig.canvas_swap_controls && e.shiftKey) ||
                (!hotkeysConfig.canvas_swap_controls &&
                    (e.ctrlKey || e.metaKey))
            ) {
                e.preventDefault();

                // Increase or decrease brush size based on scroll direction
                adjustBrushSize(elemId, e.deltaY);
            }
        });

        // Handle the move event for pan functionality. Updates the panX and panY variables and applies the new transform to the target element.
        function handleMoveKeyDown(e) {
            if (e.code === hotkeysConfig.canvas_hotkey_move) {
                if (!e.ctrlKey && !e.metaKey && isKeyDownHandlerAttached) {
                    e.preventDefault();
                    document.activeElement.blur();
                    isMoving = true;
                }
            }
        }

        function handleMoveKeyUp(e) {
            if (e.code === hotkeysConfig.canvas_hotkey_move) {
                isMoving = false;
            }
        }

        document.addEventListener("keydown", handleMoveKeyDown);
        document.addEventListener("keyup", handleMoveKeyUp);

        // Detect zoom level and update the pan speed.
        function updatePanPosition(movementX, movementY) {
            let panSpeed = 2;

            if (elemData[elemId].zoomLevel > 8) {
                panSpeed = 3.5;
            }

            elemData[elemId].panX += movementX * panSpeed;
            elemData[elemId].panY += movementY * panSpeed;

            // Delayed redraw of an element
            requestAnimationFrame(() => {
                targetElement.style.transform = `translate(${elemData[elemId].panX}px, ${elemData[elemId].panY}px) scale(${elemData[elemId].zoomLevel})`;
                toggleOverlap("on");
            });
        }

        function handleMoveByKey(e) {
            if (isMoving && elemId === activeElement) {
                updatePanPosition(e.movementX, e.movementY);
                targetElement.style.pointerEvents = "none";
            } else {
                targetElement.style.pointerEvents = "auto";
            }
        }

        // Prevents sticking to the mouse
        window.onblur = function() {
            isMoving = false;
        };

        gradioApp().addEventListener("mousemove", handleMoveByKey);
    }

    applyZoomAndPan(elementIDs.sketch);
    applyZoomAndPan(elementIDs.inpaint);
    applyZoomAndPan(elementIDs.inpaintSketch);

    // Make the function global so that other extensions can take advantage of this solution
    window.applyZoomAndPan = applyZoomAndPan;
});
