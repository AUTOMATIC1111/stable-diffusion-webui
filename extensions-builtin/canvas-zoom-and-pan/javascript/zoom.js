onUiLoaded(async() => {
    const elementIDs = {
        img2imgTabs: "#mode_img2img .tab-nav",
        inpaint: "#img2maskimg",
        inpaintSketch: "#inpaint_sketch",
        rangeGroup: "#img2img_column_size",
        sketch: "#img2img_sketch"
    };
    const tabNameToElementId = {
        "Inpaint sketch": elementIDs.inpaintSketch,
        "Inpaint": elementIDs.inpaint,
        "Sketch": elementIDs.sketch
    };


    // Helper functions
    // Get active tab

    function debounce(func, wait) {
        let timeout;

        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };

            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Waits for an element to be present in the DOM.
     */
    const waitForElement = (id) => new Promise(resolve => {
        const checkForElement = () => {
            const element = document.querySelector(id);
            if (element) return resolve(element);
            setTimeout(checkForElement, 100);
        };
        checkForElement();
    });

    function getActiveTab(elements, all = false) {
        if (!elements.img2imgTabs) return null;
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
        if (!activeTab) return null;
        return tabNameToElementId[activeTab.innerText];
    }

    // Wait until opts loaded
    async function waitForOpts() {
        for (; ;) {
            if (window.opts && Object.keys(window.opts).length) {
                return window.opts;
            }
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }

    // // Hack to make the cursor always be the same size
    function fixCursorSize() {
        window.scrollBy(0, 1);
    }

    function copySpecificStyles(sourceElement, targetElement, zoomLevel = 1) {
        const stylesToCopy = ['top', 'left', 'width', 'height'];

        stylesToCopy.forEach(styleName => {
            if (sourceElement.style[styleName]) {
                // Convert style value to number and multiply by zoomLevel.
                let adjustedStyleValue = parseFloat(sourceElement.style[styleName]) / zoomLevel;

                // Set the adjusted style value back to target element's style.
                // Important: this will work fine for top and left styles as they are usually in px.
                // But be careful with other units like em or % that might need different handling.
                targetElement.style[styleName] = `${adjustedStyleValue}px`;
            }
        });

        targetElement.style["opacity"] = sourceElement.style["opacity"];
    }


    // Detect whether the element has a horizontal scroll bar
    function hasHorizontalScrollbar(element) {
        return element.scrollWidth > element.clientWidth;
    }

    // Function for defining the "Ctrl", "Shift" and "Alt" keys
    function isModifierKey(event, key) {
        switch (key) {
        case "Ctrl":
            return event.ctrlKey;
        case "Shift":
            return event.shiftKey;
        case "Alt":
            return event.altKey;
        default:
            return false;
        }
    }

    // Check if hotkey is valid
    function isValidHotkey(value) {
        const specialKeys = ["Ctrl", "Alt", "Shift", "Disable"];
        return (
            (typeof value === "string" &&
                value.length === 1 &&
                /[a-z]/i.test(value)) ||
            specialKeys.includes(value)
        );
    }

    // Normalize hotkey
    function normalizeHotkey(hotkey) {
        return hotkey.length === 1 ? "Key" + hotkey.toUpperCase() : hotkey;
    }

    // Format hotkey for display
    function formatHotkeyForDisplay(hotkey) {
        return hotkey.startsWith("Key") ? hotkey.slice(3) : hotkey;
    }

    // Create hotkey configuration with the provided options
    function createHotkeyConfig(defaultHotkeysConfig, hotkeysConfigOpts) {
        const result = {}; // Resulting hotkey configuration
        const usedKeys = new Set(); // Set of used hotkeys

        // Iterate through defaultHotkeysConfig keys
        for (const key in defaultHotkeysConfig) {
            const userValue = hotkeysConfigOpts[key]; // User-provided hotkey value
            const defaultValue = defaultHotkeysConfig[key]; // Default hotkey value

            // Apply appropriate value for undefined, boolean, or object userValue
            if (
                userValue === undefined ||
                typeof userValue === "boolean" ||
                typeof userValue === "object" ||
                userValue === "disable"
            ) {
                result[key] =
                    userValue === undefined ? defaultValue : userValue;
            } else if (isValidHotkey(userValue)) {
                const normalizedUserValue = normalizeHotkey(userValue);

                // Check for conflicting hotkeys
                if (!usedKeys.has(normalizedUserValue)) {
                    usedKeys.add(normalizedUserValue);
                    result[key] = normalizedUserValue;
                } else {
                    console.error(
                        `Hotkey: ${formatHotkeyForDisplay(
                            userValue
                        )} for ${key} is repeated and conflicts with another hotkey. The default hotkey is used: ${formatHotkeyForDisplay(
                            defaultValue
                        )}`
                    );
                    result[key] = defaultValue;
                }
            } else {
                console.error(
                    `Hotkey: ${formatHotkeyForDisplay(
                        userValue
                    )} for ${key} is not valid. The default hotkey is used: ${formatHotkeyForDisplay(
                        defaultValue
                    )}`
                );
                result[key] = defaultValue;
            }
        }

        return result;
    }

    // Disables functions in the config object based on the provided list of function names
    function disableFunctions(config, disabledFunctions) {
        // Bind the hasOwnProperty method to the functionMap object to avoid errors
        const hasOwnProperty =
            Object.prototype.hasOwnProperty.bind(functionMap);

        // Loop through the disabledFunctions array and disable the corresponding functions in the config object
        disabledFunctions.forEach(funcName => {
            if (hasOwnProperty(funcName)) {
                const key = functionMap[funcName];
                config[key] = "disable";
            }
        });

        // Return the updated config object
        return config;
    }


    const hotkeysConfigOpts = await waitForOpts();

    // Default config
    const defaultHotkeysConfig = {
        canvas_hotkey_zoom: "Alt",
        canvas_hotkey_adjust: "Ctrl",
        canvas_hotkey_reset: "KeyR",
        canvas_hotkey_fullscreen: "KeyS",
        canvas_hotkey_move: "KeyF",
        canvas_hotkey_overlap: "KeyO",
        canvas_hotkey_shrink_brush: "KeyQ",
        canvas_hotkey_grow_brush: "KeyW",
        canvas_disabled_functions: [],
        canvas_show_tooltip: true,
        canvas_blur_prompt: false,
    };

    const functionMap = {
        "Zoom": "canvas_hotkey_zoom",
        "Adjust brush size": "canvas_hotkey_adjust",
        "Hotkey shrink brush": "canvas_hotkey_shrink_brush",
        "Hotkey enlarge brush": "canvas_hotkey_grow_brush",
        "Moving canvas": "canvas_hotkey_move",
        "Fullscreen": "canvas_hotkey_fullscreen",
        "Reset Zoom": "canvas_hotkey_reset",
        "Overlap": "canvas_hotkey_overlap"
    };

    // Loading the configuration from opts
    const preHotkeysConfig = createHotkeyConfig(
        defaultHotkeysConfig,
        hotkeysConfigOpts
    );

    // Disable functions that are not needed by the user
    const hotkeysConfig = disableFunctions(
        preHotkeysConfig,
        preHotkeysConfig.canvas_disabled_functions
    );

    let isMoving = false;
    let mouseX, mouseY;
    let activeElement;
    let interactedWithAltKey = false;

    const elements = Object.fromEntries(
        Object.keys(elementIDs).map(id => [
            id,
            gradioApp().querySelector(elementIDs[id])
        ])
    );
    const elemData = {};

    function applyZoomAndPan(elemId, isExtension = true) {
        const targetElement = gradioApp().querySelector(elemId);

        if (!targetElement) {
            console.log("Element not found", elemId);
            return;
        }

        targetElement.style.transformOrigin = "0 0";

        elemData[elemId] = {
            zoom: 1,
            panX: 0,
            panY: 0,
        };

        let fullScreenMode = false;


        // Cursor manipulation script for a painting application.
        // The purpose of this code is to create custom cursors (for painting and erasing)
        // that can change depending on which button the user presses.
        // When the mouse moves over the canvas, the appropriate custom cursor also moves,
        // replicating its appearance dynamically based on various CSS properties.

        // This is done because the original cursor is tied to the size of the kanvas, it can not be changed, so I came up with a hack that creates an exact copy that works properly

        const eraseButton = targetElement.querySelector(`button[aria-label='Erase button']`);
        const paintButton = targetElement.querySelector(`button[aria-label='Draw button']`);

        const canvasCursors = targetElement.querySelectorAll("span.svelte-btgkrd");
        const paintCursorCopy = canvasCursors[0].cloneNode(true);
        const eraserCursorCopy = canvasCursors[1].cloneNode(true);

        canvasCursors.forEach(cursor => cursor.style.display = "none");

        canvasCursors[0].parentNode.insertBefore(paintCursorCopy, canvasCursors[0].nextSibling);
        canvasCursors[1].parentNode.insertBefore(eraserCursorCopy, canvasCursors[1].nextSibling);


        // targetElement.appendChild(paintCursorCopy);
        // paintCursorCopy.style.display = "none";

        // targetElement.appendChild(eraserCursorCopy);
        // eraserCursorCopy.style.display = "none";

        let activeCursor;

        paintButton.addEventListener('click', () => {
            activateTool(paintButton, eraseButton, paintCursorCopy);
        });

        eraseButton.addEventListener('click', () => {
            activateTool(eraseButton, paintButton, eraserCursorCopy);
        });

        function activateTool(activeButton, inactiveButton, activeCursorCopy) {
            activeButton.classList.add("active");
            inactiveButton.classList.remove("active");

            // canvasCursors.forEach(cursor => cursor.style.display = "none");

            if (activeCursor) {
                activeCursor.style.display = "none";
            }

            activeCursor = activeCursorCopy;
            // activeCursor.style.display = "none";
            activeCursor.style.position = "absolute";
        }

        const canvasAreaEventsHandler = e => {

            canvasCursors.forEach(cursor => cursor.style.display = "none");

            if (!activeCursor) return;

            const cursorNum = eraseButton.classList.contains("active") ? 1 : 0;

            if (elemData[elemId].zoomLevel != 1) {
                copySpecificStyles(canvasCursors[cursorNum], activeCursor, elemData[elemId].zoomLevel);
            } else {
                // Update the styles of the currently active cursor
                copySpecificStyles(canvasCursors[cursorNum], activeCursor);
            }

            let offsetXAdjusted = e.offsetX;
            let offsetYAdjusted = e.offsetY;

            // Position the cursor based on the current mouse coordinates within target element.
            activeCursor.style.transform =
           `translate(${offsetXAdjusted}px, ${offsetYAdjusted}px)`;
        };

        const canvasAreaLeaveHandler = () => {
            if (activeCursor) {
                // activeCursor.style.opacity = 0
                activeCursor.style.display = "none";
            }
        };

        const canvasAreaEnterHandler = () => {
            if (activeCursor) {
                // activeCursor.style.opacity = 1
                activeCursor.style.display = "block";
            }
        };

        const canvasArea = targetElement.querySelector("canvas");

        // Attach event listeners to the target element and canvas area
        targetElement.addEventListener("mousemove", canvasAreaEventsHandler);
        canvasArea.addEventListener("mouseout", canvasAreaLeaveHandler);
        canvasArea.addEventListener("mouseenter", canvasAreaEnterHandler);

        // Additional listener for handling zoom or other transformations which might affect visual representation
        targetElement.addEventListener("wheel", canvasAreaEventsHandler);

        // Remove border, cause bags
        const canvasBorder = targetElement.querySelector(".border");
        canvasBorder.style.display = "none";

        // Create tooltip
        function createTooltip() {
            const toolTipElement = targetElement.querySelector(".image-container");
            const tooltip = document.createElement("div");
            tooltip.className = "canvas-tooltip";

            // Creating an item of information
            const info = document.createElement("i");
            info.className = "canvas-tooltip-info";
            info.textContent = "";

            // Create a container for the contents of the tooltip
            const tooltipContent = document.createElement("div");
            tooltipContent.className = "canvas-tooltip-content";

            // Define an array with hotkey information and their actions
            const hotkeysInfo = [
                {
                    configKey: "canvas_hotkey_zoom",
                    action: "Zoom canvas",
                    keySuffix: " + wheel"
                },
                {
                    configKey: "canvas_hotkey_adjust",
                    action: "Adjust brush size",
                    keySuffix: " + wheel"
                },
                {configKey: "canvas_hotkey_reset", action: "Reset zoom"},
                {
                    configKey: "canvas_hotkey_fullscreen",
                    action: "Fullscreen mode"
                },
                {configKey: "canvas_hotkey_move", action: "Move canvas"},
                {configKey: "canvas_hotkey_overlap", action: "Overlap"}
            ];

            // Create hotkeys array with disabled property based on the config values
            const hotkeys = hotkeysInfo.map(info => {
                const configValue = hotkeysConfig[info.configKey];
                const key = info.keySuffix ?
                    `${configValue}${info.keySuffix}` :
                    configValue.charAt(configValue.length - 1);
                return {
                    key,
                    action: info.action,
                    disabled: configValue === "disable"
                };
            });

            for (const hotkey of hotkeys) {
                if (hotkey.disabled) {
                    continue;
                }

                const p = document.createElement("p");
                p.innerHTML = `<b>${hotkey.key}</b> - ${hotkey.action}`;
                tooltipContent.appendChild(p);
            }

            // Add information and content elements to the tooltip element
            tooltip.appendChild(info);
            tooltip.appendChild(tooltipContent);

            // Add a hint element to the target element
            toolTipElement.appendChild(tooltip);

            return tooltip;
        }

        //Show tool tip if setting enable
        const canvasTooltip = createTooltip();

        if (!hotkeysConfig.canvas_show_tooltip) {
            canvasTooltip.style.display = "none";
        }

        // Reset the zoom level and pan position of the target element to their initial values
        function resetZoom() {
            elemData[elemId] = {
                zoomLevel: 1,
                panX: 0,
                panY: 0,
            };

            if (isExtension) {
                targetElement.style.overflow = "hidden";
            }

            targetElement.isZoomed = false;

            targetElement.style.transform = `scale(${elemData[elemId].zoomLevel}) translate(${elemData[elemId].panX}px, ${elemData[elemId].panY}px)`;

            const canvas = gradioApp().querySelector(
                `${elemId} canvas`
            );

            toggleOverlap("off");
            fullScreenMode = false;

            const closeBtn = targetElement.querySelector("button[aria-label='Clear canvas']");
            if (closeBtn) {
                closeBtn.addEventListener("click", resetZoom);
            }

            targetElement.style.width = "";
            fixCursorSize();
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
                    `${elemId} input[type='range']`
                ) ||
                gradioApp().querySelector(
                    `${elemId} button[aria-label="Size button"]`
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
            `${elemId} .upload-container input[type="file"][accept="image/*"]`
        );

        fileInput.addEventListener("click", resetZoom);

        // Create clickble area
        const inputCanvas = targetElement.querySelector("canvas");


        // Update the zoom level and pan position of the target element based on the values of the zoomLevel, panX and panY variables
        function updateZoom(newZoomLevel, mouseX, mouseY) {
            newZoomLevel = Math.max(0.1, Math.min(newZoomLevel, 15));

            elemData[elemId].panX +=
                mouseX - (mouseX * newZoomLevel) / elemData[elemId].zoomLevel;
            elemData[elemId].panY +=
                mouseY - (mouseY * newZoomLevel) / elemData[elemId].zoomLevel;

            targetElement.style.transformOrigin = "0 0";
            targetElement.style.transform = `translate(${elemData[elemId].panX}px, ${elemData[elemId].panY}px) scale(${newZoomLevel})`;

            toggleOverlap("on");
            if (isExtension) {
                targetElement.style.overflow = "visible";
            }

            // Hack to make the cursor always be the same size
            fixCursorSize();

            return newZoomLevel;
        }

        // Change the zoom level based on user interaction
        function changeZoomLevel(operation, e) {
            if (isModifierKey(e, hotkeysConfig.canvas_hotkey_zoom)) {
                e.preventDefault();

                if (hotkeysConfig.canvas_hotkey_zoom === "Alt") {
                    interactedWithAltKey = true;
                }

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

                targetElement.isZoomed = true;
            }
        }

        /**
         * This function fits the target element to the screen by calculating
         * the required scale and offsets. It also updates the global variables
         * zoomLevel, panX, and panY to reflect the new state.
         */

        // Fullscreen mode
        function fitToScreen() {
            const canvas = gradioApp().querySelector(
                `${elemId} canvas`
            );

            // print(canvas)

            if (!canvas) return;

            if (canvas.offsetWidth > 862 || isExtension) {
                targetElement.style.width = (canvas.offsetWidth + 2) + "px";
            }

            if (isExtension) {
                targetElement.style.overflow = "visible";
            }

            fixCursorSize();
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
            // Disable key locks to make pasting from the buffer work correctly
            if ((event.ctrlKey && event.code === 'KeyV') || (event.ctrlKey && event.code === 'KeyC') || event.code === "F5") {
                return;
            }

            // before activating shortcut, ensure user is not actively typing in an input field
            if (!hotkeysConfig.canvas_blur_prompt) {
                if (event.target.nodeName === 'TEXTAREA' || event.target.nodeName === 'INPUT') {
                    return;
                }
            }


            const hotkeyActions = {
                [hotkeysConfig.canvas_hotkey_reset]: resetZoom,
                [hotkeysConfig.canvas_hotkey_overlap]: toggleOverlap,
                [hotkeysConfig.canvas_hotkey_fullscreen]: fitToScreen,
                [hotkeysConfig.canvas_hotkey_shrink_brush]: () => adjustBrushSize(elemId, 10),
                [hotkeysConfig.canvas_hotkey_grow_brush]: () => adjustBrushSize(elemId, -10)
            };

            const action = hotkeyActions[event.code];
            if (action) {
                event.preventDefault();
                action(event);
            }

            if (
                isModifierKey(event, hotkeysConfig.canvas_hotkey_zoom) ||
                isModifierKey(event, hotkeysConfig.canvas_hotkey_adjust)
            ) {
                event.preventDefault();
            }
        }

        // Get Mouse position
        function getMousePosition(e) {
            mouseX = e.offsetX;
            mouseY = e.offsetY;
        }

        // Simulation of the function to put a long image into the screen.
        // We detect if an image has a scroll bar or not, make a fullscreen to reveal the image, then reduce it to fit into the element.
        // We hide the image and show it to the user when it is ready.

        targetElement.isExpanded = false;
        function autoExpand() {
            const canvas = document.querySelector(`${elemId} canvas`);
            if (canvas) {
                if (hasHorizontalScrollbar(targetElement) && targetElement.isExpanded === false) {
                    targetElement.style.visibility = "hidden";
                    setTimeout(() => {
                        fitToScreen();
                        resetZoom();
                        targetElement.style.visibility = "visible";
                        targetElement.isExpanded = true;
                    }, 10);
                }
            }
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

        targetElement.addEventListener("wheel", e => {
            // change zoom level
            const operation = (e.deltaY || -e.wheelDelta) > 0 ? "-" : "+";
            changeZoomLevel(operation, e);

            // Handle brush size adjustment with ctrl key pressed
            if (isModifierKey(e, hotkeysConfig.canvas_hotkey_adjust)) {
                e.preventDefault();

                if (hotkeysConfig.canvas_hotkey_adjust === "Alt") {
                    interactedWithAltKey = true;
                }

                // Increase or decrease brush size based on scroll direction
                adjustBrushSize(elemId, e.deltaY);
            }
        });

        // Handle the move event for pan functionality. Updates the panX and panY variables and applies the new transform to the target element.
        function handleMoveKeyDown(e) {

            // Disable key locks to make pasting from the buffer work correctly
            if ((e.ctrlKey && e.code === 'KeyV') || (e.ctrlKey && event.code === 'KeyC') || e.code === "F5") {
                return;
            }

            // before activating shortcut, ensure user is not actively typing in an input field
            if (!hotkeysConfig.canvas_blur_prompt) {
                if (e.target.nodeName === 'TEXTAREA' || e.target.nodeName === 'INPUT') {
                    return;
                }
            }


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


        // Prevent firefox from opening main menu when alt is used as a hotkey for zoom or brush size
        function handleAltKeyUp(e) {
            if (e.key !== "Alt" || !interactedWithAltKey) {
                return;
            }

            e.preventDefault();
            interactedWithAltKey = false;
        }

        document.addEventListener("keyup", handleAltKeyUp);


        // Detect zoom level and update the pan speed.
        function updatePanPosition(movementX, movementY) {
            let panSpeed = 2;

            if (elemData[elemId].zoomLevel > 8) {
                panSpeed = 3.5;
            }

            elemData[elemId].panX += movementX * panSpeed;
            elemData[elemId].panY += movementY * panSpeed;

            // Delayed redraw of an element
            const canvas = targetElement.querySelector("canvas");
            requestAnimationFrame(() => {
                targetElement.style.transform = `translate(${elemData[elemId].panX}px, ${elemData[elemId].panY}px) scale(${elemData[elemId].zoomLevel})`;
                toggleOverlap("on");
            });
        }

        function handleMoveByKey(e) {
            if (isMoving && elemId === activeElement) {
                updatePanPosition(e.movementX, e.movementY);
                targetElement.style.pointerEvents = "none";

                if (isExtension) {
                    targetElement.style.overflow = "visible";
                }

            } else {
                targetElement.style.pointerEvents = "auto";
            }
        }

        // Prevents sticking to the mouse
        window.onblur = function() {
            isMoving = false;
        };

        // Checks for extension
        function checkForOutBox() {
            const parentElement = targetElement.closest('[id^="component-"]');
            if (parentElement.offsetWidth < targetElement.offsetWidth && !targetElement.isExpanded) {
                resetZoom();
                targetElement.isExpanded = true;
            }

            if (parentElement.offsetWidth < targetElement.offsetWidth && elemData[elemId].zoomLevel == 1) {
                resetZoom();
            }

            if (parentElement.offsetWidth < targetElement.offsetWidth && targetElement.offsetWidth * elemData[elemId].zoomLevel > parentElement.offsetWidth && elemData[elemId].zoomLevel < 1 && !targetElement.isZoomed) {
                resetZoom();
            }
        }

        if (isExtension) {
            targetElement.addEventListener("mousemove", checkForOutBox);
        }


        window.addEventListener('resize', (e) => {
            resetZoom();

            if (isExtension) {
                targetElement.isExpanded = false;
                targetElement.isZoomed = false;
            }
        });

        gradioApp().addEventListener("mousemove", handleMoveByKey);

    }

    applyZoomAndPan(elementIDs.sketch, false);
    applyZoomAndPan(elementIDs.inpaint, false);
    applyZoomAndPan(elementIDs.inpaintSketch, false);

    // Make the function global so that other extensions can take advantage of this solution
    const applyZoomAndPanIntegration = async(id, elementIDs) => {
        const mainEl = document.querySelector(id);
        if (id.toLocaleLowerCase() === "none") {
            for (const elementID of elementIDs) {
                const el = await waitForElement(elementID);
                if (!el) break;
                applyZoomAndPan(elementID);
            }
            return;
        }

        if (!mainEl) return;
        mainEl.addEventListener("click", async() => {
            for (const elementID of elementIDs) {
                const el = await waitForElement(elementID);
                if (!el) break;
                applyZoomAndPan(elementID);
            }
        }, {once: true});
    };

    window.applyZoomAndPan = applyZoomAndPan; // Only 1 elements, argument elementID, for example applyZoomAndPan("#txt2img_controlnet_ControlNet_input_image")
    window.applyZoomAndPanIntegration = applyZoomAndPanIntegration; // for any extension


    // Return zoom functionality when send img via buttons
    const img2imgArea = document.querySelector("#img2img_settings");
    const checkForTooltip = (e) => {
        const tabId = getTabId(elements); // Make sure that the item is passed correctly to determine the tabId

        if (tabId === "#img2img_sketch" || tabId === "#inpaint_sketch" || tabId === "#img2maskimg") {
            const zoomTooltip = document.querySelector(`${tabId} .canvas-tooltip`);

            if (!zoomTooltip) {
                applyZoomAndPan(tabId, false);
                // resetZoom()
            }
        }
    };

    // Wrapping your function through debounce to reduce the number of calls
    const debouncedCheckForTooltip = debounce(checkForTooltip, 20);

    // Assigning an event handler
    img2imgArea.addEventListener("mousemove", debouncedCheckForTooltip);

    /*
        The function `applyZoomAndPanIntegration` takes two arguments:

        1. `id`: A string identifier for the element to which zoom and pan functionality will be applied on click.
        If the `id` value is "none", the functionality will be applied to all elements specified in the second argument without a click event.

        2. `elementIDs`: An array of string identifiers for elements. Zoom and pan functionality will be applied to each of these elements on click of the element specified by the first argument.
        If "none" is specified in the first argument, the functionality will be applied to each of these elements without a click event.

        Example usage:
        applyZoomAndPanIntegration("#txt2img_controlnet", ["#txt2img_controlnet_ControlNet_input_image"]);
        In this example, zoom and pan functionality will be applied to the element with the identifier "txt2img_controlnet_ControlNet_input_image" upon clicking the element with the identifier "txt2img_controlnet".
    */

    // More examples
    // Add integration with ControlNet txt2img One TAB
    // applyZoomAndPanIntegration("#txt2img_controlnet", ["#txt2img_controlnet_ControlNet_input_image"]);

    // Add integration with ControlNet txt2img Tabs
    // applyZoomAndPanIntegration("#txt2img_controlnet",Array.from({ length: 10 }, (_, i) => `#txt2img_controlnet_ControlNet-${i}_input_image`));

    // Add integration with Inpaint Anything
    // applyZoomAndPanIntegration("None", ["#ia_sam_image", "#ia_sel_mask"]);
});
