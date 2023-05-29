// Main

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

onUiLoaded(async() => {
    const hotkeysConfig = {
        resetZoom: "KeyR",
        fitToScreen: "KeyS",
        moveKey: "KeyF",
        overlap: "KeyO"
    };

    let isMoving = false;
    let mouseX, mouseY;

    const elementIDs = {
        sketch: "#img2img_sketch",
        inpaint: "#img2maskimg",
        inpaintSketch: "#inpaint_sketch",
        img2imgTabs: "#mode_img2img .tab-nav"
    };

    async function getElements() {
        const elements = await Promise.all(
            Object.values(elementIDs).map(id => document.querySelector(id))
        );
        return Object.fromEntries(
            Object.keys(elementIDs).map((key, index) => [key, elements[index]])
        );
    }

    const elements = await getElements();

    function applyZoomAndPan(targetElement, elemId) {
        targetElement.style.transformOrigin = "0 0";
        let [zoomLevel, panX, panY] = [1, 0, 0];
        let fullScreenMode = false;

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
            zoomLevel = 1;
            panX = 0;
            panY = 0;

            fixCanvas();
            targetElement.style.transform = `scale(${zoomLevel}) translate(${panX}px, ${panY}px)`;

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
            panX += mouseX - (mouseX * newZoomLevel) / zoomLevel;
            panY += mouseY - (mouseY * newZoomLevel) / zoomLevel;

            targetElement.style.transformOrigin = "0 0";
            targetElement.style.transform = `translate(${panX}px, ${panY}px) scale(${newZoomLevel})`;

            toggleOverlap("on");
            return newZoomLevel;
        }

        // Change the zoom level based on user interaction
        function changeZoomLevel(operation, e) {
            if (e.shiftKey) {
                e.preventDefault();

                let zoomPosX, zoomPosY;
                let delta = 0.2;
                if (zoomLevel > 7) {
                    delta = 0.9;
                } else if (zoomLevel > 2) {
                    delta = 0.6;
                }

                zoomPosX = e.clientX;
                zoomPosY = e.clientY;

                fullScreenMode = false;
                zoomLevel = updateZoom(
                    zoomLevel + (operation === "+" ? delta : -delta),
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
            zoomLevel = scale;
            panX = offsetX;
            panY = offsetY;

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

            // Get element and screen dimensions
            const elementWidth = targetElement.offsetWidth;
            const elementHeight = targetElement.offsetHeight;
            const screenWidth = window.innerWidth;
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
            zoomLevel = scale;
            panX = offsetX;
            panY = offsetY;

            fullScreenMode = true;
            toggleOverlap("on");
        }

        // Handle keydown events
        function handleKeyDown(event) {
            const hotkeyActions = {
                [hotkeysConfig.resetZoom]: resetZoom,
                [hotkeysConfig.overlap]: toggleOverlap,
                [hotkeysConfig.fitToScreen]: fitToScreen
                // [hotkeysConfig.moveKey] : moveCanvas,
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
            }
        }

        function handleMouseLeave() {
            if (isKeyDownHandlerAttached) {
                document.removeEventListener("keydown", handleKeyDown);
                isKeyDownHandlerAttached = false;
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
            if (e.ctrlKey || e.metaKey) {
                e.preventDefault();

                // Increase or decrease brush size based on scroll direction
                adjustBrushSize(elemId, e.deltaY);
            }
        });

        /**
         * Handle the move event for pan functionality. Updates the panX and panY variables and applies the new transform to the target element.
         * @param {MouseEvent} e - The mouse event.
         */
        function handleMoveKeyDown(e) {
            if (e.code === hotkeysConfig.moveKey) {
                if (!e.ctrlKey && !e.metaKey) {
                    isMoving = true;
                }
            }
        }

        function handleMoveKeyUp(e) {
            if (e.code === hotkeysConfig.moveKey) {
                isMoving = false;
            }
        }

        document.addEventListener("keydown", handleMoveKeyDown);
        document.addEventListener("keyup", handleMoveKeyUp);

        // Detect zoom level and update the pan speed.
        function updatePanPosition(movementX, movementY) {
            let panSpeed = 1.5;

            if (zoomLevel > 8) {
                panSpeed = 2.5;
            }

            panX = panX + movementX * panSpeed;
            panY = panY + movementY * panSpeed;

            targetElement.style.transform = `translate(${panX}px, ${panY}px) scale(${zoomLevel})`;
            toggleOverlap("on");
        }

        function handleMoveByKey(e) {
            if (isMoving) {
                updatePanPosition(e.movementX, e.movementY);
                targetElement.style.pointerEvents = "none";
            } else {
                targetElement.style.pointerEvents = "auto";
            }
        }

        gradioApp().addEventListener("mousemove", handleMoveByKey);
    }

    applyZoomAndPan(elements.sketch, elementIDs.sketch);
    applyZoomAndPan(elements.inpaint, elementIDs.inpaint);
    applyZoomAndPan(elements.inpaintSketch, elementIDs.inpaintSketch);
});
