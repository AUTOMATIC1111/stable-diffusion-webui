// Helper function to get the Gradio app's root element or document if not found
function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app');
    const elem = elems.length === 0 ? document : elems[0];

// Override getElementById to ensure compatibility with shadow DOM
    if (elem !== document) {
        elem.getElementById = function(id) {
            return document.getElementById(id);
        };
    }
    return elem.shadowRoot ? elem.shadowRoot : elem;
}

// Get the currently selected top-level UI tab button (e.g., "Extras")
function get_uiCurrentTab() {
    return gradioApp().querySelector('#tabs > .tab-nav > button.selected');
}

// Get the first currently visible top-level UI tab content (e.g., "txt2img" UI)
function get_uiCurrentTabContent() {
    return gradioApp().querySelector('#tabs > .tabitem[id^=tab_]:not([style*="display: none"])');
}

// Callback queues for various UI events
var uiUpdateCallbacks = [];
var uiAfterUpdateCallbacks = [];
var uiLoadedCallbacks = [];
var uiTabChangeCallbacks = [];
var optionsChangedCallbacks = [];
var optionsAvailableCallbacks = [];
var uiAfterUpdateTimeout = null;
var uiCurrentTab = null;

/**
 * Register a callback to be called at each UI update.
 * @param {Function} callback - The function to call during UI update.
 */
function onUiUpdate(callback) {
    uiUpdateCallbacks.push(callback);
}

/**
 * Register a callback to be called soon after UI updates.
 * Preferred if you don't need access to MutationRecords.
 * @param {Function} callback - The function to call after UI update.
 */
function onAfterUiUpdate(callback) {
    uiAfterUpdateCallbacks.push(callback);
}

/**
 * Register a callback to be called when the UI is loaded.
 * @param {Function} callback - The function to call when the UI is loaded.
 */
function onUiLoaded(callback) {
    uiLoadedCallbacks.push(callback);
}

/**
 * Register a callback to be called when the UI tab is changed.
 * @param {Function} callback - The function to call when the UI tab changes.
 */
function onUiTabChange(callback) {
    uiTabChangeCallbacks.push(callback);
}

/**
 * Register a callback to be called when the options are changed.
 * @param {Function} callback - The function to call when options change.
 */
function onOptionsChanged(callback) {
    optionsChangedCallbacks.push(callback);
}

/**
 * Register a callback to be called when options (in opts global variable) are available.
 * @param {Function} callback - The function to call when options are available.
 */
function onOptionsAvailable(callback) {
    if (Object.keys(opts).length !== 0) {
        callback();
        return;
    }
    optionsAvailableCallbacks.push(callback);
}

/**
 * Execute the callbacks registered for a specific event.
 * @param {Array<Function>} queue - The list of callback functions.
 * @param {any} arg - The argument to pass to the callbacks.
 */
function executeCallbacks(queue, arg) {
    for (const callback of queue) {
        try {
            callback(arg);
        } catch (e) {
            console.error("Error running callback", callback, ":", e);
        }
    }
}

/**
 * Schedule the execution of the callbacks registered with onAfterUiUpdate.
 * Ensures callbacks are executed only once even with multiple mutations observed.
 */
function scheduleAfterUiUpdateCallbacks() {
    clearTimeout(uiAfterUpdateTimeout);
    uiAfterUpdateTimeout = setTimeout(function() {
        executeCallbacks(uiAfterUpdateCallbacks);
    }, 200);
}

var executedOnLoaded = false;

// Observe DOM changes and trigger relevant callbacks
document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(mutations) {
        if (!executedOnLoaded && gradioApp().querySelector('#txt2img_prompt')) {
            executedOnLoaded = true;
            executeCallbacks(uiLoadedCallbacks);
        }

        executeCallbacks(uiUpdateCallbacks, mutations);
        scheduleAfterUiUpdateCallbacks();
        
        // Handle UI tab change detection
        const newTab = get_uiCurrentTab();
        if (newTab && (newTab !== uiCurrentTab)) {
            uiCurrentTab = newTab;
            executeCallbacks(uiTabChangeCallbacks);
        }
    });

    // Observe changes in the Gradio app's root or document
    mutationObserver.observe(gradioApp(), { childList: true, subtree: true });
});

/**
 * Add keyboard shortcuts:
 * - Ctrl+Enter to start/restart a generation
 * - Alt/Option+Enter to skip a generation
 * - Esc to interrupt a generation
 */
document.addEventListener('keydown', function(e) {
    const isEnter = e.key === 'Enter' || e.keyCode === 13;
    const isCtrlKey = e.metaKey || e.ctrlKey;
    const isAltKey = e.altKey;
    const isEsc = e.key === 'Escape';

    const generateButton = get_uiCurrentTabContent().querySelector('button[id$=_generate]');
    const interruptButton = get_uiCurrentTabContent().querySelector('button[id$=_interrupt]');
    const skipButton = get_uiCurrentTabContent().querySelector('button[id$=_skip]');

    if (isCtrlKey && isEnter) {
        if (interruptButton.style.display === 'block') {
            interruptButton.click();
            const callback = (mutationList) => {
                for (const mutation of mutationList) {
                    if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
                        if (interruptButton.style.display === 'none') {
                            generateButton.click();
                            observer.disconnect();
                        }
                    }
                }
            };
            const observer = new MutationObserver(callback);
            observer.observe(interruptButton, { attributes: true });
        } else {
            generateButton.click();
        }
        e.preventDefault();
    }

    if (isAltKey && isEnter) {
        skipButton.click();
        e.preventDefault();
    }

    if (isEsc) {
        const globalPopup = document.querySelector('.global-popup');
        const lightboxModal = document.querySelector('#lightboxModal');
        if (!globalPopup || globalPopup.style.display === 'none') {
            if (document.activeElement === lightboxModal) return;
            if (interruptButton.style.display === 'block') {
                interruptButton.click();
                e.preventDefault();
            }
        }
    }
});

/**
 * Check if a UI element is visible (not hidden within another element or tab)
 * @param {HTMLElement} el - The element to check visibility for.
 * @returns {boolean} - True if the element is visible, false otherwise.
 */
function uiElementIsVisible(el) {
    if (el === document) {
        return true;
    }

    const computedStyle = getComputedStyle(el);
    const isVisible = computedStyle.display !== 'none';

    if (!isVisible) return false;
    return uiElementIsVisible(el.parentNode);
}

/**
 * Check if a UI element is within the viewport
 * @param {HTMLElement} el - The element to check.
 * @returns {boolean} - True if the element is in sight, false otherwise.
 */
function uiElementInSight(el) {
    const clRect = el.getBoundingClientRect();
    const windowHeight = window.innerHeight;
    const isOnScreen = clRect.bottom > 0 && clRect.top < windowHeight;

    return isOnScreen;
}
