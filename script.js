function gradioApp() {
    const gradioShadowRoot = document.getElementsByTagName('gradio-app')[0].shadowRoot;
    return gradioShadowRoot || document;
}

function getUICurrentTab() {
    return gradioApp().querySelector('.tabs button:not(.border-transparent)');
}

function getUICurrentTabContent() {
    return gradioApp().querySelector('.tabitem[id^=tab_]:not([style*="display: none"])');
}

let uiUpdateCallbacks = [];
let uiTabChangeCallbacks = [];
let uiCurrentTab = null;

function onUiUpdate(callback) {
    uiUpdateCallbacks.push(callback);
}

function onUiTabChange(callback) {
    uiTabChangeCallbacks.push(callback);
}

function executeCallbacks(queue, message) {
    queue.forEach(function(callback) {
        try {
            callback(message);
        } catch (error) {
            (console.error || console.log).call(console, error.message, error);
        }
    });
}

document.addEventListener('DOMContentLoaded', function() {
    const mutationObserver = new MutationObserver(function(mutations) {
        executeCallbacks(uiUpdateCallbacks, mutations);
        const newTab = getUICurrentTab();
        if (newTab && newTab !== uiCurrentTab) {
            uiCurrentTab = newTab;
            executeCallbacks(uiTabChangeCallbacks);
        }
    });
    mutationObserver.observe(gradioApp(), { childList: true, subtree: true });
});

document.addEventListener('keydown', function(event) {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey || event.altKey)) {
        const button = getUICurrentTabContent().querySelector('button[id$=_generate]');
        if (button) {
            button.click();
        }
        event.preventDefault();
    }
});

function uiElementIsVisible(element) {
    let isVisible = !element.closest('.\\!hidden');
    if (!isVisible) {
        return false;
    }
  
    while ((element = element.closest('.tabitem'))) {
      if (window.getComputedStyle(element).display === 'none') {
        return false;
      }
    }
    return true;
}  
