function gradioApp(){
    return document.getElementsByTagName('gradio-app')[0].shadowRoot;
}

function get_uiCurrentTab() {
    return gradioApp().querySelector('.tabs button:not(.border-transparent)')
}

uiUpdateCallbacks = []
uiTabChangeCallbacks = []
let uiCurrentTab = null

function onUiUpdate(callback){
    uiUpdateCallbacks.push(callback)
}
function onUiTabChange(callback){
    uiTabChangeCallbacks.push(callback)
}

function runCallback(x){
    try {
        x()
    } catch (e) {
        (console.error || console.log).call(console, e.message, e);
    }
}
function executeCallbacks(queue) {
    queue.forEach(runCallback)
}

document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m){
        executeCallbacks(uiUpdateCallbacks);
        const newTab = get_uiCurrentTab();
        if ( newTab && ( newTab !== uiCurrentTab ) ) {
            uiCurrentTab = newTab;
            executeCallbacks(uiTabChangeCallbacks);
        }
    });
    mutationObserver.observe( gradioApp(), { childList:true, subtree:true })
});
