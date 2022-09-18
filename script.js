function gradioApp(){
    return document.getElementsByTagName('gradio-app')[0].shadowRoot;
}

uiUpdateCallbacks = []
function onUiUpdate(callback){
    uiUpdateCallbacks.push(callback)
}

function uiUpdate(root){
	uiUpdateCallbacks.forEach(function(x){
	    x()
	})
}

document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m){
        uiUpdate(gradioApp());
    });
    mutationObserver.observe( gradioApp(), { childList:true, subtree:true })
});
