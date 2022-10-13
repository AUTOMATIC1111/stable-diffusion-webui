
var _app;
function gradioApp(){
	return _app||(_app=document.getElementsByTagName('gradio-app')[0].shadowRoot);
}

function get_uiCurrentTab() {
	return gradioApp().querySelector('.tabs button:not(.border-transparent)')
}

function get_uiCurrentTabContent() {
	return gradioApp().querySelector('.tabitem[id^=tab_]:not([style*="display: none"])')
}

let onupdates = [], ontabchanges = [], onloads = [];
let uiCurrentTab = null;

function onUiUpdate(fn){
	onupdates.push(fn);
}
function onUiTabChange(fn){
	ontabchanges.push(fn);
}
function onLoad(fn) {
	onloads?onloads.push(fn):fn();
}

function runCallback(x) {
	try {
		x();
	} catch (e) {
		console.error(e.message, e);
	}
}

document.addEventListener("DOMContentLoaded", function() {
	var debounce = setInterval(function() {
		if (!(uiCurrentTab = get_uiCurrentTab())) return;
		clearInterval(debounce);

		console.clear&&console.clear();

		onloads.forEach((e) => e());
		onloads = null;

		moRunning.observe(gradioApp(), { childList:true, subtree:true });
	}, 10);

	var moRunning = new MutationObserver(function(m) {
		clearTimeout(debounce);
		debounce = setTimeout(function() {
			onupdates.forEach(runCallback);
			const newTab = get_uiCurrentTab();
			if (newTab != uiCurrentTab) {
				uiCurrentTab = newTab;
				ontabchanges.forEach(runCallback);
			}
		}, 20);
	});
});

/**
 * Add a ctrl+enter as a shortcut to start a generation
 */
document.addEventListener('keydown', function(e) {
	var handled = false;
	if (e.key !== undefined) {
		if((e.key == "Enter" && (e.metaKey || e.ctrlKey))) handled = true;
	} else if (e.keyCode !== undefined) {
		if((e.keyCode == 13 && (e.metaKey || e.ctrlKey))) handled = true;
	}
	if (handled) {
		button = get_uiCurrentTabContent().querySelector('button[id$=_generate]');
		button&&button.click();
		e.preventDefault();
	}
});

/**
 * checks that a UI element is not in another hidden element or tab content
 */
function uiElementIsVisible(el) {
	console.warn(el);
	let isVisible = !el.closest('.\\!hidden');
	if (!isVisible) return false;

	while( isVisible = el.closest('.tabitem')?.style.display !== 'none' ) {
		if ( ! isVisible ) {
			return false;
		} else if ( el.parentElement ) {
			el = el.parentElement
		} else {
			break;
		}
	}
	return isVisible;
}