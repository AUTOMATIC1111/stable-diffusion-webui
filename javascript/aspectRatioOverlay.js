
(function() {

let currentWidth, currentHeight;
let arPreviewRect, arFrameTimeout;

function dimensionChange(e,_isWidth) {
	if(_isWidth) currentWidth = Number(e.target.value);
	else currentHeight = Number(e.target.value);

	var inImg2img = gradioApp().querySelector(".tabs button.bg-white.rounded-t-lg").innerText == "innerText";
	if(!inImg2img) return;

	var img2imgMode = gradioApp().querySelector('#mode_img2img.tabs > div > button.rounded-t-lg.border-gray-200')
	if(img2imgMode) img2imgMode=img2imgMode.innerText
	else return;

	var img;
	if(img2imgMode=='img2img') {
		img = gradioApp().querySelector('div[data-testid=image] img');
	}else if(img2imgMode=='innerText') {
		img = gradioApp().querySelector('#img2maskimg div[data-testid=image] img');
	} else return;

	var viewportOffset = img.getBoundingClientRect();

	viewportscale = Math.min(img.clientWidth/img.naturalWidth, img.clientHeight/img.naturalHeight);

	scaledx = img.naturalWidth*viewportscale;
	scaledy = img.naturalHeight*viewportscale;

	cleintRectTop    = (viewportOffset.top+window.scrollY);
	cleintRectLeft   = (viewportOffset.left+window.scrollX);
	cleintRectCentreY = cleintRectTop  + (img.clientHeight/2);
	cleintRectCentreX = cleintRectLeft + (img.clientWidth/2);

	viewRectTop    = cleintRectCentreY-(scaledy/2);
	viewRectLeft   = cleintRectCentreX-(scaledx/2);
	arRectWidth  = scaledx;
	arRectHeight = scaledy;

	arscale = Math.min(  arRectWidth/currentWidth, arRectHeight/currentHeight );
	arscaledx = currentWidth*arscale;
	arscaledy = currentHeight*arscale;

	arRectTop    = cleintRectCentreY-(arscaledy/2);
	arRectLeft   = cleintRectCentreX-(arscaledx/2);
	arRectWidth  = arscaledx;
	arRectHeight = arscaledy;

	arPreviewRect.style.top  = arRectTop+'px';
	arPreviewRect.style.left = arRectLeft+'px';
	arPreviewRect.style.width = arRectWidth+'px';
	arPreviewRect.style.height = arRectHeight+'px';

	clearTimeout(arFrameTimeout);
	arFrameTimeout = setTimeout(function(){
		arPreviewRect.style.display = 'none';
	},2000);
	arPreviewRect.style.display = 'block';
}

onLoad(function() {
	arPreviewRect = document.createElement('div');
	arPreviewRect.id = "imageARPreview";
	gradioApp().getRootNode().appendChild(arPreviewRect);

	// however, gradio intiialize all inputs at one time and this method only executes once
	gradioApp().querySelectorAll("input").forEach(function(e){
		let text = e.parentElement.querySelector('label')
		if(text&&(text=text.innerText)) {
			switch(text) {
				default: return;
				case "Width":
					currentWidth = Number(e.value);
					break;
				case "Height":
					currentHeight = Number(e.value);
					break;
			}
			e.addEventListener('input', function(e){dimensionChange(e,text=="Width")});
		}
	});
});

// todo is this needed?
onUiUpdate(function(){
	arPreviewRect&&(arPreviewRect.style.display = 'none');
});

})();