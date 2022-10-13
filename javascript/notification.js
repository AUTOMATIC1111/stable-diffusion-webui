// Monitors the gallery and sends a browser notification when the leading image is new.

let lastHeadImg;

onLoad(function() {
	var btn = gradioApp().getElementById('request_notifications');

	if(btn) {
		btn.addEventListener('click', function (e) {
			Notification.requestPermission();
		}, true);
	}
});

onUiUpdate(function(){
	const galleryPreviews = gradioApp().querySelectorAll('img.h-full.w-full.overflow-hidden');
	if (!galleryPreviews) return;

	const headImg = galleryPreviews[0]?.src;

	if (!headImg || headImg == lastHeadImg) return;
	lastHeadImg = headImg;

	// play notification sound if available
	gradioApp().querySelector('#audio_notification audio')?.play();

	if (document.hasFocus()) return;

	// Multiple copies of the images are in the DOM when one is selected. Dedup with a Set to get the real number generated.
	const imgs = new Set(Array.from(galleryPreviews).map(img => img.src)).size;

    const notification = new Notification(
        'Stable Diffusion',
        {
            body: `Generated ${imgs.size > 1 ? imgs.size - 1 : 1} image${imgs.size > 1 ? 's' : ''}`,
            icon: headImg,
            image: headImg,
        }
    );

	n.onclick = function() {
		parent.focus();
		this.close();
	};
});
