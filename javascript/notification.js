// Monitors the gallery and sends a browser notification when the leading image is new.

let lastHeadImg = null;

let notificationButton = null;

onAfterUiUpdate(function() {
    if (notificationButton == null) {
        notificationButton = gradioApp().getElementById('request_notifications');

        if (notificationButton != null) {
            notificationButton.addEventListener('click', () => {
                void Notification.requestPermission();
            }, true);
        }
    }

    const galleryPreviews = gradioApp().querySelectorAll('div[id^="tab_"] div[id$="_results"] .thumbnail-item > img');

    if (galleryPreviews == null) return;

    const headImg = galleryPreviews[0]?.src;

    if (headImg == null || headImg == lastHeadImg) return;

    lastHeadImg = headImg;

    // play notification sound if available
    const notificationAudio = gradioApp().querySelector('#audio_notification audio');
    if (notificationAudio) {
        notificationAudio.volume = opts.notification_volume / 100.0 || 1.0;
        notificationAudio.play();
    }

    if (document.hasFocus()) return;

    // Multiple copies of the images are in the DOM when one is selected. Dedup with a Set to get the real number generated.
    const imgs = new Set(Array.from(galleryPreviews).map(img => img.src));

    const notification = new Notification(
        'Stable Diffusion',
        {
            body: `Generated ${imgs.size > 1 ? imgs.size - opts.return_grid : 1} image${imgs.size > 1 ? 's' : ''}`,
            icon: headImg,
            image: headImg,
        }
    );

    notification.onclick = function(_) {
        parent.focus();
        this.close();
    };
});
