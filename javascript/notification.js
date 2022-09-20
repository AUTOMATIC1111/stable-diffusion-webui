// Monitors the gallery and sends a browser notification when the leading image is new.

let lastHeadImg = null;

onUiUpdate(function(){
    const galleryPreviews = gradioApp().querySelectorAll('img.h-full.w-full.overflow-hidden');

    if (galleryPreviews == null) return;

    const headImg = galleryPreviews[0]?.src;

    if (headImg == null || headImg == lastHeadImg) return;

    lastHeadImg = headImg;

    if (document.hasFocus()) return;

    // Multiple copies of the images are in the DOM when one is selected. Dedup with a Set to get the real number generated.
    const imgs = new Set(Array.from(galleryPreviews).map(img => img.src));

    const notification = new Notification(
        'Stable Diffusion',
        {
            body: `Generated ${imgs.size > 1 ? imgs.size - 1 : 1} image${imgs.size > 1 ? 's' : ''}`,
            icon: headImg,
            image: headImg,
        }
    );

    notification.onclick = function(_){
        parent.focus();
        this.close();
    };
});
