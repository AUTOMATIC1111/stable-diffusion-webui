// Monitors the gallery and sends a browser notification when the leading image is new.

let lastHeadImg = null;
let notificationButton = null;

onUiUpdate(function () {
  if (!notificationButton) {
    notificationButton = gradioApp().getElementById('request_notifications');
    if (notificationButton) notificationButton.addEventListener('click', (evt) => Notification.requestPermission(), true);
  }
  if (document.hasFocus()) return; // window is in focus so don't send notifications
  const galleryPreviews = gradioApp().querySelectorAll('div[id^="tab_"][style*="display: block"] div[id$="_results"] .thumbnail-item > img');
  if (!galleryPreviews) return;
  const headImg = galleryPreviews[0]?.src;
  if (!headImg || headImg == lastHeadImg || headImg.endsWith('logo.png')) return;
  const audioNotification = gradioApp().querySelector('#audio_notification audio');
  if (audioNotification) audioNotification.play();
  lastHeadImg = headImg;
  const imgs = new Set(Array.from(galleryPreviews).map((img) => img.src)); // Multiple copies of the images are in the DOM when one is selected
  const notification = new Notification('SD.Next', {
    body: `Generated ${imgs.size > 1 ? imgs.size - opts.return_grid : 1} image${imgs.size > 1 ? 's' : ''}`,
    icon: headImg,
    image: headImg,
  });
  notification.onclick = () => {
    parent.focus();
    this.close();
  };
});
