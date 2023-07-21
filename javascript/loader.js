async function preloadImages() {
  const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  const imagePromises = [];
  const num = Math.floor(10 * Math.random());
  const imageUrls = [
    `file=html/logo-bg-${dark ? 'dark' : 'light'}.jpg`,
    `file=html/logo-bg-${num}.jpg`,
  ];
  for (const url of imageUrls) {
    const img = new Image();
    const promise = new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
    });
    img.src = url;
    imagePromises.push(promise);
  }
  try {
    await Promise.all(imagePromises);
  } catch (error) {
    console.error('Error preloading images:', error);
  }
}

async function createSplash() {
  const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  console.log('createSplash', { theme: dark ? 'dark' : 'light' });
  const num = Math.floor(Math.random() * 7) + 1;
  const splash = `
    <div id="splash" class="splash" style="background: ${dark ? 'black' : 'white'}">
      <div class="loading"><div class="loader"></div></div>
    </div>`;
  document.body.insertAdjacentHTML('beforeend', splash);
  await preloadImages();
  const imgElement = `<div class="splash-img" alt="logo" style="background-image: url(file=html/logo-bg-${dark ? 'dark' : 'light'}.jpg), url(file=html/logo-bg-${num}.jpg); background-blend-mode: ${dark ? 'darken' : 'lighten'}"></div>`;
  document.getElementById('splash').insertAdjacentHTML('afterbegin', imgElement);
}
async function removeSplash() {
  const splash = document.getElementById('splash');
  if (splash) splash.remove();
  console.log('removeSplash');
}

window.onload = createSplash;
