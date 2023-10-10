const appStartTime = performance.now();

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
  log('createSplash', { theme: dark ? 'dark' : 'light' });
  const num = Math.floor(11 * Math.random());
  const splash = `
    <div id="splash" class="splash" style="background: ${dark ? 'black' : 'white'}">
      <div class="loading"><div class="loader"></div></div>
      <div id="motd" class="motd""></div>
    </div>`;
  document.body.insertAdjacentHTML('beforeend', splash);
  await preloadImages();
  const imgEl = `<div id="spash-img" class="splash-img" alt="logo" style="background-image: url(file=html/logo-bg-${dark ? 'dark' : 'light'}.jpg), url(file=html/logo-bg-${num}.jpg); background-blend-mode: ${dark ? 'multiply' : 'lighten'}"></div>`;
  document.getElementById('splash').insertAdjacentHTML('afterbegin', imgEl);
  fetch('/sdapi/v1/motd')
    .then((res) => res.text())
    .then((text) => document.getElementById('motd').innerHTML = text.replace(/["]+/g, ''))
    .catch((err) => console.error('getMOTD:', err));
}

async function removeSplash() {
  const splash = document.getElementById('splash');
  if (splash) splash.remove();
  log('removeSplash');
  log('startupTime', Math.round(performance.now() - appStartTime) / 1000);
}

window.onload = createSplash;
