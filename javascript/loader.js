async function createSplash() {
  const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  const num = Math.floor(Math.random() * 7) + 1;
  const splash = `
    <div id="splash" class="splash" style="background: ${dark ? 'black' : 'white'}">
      <div class="loading"><div class="loader"></div></div>
      <div class="splash-img" alt="logo" style="background-image: url(file=html/logo-bg-${dark ? 'dark' : 'light'}.jpg), url(file=html/logo-bg-${num}.jpg); background-blend-mode: ${dark ? 'darken' : 'lighten'}"></div>
    </div>`;
  document.body.insertAdjacentHTML('beforeend', splash);
  console.log('createSplash', { 'system-theme': dark ? 'dark' : 'light' });
}

async function removeSplash() { // called at the end of setHints
  const splash = document.getElementById('splash');
  if (splash) splash.remove();
  console.log('removeSplash');
}

window.onload = createSplash;
