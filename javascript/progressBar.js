let lastState = {};

function request(url, data, handler, errorHandler) {
  const xhr = new XMLHttpRequest();
  xhr.open('POST', url, true);
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.onreadystatechange = () => {
    if (xhr.readyState === 4) {
      if (xhr.status === 200) {
        try {
          const js = JSON.parse(xhr.responseText);
          handler(js);
        } catch (error) {
          console.error(error);
          errorHandler();
        }
      } else {
        errorHandler();
      }
    }
  };
  const js = JSON.stringify(data);
  xhr.send(js);
}

function pad2(x) {
  return x < 10 ? `0${x}` : x;
}

function formatTime(secs) {
  if (secs > 3600) return `${pad2(Math.floor(secs / 60 / 60))}:${pad2(Math.floor(secs / 60) % 60)}:${pad2(Math.floor(secs) % 60)}`;
  if (secs > 60) return `${pad2(Math.floor(secs / 60))}:${pad2(Math.floor(secs) % 60)}`;
  return `${Math.floor(secs)}s`;
}

function checkPaused(state) {
  lastState.paused = state ? !state : !lastState.paused;
  document.getElementById('txt2img_pause').innerText = lastState.paused ? 'Resume' : 'Pause';
  document.getElementById('img2img_pause').innerText = lastState.paused ? 'Resume' : 'Pause';
}

function setProgress(res) {
  const elements = ['txt2img_generate', 'img2img_generate', 'extras_generate', 'control_generate'];
  const progress = (res?.progress || 0);
  const job = res?.job || '';
  const perc = res && (progress > 0) ? `${Math.round(100.0 * progress)}%` : '';
  let sec = res?.eta || 0;
  let eta = '';
  if (res?.paused) eta = 'Paused';
  else if (res?.completed || (progress > 0.99)) eta = 'Finishing';
  else if (sec === 0) eta = 'Starting';
  else {
    const min = Math.floor(sec / 60);
    sec %= 60;
    eta = min > 0 ? `${Math.round(min)}m ${Math.round(sec)}s` : `${Math.round(sec)}s`;
  }
  document.title = `SD.Next ${perc}`;
  for (const elId of elements) {
    const el = document.getElementById(elId);
    if (el) {
      el.innerText = (res ? `${job} ${perc} ${eta}` : 'Generate');
      el.style.background = res && (progress > 0)
        ? `linear-gradient(to right, var(--primary-500) 0%, var(--primary-800) ${perc}, var(--neutral-700) ${perc})`
        : 'var(--button-primary-background-fill)';
    }
  }
}

function requestInterrupt() {
  setProgress();
}

function randomId() {
  return `task(${Math.random().toString(36).slice(2, 7)}${Math.random().toString(36).slice(2, 7)}${Math.random().toString(36).slice(2, 7)})`;
}

// starts sending progress requests to "/internal/progress" uri, creating progressbar above progressbarContainer element and preview inside gallery element
// Cleans up all created stuff when the task is over and calls atEnd. calls onProgress every time there is a progress update
function requestProgress(id_task, progressEl, galleryEl, atEnd = null, onProgress = null, once = false) {
  localStorage.setItem('task', id_task);
  let hasStarted = false;
  const dateStart = new Date();
  const prevProgress = null;
  const parentGallery = galleryEl ? galleryEl.parentNode : null;
  let livePreview;
  let img;

  const initLivePreview = () => {
    img = new Image();
    if (parentGallery) {
      livePreview = document.createElement('div');
      livePreview.className = 'livePreview';
      parentGallery.insertBefore(livePreview, galleryEl);
      const rect = galleryEl.getBoundingClientRect();
      if (rect.width) {
        livePreview.style.width = `${rect.width}px`;
        livePreview.style.height = `${rect.height}px`;
      }
      img.onload = () => {
        livePreview.appendChild(img);
        if (livePreview.childElementCount > 2) livePreview.removeChild(livePreview.firstElementChild);
      };
    }
  };

  const done = () => {
    debug('taskEnd:', id_task);
    localStorage.removeItem('task');
    setProgress();
    if (parentGallery && livePreview) parentGallery.removeChild(livePreview);
    checkPaused(true);
    if (atEnd) atEnd();
  };

  const start = (id_task, id_live_preview) => { // eslint-disable-line no-shadow
    if (!opts.live_previews_enable || opts.live_preview_refresh_period === 0 || opts.show_progress_every_n_steps === 0) return;
    request('./internal/progress', { id_task, id_live_preview }, (res) => {
      lastState = res;
      const elapsedFromStart = (new Date() - dateStart) / 1000;
      hasStarted |= res.active;
      if (res.completed || (!res.active && (hasStarted || once)) || (elapsedFromStart > 30 && !res.queued && res.progress === prevProgress)) {
        done();
        return;
      }
      setProgress(res);
      if (res.live_preview && !livePreview) initLivePreview();
      if (res.live_preview && galleryEl) img.src = res.live_preview;
      if (onProgress) onProgress(res);
      setTimeout(() => start(id_task, id_live_preview), opts.live_preview_refresh_period || 500);
    }, done);
  };
  start(id_task, 0);
}
