// code related to showing and updating progressbar shown as the image is being made

function rememberGallerySelection() {

}

function getGallerySelectedIndex() {

}

function request(url, data, handler, errorHandler) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                try {
                    var js = JSON.parse(xhr.responseText);
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
    var js = JSON.stringify(data);
    xhr.send(js);
}

function pad2(x) {
    return x < 10 ? '0' + x : x;
}

function formatTime(secs) {
  if (secs > 3600) {
      return pad2(Math.floor(secs / 60 / 60)) + ":" + pad2(Math.floor(secs / 60) % 60) + ":" + pad2(Math.floor(secs) % 60);
  } else if (secs > 60) {
      return pad2(Math.floor(secs / 60)) + ":" + pad2(Math.floor(secs) % 60);
  } else {
      return Math.floor(secs) + "s";
  }
}

function setTitle(progress) {
    var title = 'Stable Diffusion';

    if (opts.show_progress_in_title && progress) {
        title = '[' + progress.trim() + '] ' + title;
    }

    if (document.title != title) {
        document.title = title;
    }
}

function randomId() {
    return "task(" + Math.random().toString(36).slice(2, 7) + Math.random().toString(36).slice(2, 7) + Math.random().toString(36).slice(2, 7) + ")";
}

// starts sending progress requests to "/internal/progress" uri, creating progressbar above progressbarContainer element and
// preview inside gallery element. Cleans up all created stuff when the task is over and calls atEnd.
// calls onProgress every time there is a progress update
function requestProgress(
  id_task,
  progressbarContainer,
  gallery,
  atEnd,
  onProgress,
  inactivityTimeout = 40
) {
  var dateStart = new Date();
  var wasEverActive = false;
  var parentProgressbar = progressbarContainer.parentNode;
  var parentGallery = gallery ? gallery.parentNode : null;

  var divProgress = document.createElement("div");
  divProgress.className = "progressDiv";

  divProgress.style.display = opts.show_progressbar ? "block" : "none";
  var divInner = document.createElement("div");
  divInner.className = "progress";

  divProgress.appendChild(divInner);
  parentProgressbar.insertBefore(divProgress, progressbarContainer);

  if (parentGallery) {
    var livePreview = gradioApp().querySelector(".livePreview");
    if (!livePreview) {
      livePreview = document.createElement("div");
      livePreview.classList.add("livePreview", "init");
      parentGallery.insertBefore(livePreview, gallery);
    }
    livePreview.classList.remove("dropPreview");
  }

  var removeProgressBar = function () {
    setTitle("");
    if (divProgress) {
      parentProgressbar.removeChild(divProgress);
    }

    if (progressbarContainer.id == "txt2img_gallery_container") {
      showSubmitButtons("txt2img", true);
    } else if (progressbarContainer.id == "img2img_gallery_container") {
      showSubmitButtons("img2img", true);
    }
    if (livePreview) {
      if (parentGallery) parentGallery.removeChild(livePreview);
    }
    gradioApp()
      .querySelectorAll("#tabs + div, #tabs + div > *:not(ul)")
      .forEach(function (elem) {
        elem.style.setProperty("display", "block", "important");
      });
    atEnd();
  };

  var fun = function (id_task, id_live_preview) {
    request(
      "./internal/progress",
      { id_task: id_task, id_live_preview: id_live_preview },
      function (res) {
        if (res.completed) {
          removeProgressBar();
          return;
        }

        var rect = progressbarContainer.getBoundingClientRect();

        if (rect.width) {
          divProgress.style.width = rect.width + "px";
        }

        let progressText = "";

        divInner.style.width = (res.progress || 0) * 100.0 + "%";
        divInner.style.background = res.progress ? "" : "transparent";

        if (res.progress > 0) {
          progressText = ((res.progress || 0) * 100.0).toFixed(0) + "%";
        }

        if (res.eta) {
          progressText += " ETA: " + formatTime(res.eta);
        }

        setTitle(progressText);

        if (res.textinfo && res.textinfo.indexOf("\n") == -1) {
          progressText = res.textinfo + " " + progressText;
        }

        divInner.textContent = progressText;

        var elapsedFromStart = (new Date() - dateStart) / 1000;

        if (res.active) wasEverActive = true;

        if (!res.active && wasEverActive) {
          removeProgressBar();
          return;
        }

        if (
          elapsedFromStart > inactivityTimeout &&
          !res.queued &&
          !res.active
        ) {
          removeProgressBar();
          return;
        }

        if (res.live_preview && gallery) {

          livePreview.classList.remove("init");

          var img = new Image();
          img.onload = function () {
            img.width = img.naturalWidth;
            img.height = img.naturalHeight;
            livePreview.appendChild(img);
            if (livePreview.childElementCount > 2) {
              livePreview.removeChild(livePreview.firstElementChild);
            }
          };
          img.src = res.live_preview;
        }

        if (onProgress) {
          onProgress(res);
        }

        setTimeout(() => {
          fun(id_task, res.id_live_preview);
        }, opts.live_preview_refresh_period || 500);
      },
      function () {
        removeProgressBar();
      }
    );
  };

  fun(id_task, 0);
}
