let prevSelectedIndex = -1;

window.JobController.addEventListener("jobStart", (jobName) => {
  console.log("Job started:", jobName);
  const interruptBtn = gradioApp().getElementById(jobName + "_interrupt");
  const skipBtn = gradioApp().getElementById(jobName + "_skip");

  if (interruptBtn) {
    interruptBtn.style.display = "block";
  }

  if (skipBtn) {
    skipBtn.style.display = "block";
  }

  prevSelectedIndex = selected_gallery_index();
});

window.JobController.addEventListener("jobEnd", (jobName) => {
  console.log("Job ended:", jobName);
  const interruptBtn = gradioApp().getElementById(jobName + "_interrupt");
  const skipBtn = gradioApp().getElementById(jobName + "_skip");

  interruptBtn.style.display = "none";

  if (skipBtn) {
    skipBtn.style.display = "none";
  }

  updateGallery(jobName);
  updatePageTitle(opts);
});

function updatePageTitle(opts, textToAppend) {
  if (opts.show_progress_in_title) {
    let newTitle = "Stable Diffusion";

    if (textToAppend) {
      newTitle = "Stable Diffusion - " + textToAppend;
    }

    document.title = newTitle;
  }
}

function updateGallery(jobName) {
  let galleryButtons = gradioApp().querySelectorAll(`#${jobName}_gallery .gallery-item`);
  let galleryBtnSelected = gradioApp().querySelector(`#${jobName}_gallery .gallery-item.\\!ring-2`);

  if (prevSelectedIndex !== -1 && galleryButtons.length > prevSelectedIndex && !galleryBtnSelected) {
    // automatically re-open previously selected index (if exists)
    activeElement = gradioApp().activeElement;

    galleryButtons[prevSelectedIndex].click();
    showGalleryImage();

    if (activeElement) {
      // i fought this for about an hour; i don't know why the focus is lost or why this helps recover it
      // if somenoe has a better solution please by all means
      setTimeout(function () {
        activeElement.focus();
      }, 1);
    }
  }
}

let progressSpanExists = false;

onUiUpdate(function () {
  const jobName = window.JobController.getJob();

  if (jobName) {
    const existsNow = !!gradioApp().getElementById(jobName + "_progress_span");
    console.log("UI update jobName: " + jobName);
    console.log("progressSpanExists: " + progressSpanExists);

    if (progressSpanExists && !existsNow) {
      window.JobController.endJob();
    }

    progressSpanExists = existsNow;

    if (progressSpanExists) {
      const progressBar = gradioApp().getElementById(jobName + "_progressbar");
      updatePageTitle(opts, progressBar.innerText);

      console.log("Calling check progress", jobName);

      const checkProgressBtn = gradioApp().getElementById(jobName + "_check_progress");
      if (checkProgressBtn) {
        checkProgressBtn.click();
      }
    } else {
      console.log("UI update, no job");
    }
  }
});

function requestProgress(id_part) {
  window.JobController.startJob(id_part);

  btn = gradioApp().getElementById(id_part + "_check_progress_initial");

  if (btn) {
    btn.click();
  }
}
