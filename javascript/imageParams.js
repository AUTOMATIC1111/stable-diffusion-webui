let dragDropInitialized = false;

async function initDragDrop() {
  if (dragDropInitialized) return;
  dragDropInitialized = true;
  log('initDragDrop');
  window.addEventListener('drop', (e) => {
    const target = e.composedPath()[0];
    if (!target.placeholder) return;
    if (target.placeholder.indexOf('Prompt') === -1) return;
    const promptTarget = get_tab_index('tabs') === 1 ? 'img2img_prompt_image' : 'txt2img_prompt_image';
    const imgParent = gradioApp().getElementById(promptTarget);
    const fileInput = imgParent.querySelector('input[type="file"]');
    if (!imgParent || !fileInput) return;
    if ((e.dataTransfer?.files?.length || 0) > 0) {
      e.stopPropagation();
      e.preventDefault();
      fileInput.files = e.dataTransfer.files;
      fileInput.dispatchEvent(new Event('change'));
      log('dropEvent files', fileInput.files);
    }
  });
}

onAfterUiUpdate(initDragDrop);
