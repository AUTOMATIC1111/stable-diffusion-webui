let dragDropInitialized = false;

async function initDragDrop() {
  if (dragDropInitialized) return;
  dragDropInitialized = true;
  log('initDragDrop');
  window.addEventListener('drop', (e) => {
    const target = e.composedPath()[0];
    if (!target.placeholder) return;
    if (target.placeholder.indexOf('Prompt') === -1) return;
    const tab = get_tab_index('tabs');
    let promptTarget = '';
    if (tab === 0) promptTarget = 'txt2img_prompt_image';
    else if (tab === 1) promptTarget = 'img2img_prompt_image';
    else if (tab === 2) promptTarget = 'control_prompt_image';
    else return;
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
