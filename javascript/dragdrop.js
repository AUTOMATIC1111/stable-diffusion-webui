// allows drag-dropping files into gradio image elements, and also pasting images from clipboard

function isValidImageList( files ) {
    return files && files?.length === 1 && ['image/png', 'image/gif', 'image/jpeg'].includes(files[0].type);
}

function dropReplaceImage( imgWrap, files ) {
    if ( ! isValidImageList( files ) ) {
        return;
    }

    imgWrap.querySelector('.modify-upload button + button')?.click();
    window.requestAnimationFrame( () => {
        const fileInput = imgWrap.querySelector('input[type="file"]');
        if ( fileInput ) {
            fileInput.files = files;
            fileInput.dispatchEvent(new Event('change'));   
        }
    });
}

window.document.addEventListener('dragover', e => {
    const target = e.composedPath()[0];
    const imgWrap = target.closest('[data-testid="image"]');
    if ( !imgWrap ) {
        return;
    }
    e.stopPropagation();
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
});

window.document.addEventListener('drop', e => {
    const target = e.composedPath()[0];
    const imgWrap = target.closest('[data-testid="image"]');
    if ( !imgWrap ) {
        return;
    }
    e.stopPropagation();
    e.preventDefault();
    const files = e.dataTransfer.files;
    dropReplaceImage( imgWrap, files );
});

window.addEventListener('paste', e => {
    const files = e.clipboardData.files;
    if ( ! isValidImageList( files ) ) {
        return;
    }
    [...gradioApp().querySelectorAll('input[type=file][accept="image/x-png,image/gif,image/jpeg"]')]
        .filter(input => !input.matches('.\\!hidden input[type=file]'))
        .forEach(input => {
            input.files = files;
            input.dispatchEvent(new Event('change'))
        });
    [...gradioApp().querySelectorAll('[data-testid="image"]')]
        .filter(imgWrap => !imgWrap.closest('.\\!hidden'))
        .forEach(imgWrap => dropReplaceImage( imgWrap, files ));
});
