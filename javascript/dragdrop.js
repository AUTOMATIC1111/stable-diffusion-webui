// allows drag-dropping files into gradio image elements, and also pasting images from clipboard

function isValidImageList( files ) {
    return files && files?.length === 1 && ['image/png', 'image/gif', 'image/jpeg'].includes(files[0].type);
}

function dropReplaceImage( imgWrap, files ) {
    if ( ! isValidImageList( files ) ) {
        return;
    }

    const tmpFile = files[0];

    imgWrap.querySelector('.modify-upload button + button, .touch-none + div button + button')?.click();
    const callback = () => {
        const fileInput = imgWrap.querySelector('input[type="file"]');
        if ( fileInput ) {
            if ( files.length === 0 ) {
                files = new DataTransfer();
                files.items.add(tmpFile);
                fileInput.files = files.files;
            } else {
                fileInput.files = files;
            }
            fileInput.dispatchEvent(new Event('change'));   
        }
    };
    
    if ( imgWrap.closest('#pnginfo_image') ) {
        // special treatment for PNG Info tab, wait for fetch request to finish
        const oldFetch = window.fetch;
        window.fetch = async (input, options) => {
            const response = await oldFetch(input, options);
            if ( 'api/predict/' === input ) {
                const content = await response.text();
                window.fetch = oldFetch;
                window.requestAnimationFrame( () => callback() );
                return new Response(content, {
                    status: response.status,
                    statusText: response.statusText,
                    headers: response.headers
                })
            }
            return response;
        };        
    } else {
        window.requestAnimationFrame( () => callback() );
    }
}

window.document.addEventListener('dragover', e => {
    const target = e.composedPath()[0];
    const imgWrap = target.closest('[data-testid="image"]');
    if ( !imgWrap && target.placeholder && target.placeholder.indexOf("Prompt") == -1) {
        return;
    }
    e.stopPropagation();
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
});

window.document.addEventListener('drop', e => {
    const target = e.composedPath()[0];
    if (target.placeholder.indexOf("Prompt") == -1) {
        return;
    }
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

    const visibleImageFields = [...gradioApp().querySelectorAll('[data-testid="image"]')]
        .filter(el => uiElementIsVisible(el));
    if ( ! visibleImageFields.length ) {
        return;
    }
    
    const firstFreeImageField = visibleImageFields
        .filter(el => el.querySelector('input[type=file]'))?.[0];

    dropReplaceImage(
        firstFreeImageField ?
        firstFreeImageField :
        visibleImageFields[visibleImageFields.length - 1]
    , files );
});
