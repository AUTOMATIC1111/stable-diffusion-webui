window.onload = (function(){
    window.addEventListener('drop', e => {
        const target = e.composedPath()[0];
        const idx = selected_gallery_index();
        let prompt_target = "txt2img_prompt_image";
        if (idx === 1) {
            prompt_target = "img2img_prompt_image";
        }
        if (target.placeholder === "Prompt") {
            e.stopPropagation();
            e.preventDefault();
            const imgParent = gradioApp().getElementById(prompt_target);
            const files = e.dataTransfer.files;
            const fileInput = imgParent.querySelector('input[type="file"]');
            if ( fileInput ) {
                fileInput.files = files;
                fileInput.dispatchEvent(new Event('change'));
            }
        }
    });

});