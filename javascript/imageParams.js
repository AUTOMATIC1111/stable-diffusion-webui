window.onload = (function(){
    window.addEventListener('drop', e => {
        const target = e.composedPath()[0];
        const idx = selected_gallery_index();
        if (target.placeholder.indexOf("Prompt") == -1) return;

        let prompt_target = get_tab_index('tabs') == 1 ? "img2img_prompt_image" : "txt2img_prompt_image";

        e.stopPropagation();
        e.preventDefault();
        const imgParent = gradioApp().getElementById(prompt_target);
        const files = e.dataTransfer.files;
        const fileInput = imgParent.querySelector('input[type="file"]');
        if ( fileInput ) {
            fileInput.files = files;
            fileInput.dispatchEvent(new Event('change'));
        }
    });
});
