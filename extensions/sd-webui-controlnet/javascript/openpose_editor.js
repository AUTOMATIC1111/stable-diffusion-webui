(function () {
    async function checkEditorAvailable() {
        const LOCAL_EDITOR_PATH = '/openpose_editor_index';
        const REMOTE_EDITOR_PATH = 'https://huchenlei.github.io/sd-webui-openpose-editor/';

        async function testEditorPath(path) {
            const res = await fetch(path);
            return res.status === 200 ? path : null;
        }

        // Use local editor if the user has the extension installed. Fallback
        // onto remote editor if the local editor is not ready yet.
        // See https://github.com/huchenlei/sd-webui-openpose-editor/issues/53
        // for more details.
        return await testEditorPath(LOCAL_EDITOR_PATH) || await testEditorPath(REMOTE_EDITOR_PATH);
    }

    const cnetOpenposeEditorRegisteredElements = new Set();
    let editorURL = null;
    function loadOpenposeEditor() {
        // Simulate an `input` DOM event for Gradio Textbox component. Needed after you edit its contents in javascript, otherwise your edits
        // will only visible on web page and not sent to python.
        function updateInput(target) {
            let e = new Event("input", { bubbles: true })
            Object.defineProperty(e, "target", { value: target })
            target.dispatchEvent(e);
        }

        function navigateIframe(iframe, editorURL) {
            function getPathname(rawURL) {
                try {
                    return new URL(rawURL).pathname;
                } catch (e) {
                    return rawURL;
                }
            }

            return new Promise((resolve) => {
                const darkThemeParam = document.body.classList.contains('dark') ?
                    new URLSearchParams({ theme: 'dark' }).toString() :
                    '';

                window.addEventListener('message', (event) => {
                    const message = event.data;
                    if (message['ready']) resolve();
                }, { once: true });

                if ((editorURL.startsWith("http") ? iframe.src : getPathname(iframe.src)) !== editorURL) {
                    iframe.src = `${editorURL}?${darkThemeParam}`;
                    // By default assume 5 second is enough for the openpose editor
                    // to load.
                    setTimeout(resolve, 5000);
                } else {
                    // If no navigation is required, immediately return.
                    resolve();
                }
            });
        }
        const tabs = gradioApp().querySelectorAll('.cnet-unit-tab');
        tabs.forEach(tab => {
            if (cnetOpenposeEditorRegisteredElements.has(tab)) return;
            cnetOpenposeEditorRegisteredElements.add(tab);

            const generatedImageGroup = tab.querySelector('.cnet-generated-image-group');
            const editButton = generatedImageGroup.querySelector('.cnet-edit-pose');

            editButton.addEventListener('click', async () => {
                const inputImageGroup = tab.querySelector('.cnet-input-image-group');
                const inputImage = inputImageGroup.querySelector('.cnet-image img');
                const downloadLink = generatedImageGroup.querySelector('.cnet-download-pose a');
                const modalId = editButton.id.replace('cnet-modal-open-', '');
                const modalIframe = generatedImageGroup.querySelector('.cnet-modal iframe');

                if (!editorURL) {
                    editorURL = await checkEditorAvailable();
                    if (!editorURL) {
                        alert("No openpose editor available.")
                    }
                }

                await navigateIframe(modalIframe, editorURL);
                modalIframe.contentWindow.postMessage({
                    modalId,
                    imageURL: inputImage ? inputImage.src : undefined,
                    poseURL: downloadLink.href,
                }, '*');
                // Focus the iframe so that the focus is no longer on the `Edit` button.
                // Pressing space when the focus is on `Edit` button will trigger
                // the click again to resend the frame message.
                modalIframe.contentWindow.focus();
            });
            /*
            * Writes the pose data URL to an link element on input image group.
            * Click a hidden button to trigger a backend rendering of the pose JSON.
            *
            * The backend should:
            * - Set the rendered pose image as preprocessor generated image.
            */
            function updatePreviewPose(poseURL) {
                const downloadLink = generatedImageGroup.querySelector('.cnet-download-pose a');
                const renderButton = generatedImageGroup.querySelector('.cnet-render-pose');
                const poseTextbox = generatedImageGroup.querySelector('.cnet-pose-json textarea');
                const allowPreviewCheckbox = tab.querySelector('.cnet-allow-preview input');

                if (!allowPreviewCheckbox.checked)
                    allowPreviewCheckbox.click();

                // Only set href when download link exists and needs an update. `downloadLink`
                // can be null when user closes preview and click `Upload JSON` button again.
                // https://github.com/Mikubill/sd-webui-controlnet/issues/2308
                if (downloadLink !== null)
                    downloadLink.href = poseURL;

                poseTextbox.value = poseURL;
                updateInput(poseTextbox);
                renderButton.click();
            }

            // Updates preview image when edit is done.
            window.addEventListener('message', (event) => {
                const message = event.data;
                const modalId = editButton.id.replace('cnet-modal-open-', '');
                if (message.modalId !== modalId) return;
                updatePreviewPose(message.poseURL);

                const closeModalButton = generatedImageGroup.querySelector('.cnet-modal .cnet-modal-close');
                closeModalButton.click();
            });

            const inputImageGroup = tab.querySelector('.cnet-input-image-group');
            const uploadButton = inputImageGroup.querySelector('.cnet-upload-pose input');
            // Updates preview image when JSON file is uploaded.
            uploadButton.addEventListener('change', (event) => {
                const file = event.target.files[0];
                if (!file)
                    return;

                const reader = new FileReader();
                reader.onload = function (e) {
                    const contents = e.target.result;
                    const poseURL = `data:application/json;base64,${btoa(contents)}`;
                    updatePreviewPose(poseURL);
                };
                reader.readAsText(file);
                // Reset the file input value so that uploading the same file still triggers callback.
                event.target.value = '';
            });
        });
    }

    onUiUpdate(loadOpenposeEditor);
})();