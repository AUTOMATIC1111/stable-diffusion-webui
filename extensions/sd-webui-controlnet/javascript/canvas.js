(function () {
    var hasApplied = false;
    onUiUpdate(function () {
        if (!hasApplied) {
            if (typeof window.applyZoomAndPanIntegration === "function") {
                hasApplied = true;
                window.applyZoomAndPanIntegration("#txt2img_controlnet", Array.from({ length: 20 }, (_, i) => `#txt2img_controlnet_ControlNet-${i}_input_image`));
                window.applyZoomAndPanIntegration("#img2img_controlnet", Array.from({ length: 20 }, (_, i) => `#img2img_controlnet_ControlNet-${i}_input_image`));
                window.applyZoomAndPanIntegration("#txt2img_controlnet", ["#txt2img_controlnet_ControlNet_input_image"]);
                window.applyZoomAndPanIntegration("#img2img_controlnet", ["#img2img_controlnet_ControlNet_input_image"]);
                //console.log("window.applyZoomAndPanIntegration applied.");
            } else {
                //console.log("window.applyZoomAndPanIntegration is not available.");
            }
        }
    });
})();
