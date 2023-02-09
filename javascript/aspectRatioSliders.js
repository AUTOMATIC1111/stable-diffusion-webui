class AspectRatioSliderController {
    constructor(widthSlider, heightSlider, ratioSource, roundingSource, roundingMethod) {
        //References
        this.widthSlider = new SliderComponentController(widthSlider);
        this.heightSlider = new SliderComponentController(heightSlider);
        this.ratioSource = new DropdownComponentController(ratioSource);
        this.roundingSource = new CheckboxComponentController(roundingSource);
        this.roundingMethod = new RadioComponentController(roundingMethod);
        this.roundingIndicatorBadge = document.createElement("div");
        // Badge implementation
        this.roundingIndicatorBadge.innerText = "📐";
        this.roundingIndicatorBadge.classList.add("rounding-badge");
        this.ratioSource.element.appendChild(this.roundingIndicatorBadge);
        // Check initial value of ratioSource to set badge visbility
        let initialRatio = this.ratioSource.getVal();
        if (!initialRatio.includes(":")) {
            this.roundingIndicatorBadge.style.display = "none";
        }
        //Adjust badge icon if rounding is on
        if (this.roundingSource.getVal()) {
            this.roundingIndicatorBadge.classList.add("active");
            this.roundingIndicatorBadge.innerText = "⚠️";
        }
        //Make badge clickable to toggle setting
        this.roundingIndicatorBadge.addEventListener("click", () => {
            this.roundingSource.setVal(!this.roundingSource.getVal());
        });
        //Make rounding setting toggle badge text and style if setting changes
        this.roundingSource.child.addEventListener("change", () => {
            if (this.roundingSource.getVal()) {
                this.roundingIndicatorBadge.classList.add("active");
                this.roundingIndicatorBadge.innerText = "⚠️";
            }
            else {
                this.roundingIndicatorBadge.classList.remove("active");
                this.roundingIndicatorBadge.innerText = "📐";
            }
            this.adjustStepSize();
        });
        //Other event listeners
        this.widthSlider.childRangeField.addEventListener("change", (e) => { e.preventDefault(); this.resize("width"); });
        this.widthSlider.childNumField.addEventListener("change", (e) => { e.preventDefault(); this.resize("width"); });
        this.heightSlider.childRangeField.addEventListener("change", (e) => { e.preventDefault(); this.resize("height"); });
        this.heightSlider.childNumField.addEventListener("change", (e) => { e.preventDefault(); this.resize("height"); });
        this.ratioSource.childSelector.addEventListener("change", (e) => {
            e.preventDefault();
            //Check and toggle display of badge conditionally on dropdown selection
            if (!this.ratioSource.getVal().includes(":")) {
                this.roundingIndicatorBadge.style.display = 'none';
            }
            else {
                this.roundingIndicatorBadge.style.display = 'block';
            }
            this.adjustStepSize();
        });
    }
    resize(dimension) {
        //For moving slider or number field
        let val = this.ratioSource.getVal();
        if (!val.includes(":")) {
            return;
        }
        let [width, height] = val.split(":").map(Number);
        let ratio = width / height;
        if (dimension == 'width') {
            let newHeight = parseInt(this.widthSlider.getVal()) / ratio;
            if (this.roundingSource.getVal()) {
                switch (this.roundingMethod.getVal()) {
                    case 'Round':
                        newHeight = Math.round(newHeight / 8) * 8;
                        break;
                    case 'Ceiling':
                        newHeight = Math.ceil(newHeight / 8) * 8;
                        break;
                    case 'Floor':
                        newHeight = Math.floor(newHeight / 8) * 8;
                        break;
                }
            }
            this.heightSlider.setVal(newHeight.toString());
        }
        else if (dimension == "height") {
            let newWidth = parseInt(this.heightSlider.getVal()) * ratio;
            if (this.roundingSource.getVal()) {
                switch (this.roundingMethod.getVal()) {
                    case 'Round':
                        newWidth = Math.round(newWidth / 8) * 8;
                        break;
                    case 'Ceiling':
                        newWidth = Math.ceil(newWidth / 8) * 8;
                        break;
                    case 'Floor':
                        newWidth = Math.floor(newWidth / 8) * 8;
                        break;
                }
            }
            this.widthSlider.setVal(newWidth.toString());
        }
    }
    adjustStepSize() {
        /* Sets scales/precision/rounding steps;*/
        let val = this.ratioSource.getVal();
        if (!val.includes(":")) {
            //If ratio unlocked
            this.widthSlider.childRangeField.step = "8";
            this.widthSlider.childRangeField.min = "64";
            this.widthSlider.childNumField.step = "8";
            this.widthSlider.childNumField.min = "64";
            this.heightSlider.childRangeField.step = "8";
            this.heightSlider.childRangeField.min = "64";
            this.heightSlider.childNumField.step = "8";
            this.heightSlider.childNumField.min = "64";
            return;
        }
        //Format string and calculate step sizes
        let [width, height] = val.split(":").map(Number);
        let decimalPlaces = (width.toString().split(".")[1] || []).length;
        //keep upto 6 decimal points of precision of ratio
        //euclidean gcd does not support floats, so we scale it up 
        decimalPlaces = decimalPlaces > 6 ? 6 : decimalPlaces;
        let gcd = this.gcd(width * 10 ** decimalPlaces, height * 10 ** decimalPlaces) / 10 ** decimalPlaces;
        let stepSize = 8 * height / gcd;
        let stepSizeOther = 8 * width / gcd;
        if (this.roundingSource.getVal()) {
            //If rounding is on set/keep default stepsizes
            this.widthSlider.childRangeField.step = "8";
            this.widthSlider.childRangeField.min = "64";
            this.widthSlider.childNumField.step = "8";
            this.widthSlider.childNumField.min = "64";
            this.heightSlider.childRangeField.step = "8";
            this.heightSlider.childRangeField.min = "64";
            this.heightSlider.childNumField.step = "8";
            this.heightSlider.childNumField.min = "64";
        }
        else {
            //if rounding is off, set step sizes so they enforce snapping
            //min is changed, because it offsets snap positions
            this.widthSlider.childRangeField.step = stepSizeOther.toString();
            this.widthSlider.childRangeField.min = stepSizeOther.toString();
            this.widthSlider.childNumField.step = stepSizeOther.toString();
            this.widthSlider.childNumField.min = stepSizeOther.toString();
            this.heightSlider.childRangeField.step = stepSize.toString();
            this.heightSlider.childRangeField.min = stepSize.toString();
            this.heightSlider.childNumField.step = stepSize.toString();
            this.heightSlider.childNumField.min = stepSize.toString();
        }
        let currentWidth = parseInt(this.widthSlider.getVal());
        //Rounding treated kinda like pythons divmod
        let stepsTaken = Math.round(currentWidth / stepSizeOther);
        //this snaps it to closest rule matches (rules being html step points, and ratio)
        let newWidth = stepsTaken * stepSizeOther;
        this.widthSlider.setVal(newWidth.toString());
        this.heightSlider.setVal(Math.round(newWidth / (width / height)).toString());
    }
    gcd(a, b) {
        //euclidean gcd
        if (b === 0) {
            return a;
        }
        return this.gcd(b, a % b);
    }
    static observeStartup(widthSliderId, heightSliderId, ratioSourceId, roundingSourceId, roundingMethodId) {
        let observer = new MutationObserver(() => {
            let widthSlider = document.querySelector("gradio-app").shadowRoot.getElementById(widthSliderId);
            let heightSlider = document.querySelector("gradio-app").shadowRoot.getElementById(heightSliderId);
            let ratioSource = document.querySelector("gradio-app").shadowRoot.getElementById(ratioSourceId);
            let roundingSource = document.querySelector("gradio-app").shadowRoot.getElementById(roundingSourceId);
            let roundingMethod = document.querySelector("gradio-app").shadowRoot.getElementById(roundingMethodId);
            if (widthSlider && heightSlider && ratioSource && roundingSource && roundingMethod) {
                observer.disconnect();
                new AspectRatioSliderController(widthSlider, heightSlider, ratioSource, roundingSource, roundingMethod);
            }
        });
        observer.observe(gradioApp(), { childList: true, subtree: true });
    }
}
document.addEventListener("DOMContentLoaded", () => {
    //Register mutation observer for self start-up;
    AspectRatioSliderController.observeStartup("txt2img_width", "txt2img_height", "txt2img_ratio", "setting_aspect_ratios_rounding", "setting_aspect_ratios_rounding_method");
    AspectRatioSliderController.observeStartup("img2img_width", "img2img_height", "img2img_ratio", "setting_aspect_ratios_rounding", "setting_aspect_ratios_rounding_method");
});
