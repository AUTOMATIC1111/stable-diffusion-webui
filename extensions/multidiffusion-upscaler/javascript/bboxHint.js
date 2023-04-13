const BBOX_MAX_NUM = 16;
const BBOX_WARNING_SIZE = 1280;
const DEFAULT_X = 0.4;
const DEFAULT_Y = 0.4;
const DEFAULT_H = 0.2;
const DEFAULT_W = 0.2;

// ref: https://html-color.codes/
const COLOR_MAP = [
    ['#ff0000', 'rgba(255, 0, 0, 0.3)'],          // red
    ['#ff9900', 'rgba(255, 153, 0, 0.3)'],        // orange
    ['#ffff00', 'rgba(255, 255, 0, 0.3)'],        // yellow
    ['#33cc33', 'rgba(51, 204, 51, 0.3)'],        // green
    ['#33cccc', 'rgba(51, 204, 204, 0.3)'],       // indigo
    ['#0066ff', 'rgba(0, 102, 255, 0.3)'],        // blue
    ['#6600ff', 'rgba(102, 0, 255, 0.3)'],        // purple
    ['#cc00cc', 'rgba(204, 0, 204, 0.3)'],        // dark pink
    ['#ff6666', 'rgba(255, 102, 102, 0.3)'],      // light red
    ['#ffcc66', 'rgba(255, 204, 102, 0.3)'],      // light orange
    ['#99cc00', 'rgba(153, 204, 0, 0.3)'],        // lime green
    ['#00cc99', 'rgba(0, 204, 153, 0.3)'],        // teal
    ['#0099cc', 'rgba(0, 153, 204, 0.3)'],        // steel blue
    ['#9933cc', 'rgba(153, 51, 204, 0.3)'],       // lavender
    ['#ff3399', 'rgba(255, 51, 153, 0.3)'],       // hot pink
    ['#996633', 'rgba(153, 102, 51, 0.3)'],       // brown
];

const RESIZE_BORDER = 5;
const MOVE_BORDER = 5;

const t2i_bboxes = new Array(BBOX_MAX_NUM).fill(null);
const i2i_bboxes = new Array(BBOX_MAX_NUM).fill(null);


function getUpscalerFactor() {
    const upscalerInput = parseFloat(gradioApp().querySelector('#MD-upscaler-factor input').value);
    if (!isNaN(upscalerInput)) { return upscalerInput; }
}


function updateCursorStyle(e, is_t2i, idx) {
    // This function changes the cursor style when hovering over the bounding box
    const bboxes = is_t2i ? t2i_bboxes : i2i_bboxes;
    if (!bboxes[idx]) return;

    const div = bboxes[idx][0];
    const boxRect = div.getBoundingClientRect();
    const mouseX = e.clientX;
    const mouseY = e.clientY;

    const resizeLeft   = mouseX >= boxRect.left && mouseX <= boxRect.left + RESIZE_BORDER;
    const resizeRight  = mouseX >= boxRect.right - RESIZE_BORDER && mouseX <= boxRect.right;
    const resizeTop    = mouseY >= boxRect.top && mouseY <= boxRect.top + RESIZE_BORDER;
    const resizeBottom = mouseY >= boxRect.bottom - RESIZE_BORDER && mouseY <= boxRect.bottom;

    if ((resizeLeft && resizeTop) || (resizeRight && resizeBottom)) {
        div.style.cursor = 'nwse-resize';
    } else if ((resizeLeft && resizeBottom) || (resizeRight && resizeTop)) {
        div.style.cursor = 'nesw-resize';
    } else if (resizeLeft || resizeRight) {
        div.style.cursor = 'ew-resize';
    } else if (resizeTop || resizeBottom) {
        div.style.cursor = 'ns-resize';
    } else {
        div.style.cursor = 'move';
    }
}

function displayBox(canvas, is_t2i, bbox_info) {
    // check null input
    const [div, bbox, shower] = bbox_info;
    const [x, y, w, h] = bbox;
    if (!canvas || !div || x == null || y == null || w == null || h == null) { return; }

    // client: canvas widget display size
    // natural: content image real size
    let vpScale = Math.min(canvas.clientWidth / canvas.naturalWidth, canvas.clientHeight / canvas.naturalHeight);
    let canvasCenterX = canvas.clientWidth  / 2;
    let canvasCenterY = canvas.clientHeight / 2;
    let scaledX = canvas.naturalWidth  * vpScale;
    let scaledY = canvas.naturalHeight * vpScale;
    let viewRectLeft  = canvasCenterX - scaledX / 2;
    let viewRectRight = canvasCenterX + scaledX / 2;
    let viewRectTop   = canvasCenterY - scaledY / 2;
    let viewRectDown  = canvasCenterY + scaledY / 2;

    let xDiv = viewRectLeft + scaledX * x;
    let yDiv = viewRectTop  + scaledY * y;
    let wDiv = Math.min(scaledX * w, viewRectRight - xDiv);
    let hDiv = Math.min(scaledY * h, viewRectDown - yDiv);

    // Calculate warning bbox size
    let upscalerFactor = 1.0;
    if (!is_t2i) {
        upscalerFactor = getUpscalerFactor();
    }
    let maxSize = BBOX_WARNING_SIZE / upscalerFactor * vpScale;
    let maxW = maxSize / scaledX;
    let maxH = maxSize / scaledY;
    if (w > maxW || h > maxH) {
        div.querySelector('span').style.display = 'block';
    } else {
        div.querySelector('span').style.display = 'none';
    }

    // update <div> when not equal
    div.style.left    = xDiv + 'px';
    div.style.top     = yDiv + 'px';
    div.style.width   = wDiv + 'px';
    div.style.height  = hDiv + 'px';
    div.style.display = 'block';

    // insert it to DOM if not appear
    shower();
}

function onBoxEnableClick(is_t2i, idx, enable) {
    let canvas = null;
    let bboxes = null;
    let locator = null;
    if (is_t2i) {
        locator = () => gradioApp().querySelector('#MD-bbox-ref-t2i');
        bboxes = t2i_bboxes;
    } else {
        locator = () => gradioApp().querySelector('#MD-bbox-ref-i2i');
        bboxes = i2i_bboxes;
    }
    ref_div = locator();
    canvas = ref_div.querySelector('img');
    if (!canvas) { return false; }

    if (enable) {
        // Check if the bounding box already exists
        if (!bboxes[idx]) {
            // Initialize bounding box
            const bbox = [DEFAULT_X, DEFAULT_Y, DEFAULT_W, DEFAULT_H];
            const colorMap = COLOR_MAP[idx % COLOR_MAP.length];
            const div = document.createElement('div');
            div.id = 'MD-bbox-' + (is_t2i ? 't2i-' : 'i2i-') + idx;
            div.style.left       = '0px';
            div.style.top        = '0px';
            div.style.width      = '0px';
            div.style.height     = '0px';
            div.style.position   = 'absolute';
            div.style.border     = '2px solid ' + colorMap[0];
            div.style.background = colorMap[1];
            div.style.zIndex     = '900';
            div.style.display    = 'none';
            // A text tip to warn the user if bbox is too large
            const tip = document.createElement('span');
            tip.id = 'MD-tip-' + (is_t2i ? 't2i-' : 'i2i-') + idx;
            tip.style.left       = '50%';
            tip.style.top        = '50%';
            tip.style.position   = 'absolute';
            tip.style.transform  = 'translate(-50%, -50%)';
            tip.style.fontSize   = '12px';
            tip.style.fontWeight = 'bold';
            tip.style.textAlign  = 'center';
            tip.style.color      = colorMap[0];
            tip.style.zIndex     = '901';
            tip.style.display    = 'none';
            tip.innerHTML        = 'Warning: Region very large!<br>Take care of VRAM usage!';
            div.appendChild(tip);

            div.addEventListener('mousedown', function (e) {
                if (e.button === 0) { onBoxMouseDown(e, is_t2i, idx); }
            });
            div.addEventListener('mousemove', function (e) {
                updateCursorStyle(e, is_t2i, idx);
            });

            const shower = function() { // insert to DOM if necessary
                if (!gradioApp().querySelector('#' + div.id)) {
                    locator().appendChild(div);
                }
            }
            bboxes[idx] = [div, bbox, shower];
        }

        // Show the bounding box
        displayBox(canvas, is_t2i, bboxes[idx]);
        return true;
    } else {
        if (!bboxes[idx]) { return false; }
        const [div, bbox, shower] = bboxes[idx];
        div.style.display = 'none';
    }
    return false;
}

function onBoxChange(is_t2i, idx, what, v) {
    // This function handles all the changes of the bounding box
    // Including the rendering and python slider update
    let bboxes = null;
    let canvas = null;
    if (is_t2i) {
        bboxes = t2i_bboxes;
        canvas = gradioApp().querySelector('#MD-bbox-ref-t2i img');
    } else {
        bboxes = i2i_bboxes;
        canvas = gradioApp().querySelector('#MD-bbox-ref-i2i img');
    }
    if (!bboxes[idx] || !canvas) {
        switch (what) {
            case 'x': return DEFAULT_X;
            case 'y': return DEFAULT_Y;
            case 'w': return DEFAULT_W;
            case 'h': return DEFAULT_H;
        }
    }
    const [div, bbox, shower] = bboxes[idx];
    if (div.style.display === 'none') { return v; }

    // parse trigger
    switch (what) {
        case 'x': bbox[0] = v; break;
        case 'y': bbox[1] = v; break;
        case 'w': bbox[2] = v; break;
        case 'h': bbox[3] = v; break;
    }
    displayBox(canvas, is_t2i, bboxes[idx]);
    return v;
}

function onBoxMouseDown(e, is_t2i, idx) {
    let bboxes = null;
    let canvas = null;
    if (is_t2i) {
        bboxes = t2i_bboxes;
        canvas = gradioApp().querySelector('#MD-bbox-ref-t2i img');
    } else {
        bboxes = i2i_bboxes;
        canvas = gradioApp().querySelector('#MD-bbox-ref-i2i img');
    }
    // Get the bounding box
    if (!canvas || !bboxes[idx]) { return; }
    const [div, bbox, shower] = bboxes[idx];

    // Check if the click is inside the bounding box
    const boxRect = div.getBoundingClientRect();
    let mouseX = e.clientX;
    let mouseY = e.clientY;

    const resizeLeft   = mouseX >= boxRect.left && mouseX <= boxRect.left + RESIZE_BORDER;
    const resizeRight  = mouseX >= boxRect.right - RESIZE_BORDER && mouseX <= boxRect.right;
    const resizeTop    = mouseY >= boxRect.top && mouseY <= boxRect.top + RESIZE_BORDER;
    const resizeBottom = mouseY >= boxRect.bottom - RESIZE_BORDER && mouseY <= boxRect.bottom;

    const moveHorizontal = mouseX >= boxRect.left + MOVE_BORDER && mouseX <= boxRect.right  - MOVE_BORDER;
    const moveVertical   = mouseY >= boxRect.top  + MOVE_BORDER && mouseY <= boxRect.bottom - MOVE_BORDER;

    if (!resizeLeft && !resizeRight && !resizeTop && !resizeBottom && !moveHorizontal && !moveVertical) { return; }

    const horizontalPivot = resizeLeft ? bbox[0] + bbox[2] : bbox[0];
    const verticalPivot   = resizeTop  ? bbox[1] + bbox[3] : bbox[1];

    // Canvas can be regarded as invariant during the drag operation
    // Calculate in advance to reduce overhead

    // Calculate viewport scale based on the current canvas size and the natural image size
    let vpScale = Math.min(canvas.clientWidth / canvas.naturalWidth, canvas.clientHeight / canvas.naturalHeight);
    let vpOffset = canvas.getBoundingClientRect();

    // Calculate scaled dimensions of the canvas
    let scaledX = canvas.naturalWidth * vpScale;
    let scaledY = canvas.naturalHeight * vpScale;

    // Calculate the canvas center and view rectangle coordinates
    let canvasCenterX = (vpOffset.left + window.scrollX) + canvas.clientWidth  / 2;
    let canvasCenterY = (vpOffset.top  + window.scrollY) + canvas.clientHeight / 2;
    let viewRectLeft  = canvasCenterX - scaledX / 2 - window.scrollX;
    let viewRectRight = canvasCenterX + scaledX / 2 - window.scrollX;
    let viewRectTop   = canvasCenterY - scaledY / 2 - window.scrollY;
    let viewRectDown  = canvasCenterY + scaledY / 2 - window.scrollY;

    mouseX = Math.min(Math.max(mouseX, viewRectLeft), viewRectRight);
    mouseY = Math.min(Math.max(mouseY, viewRectTop),  viewRectDown);

    const accordion = gradioApp().querySelector(`#MD-accordion-${is_t2i ? 't2i' : 'i2i'}-${idx}`);

    // Move or resize the bounding box on mousemove
    function onMouseMove(e) {
        // Prevent selecting anything irrelevant
        e.preventDefault();

        // Get the new mouse position
        let newMouseX = e.clientX;
        let newMouseY = e.clientY;

        // clamp the mouse position to the view rectangle
        newMouseX = Math.min(Math.max(newMouseX, viewRectLeft), viewRectRight);
        newMouseY = Math.min(Math.max(newMouseY, viewRectTop),  viewRectDown);

        // Calculate the mouse movement delta
        const dx = (newMouseX - mouseX) / scaledX;
        const dy = (newMouseY - mouseY) / scaledY;

        // Update the mouse position
        mouseX = newMouseX;
        mouseY = newMouseY;

        // if no move just return
        if (dx === 0 && dy === 0) { return; }

        // Update the mouse position
        let [x, y, w, h] = bbox;
        if (moveHorizontal && moveVertical) {
            // If moving the bounding box
            x = Math.min(Math.max(x + dx, 0), 1 - w);
            y = Math.min(Math.max(y + dy, 0), 1 - h);
        } else {
            // If resizing the bounding box
            if (resizeLeft || resizeRight) {
                if (x < horizontalPivot) {
                    if (dx <= w) {
                        // If still within the left side of the pivot
                        x = x + dx;
                        w = w - dx;
                    } else {
                        // If crossing the pivot
                        w = dx - w;
                        x = horizontalPivot;
                    }
                } else {
                    if (w + dx < 0) {
                        // If still within the right side of the pivot
                        x = horizontalPivot + w + dx;
                        w = - dx - w;
                    } else {
                        // If crossing the pivot
                        x = horizontalPivot;
                        w = w + dx;
                    }
                }

                // Clamp the bounding box to the image
                if (x < 0) {
                    w = w + x;
                    x = 0;
                } else if (x + w > 1) {
                    w = 1 - x;
                }
            }
            // Same as above, but for the vertical axis
            if (resizeTop || resizeBottom) {
                if (y < verticalPivot) {
                    if (dy <= h) {
                        y = y + dy;
                        h = h - dy;
                    } else {
                        h = dy - h;
                        y = verticalPivot;
                    }
                } else {
                    if (h + dy < 0) {
                        y = verticalPivot + h + dy;
                        h = - dy - h;
                    } else {
                        y = verticalPivot;
                        h = h + dy;
                    }
                }
                if (y < 0) {
                    h = h + y;
                    y = 0;
                } else if (y + h > 1) {
                    h = 1 - y;
                }
            }
        }
        const [div, old_bbox, _] = bboxes[idx];

        // If all the values are the same, just return
        if (old_bbox[0] === x && old_bbox[1] === y && old_bbox[2] === w && old_bbox[3] === h) { return; }
        // else update the bbox
        const event = new Event('input');
        const coords = [x, y, w, h];
        // <del>The querySelector is not very efficient, so we query it once and reuse it</del>
        // caching will result gradio bugs that stucks bbox and cannot move & drag
        const sliderIds = ['x', 'y', 'w', 'h'];
        // We try to select the input sliders
        const sliderSelectors = sliderIds.map(id => `#MD-${is_t2i ? 't2i' : 'i2i'}-${idx}-${id} input`).join(', ');
        let sliderInputs = accordion.querySelectorAll(sliderSelectors);
        if (sliderInputs.length == 0) {
            // If we failed, the accordion is probably closed and sliders are removed in the dom, so we open it
            accordion.querySelector('.label-wrap').click();
            // and try again
            sliderInputs = accordion.querySelectorAll(sliderSelectors);
            // If we still failed, we just return
            if (sliderInputs.length == 0) { return; }
        }
        for (let i = 0; i < 4; i++) {
            if (old_bbox[i] !== coords[i]) {
                sliderInputs[2*i].value = coords[i];
                sliderInputs[2*i].dispatchEvent(event);
            }
        }
    }

    // Remove the mousemove and mouseup event listeners
    function onMouseUp() {
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup',   onMouseUp);
    }

    // Add the event listeners
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup',   onMouseUp);
}


function onCreateT2IRefClick(overwrite) {
    let width, height;
    if (overwrite) {
        const overwriteInputs = gradioApp().querySelectorAll('#MD-overwrite-width-t2i input, #MD-overwrite-height-t2i input');
        width  = parseInt(overwriteInputs[0].value);
        height = parseInt(overwriteInputs[2].value);
    } else {
        const sizeInputs = gradioApp().querySelectorAll('#txt2img_width input, #txt2img_height input');
        width  = parseInt(sizeInputs[0].value);
        height = parseInt(sizeInputs[2].value);
    }

    if (isNaN(width))  width  = 512;
    if (isNaN(height)) height = 512;

    // Concat it to string to bypass the gradio bug
    // 向黑恶势力低头
    return width.toString() + 'x' + height.toString();
}

function onCreateI2IRefClick() {
    const canvas = gradioApp().querySelector('#img2img_image img');
    return canvas.src;
}

function onBoxLocked(v) {
    return v
}


function updateBoxes(is_t2i) {
    // This function redraw all bounding boxes
    let bboxes = null;
    let canvas = null;
    if (is_t2i) {
        bboxes = t2i_bboxes;
        canvas = gradioApp().querySelector('#MD-bbox-ref-t2i img');
    } else {
        bboxes = i2i_bboxes;
        canvas = gradioApp().querySelector('#MD-bbox-ref-i2i img');
    }
    if (!canvas) return;

    for (let idx = 0; idx < bboxes.length; idx++) {
        if (!bboxes[idx]) continue;
        const [div, bbox, shower] = bboxes[idx];
        if (div.style.display === 'none') { return; }

        displayBox(canvas, is_t2i, bboxes[idx]);
    }
}

function getSeedInfo(is_t2i, id, current_seed){
    const info_id = is_t2i ? '#html_info_txt2img' : '#html_info_img2img';
    const info_div = gradioApp().querySelector(info_id);
    try{
        current_seed = parseInt(current_seed);
    }catch(e){
        current_seed = -1;
    }
    if (!info_div) return current_seed;
    let info = info_div.innerHTML;
    if (!info) return current_seed;
    // remove all html tags
    info = info.replace(/<[^>]*>/g, '');
    // Find a json string 'region control:' in the info
    // get its index
    idx = info.indexOf('Region control');
    if (idx == -1) return current_seed;
    // get the json string (detect the bracket)
    // find the first '{'
    let start_idx = info.indexOf('{', idx);
    let bracket = 1;
    let end_idx = start_idx + 1;
    while (bracket > 0 && end_idx < info.length) {
        if (info[end_idx] == '{') bracket++;
        if (info[end_idx] == '}') bracket--;
        end_idx++;
    }
    if (bracket > 0) {
        return current_seed;
    }
    // get the json string
    let json_str = info.substring(start_idx, end_idx);
    // replace the single quote to double quote
    json_str = json_str.replace(/'/g, '"');
    // replace python True to javascript true, False to false
    json_str = json_str.replace(/True/g, 'true');
    // parse the json string
    let json = JSON.parse(json_str);
    // get the seed if the region id is in the json
    const region_id = 'Region ' + id.toString();
    if (!(region_id in json)) return current_seed;
    const region = json[region_id];
    if (!('seed' in region)) return current_seed;
    let seed = region['seed'];
    try{
        seed = parseInt(seed);
    }catch(e){
        return current_seed;
    }
    return seed;
}

window.addEventListener('resize', _ => {
    updateBoxes(true);
    updateBoxes(false);
});

window.addEventListener('DOMNodeInserted', e => {
    if (!e) { return; }
    if (!e.target) { return; }
    if (!e.target.classList) { return; }
    if (!e.target.classList.contains('label-wrap')) { return; }

    for (let tab of ['t2i', 'i2i']) {
        const div = gradioApp().querySelector('#MD-bbox-control-' + tab +' div.label-wrap');
        if (!div) { continue; }

        updateBoxes(tab === 't2i');
    }
});
