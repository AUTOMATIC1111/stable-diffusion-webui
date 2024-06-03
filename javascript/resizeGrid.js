/** resizeGrid.js
 *
 *  This allows generation of a grid of items which are separated by resizable handles.
 *
 *  Generation of this grid is inferred from HTML.
 *  An example can be found in `javascript/resizeGridExample.html`.
 *  This example will generate a 2x2 grid with 2 rows and 2 cells per row.
 *  The top right and bottom left cells should fill the majority of their rows.
 *  Play around with the buttons to show/hide and drag to resize the grid.
 *
 *  USAGE:
 *  Current limitations require that every row and column contain AT LEAST ONE
 *  child with flexGrow=1. CSS shortcuts are allowed (i.e. flex: 1 0 0px).
 *
 *  You CANNOT have rows and columns at the same level. This will throw an error.
 *
 *  flexBasis MUST be set for every item that does not have flexGrow=1. flexBasis can be
 *  set using `px` or any of the following relative units: [vh, vw, rem, em]. These
 *  units are limited by `utils.js::cssRelativeUnitToPx()`.
 *
 *  The value set in flexGrow determines an item's size along its respective axis.
 *  So if you have a `resize-grid--col` item, then `flexBasis` will determine the item's
 *  height. `resize-grid--cell` items infer their axis from their parent element.
 *
 *  You can also set a `data-min-size` attribute on every row/col/cell. If this attribute
 *  is not set, then the value set in flexBasis will be the minimum size of the item.
 *  If you are setting the flexBasis to 0, you MUST specify data-min-size="0px" as well
 *  otherwise the item's min_size will be set to its rendered size on build.
 *
 *  If you do not set any `resize-grid--row` or `resize-grid--col` elements, but only
 *  set `resize-grid--cell` elements, then the grid will automatically wrap the cells
 *  in a single row.
 *
 *  You can have a grid that only contains one row/col as long as that row/col contains
 *  at least one item.
*/

// Prevent eslint errors on functions defined in other files.
/*global
    isNullOrUndefinedThrowError,
    isNullOrUndefinedLogError,
    isNullOrUndefined,
    isString,
    isStringThrowError,
    isObject,
    isFunction,
    cssRelativeUnitToPx,
    isNumber,
    isNumberThrowError,
    isElementThrowError,
    isElement,
*/
/*eslint no-undef: "error"*/

// Should be between 0 and 15. Any higher and the delay becomes noticable.
// Higher values reduce computational load.
const MOVE_TIME_DELAY_MS = 15;
// Prevents handling element resize events too quickly. Lower values increase
// computational load and may lead to lag when resizing.
const RESIZE_DEBOUNCE_TIME_MS = 50;
// The timeframe in which a second pointerup event must be fired to be treated
// as a double click.
const DBLCLICK_TIME_MS = 500;
// The padding around the draggable resize handle.
// NOTE: Must be an even number.
const PAD_PX = 16;
if (!(PAD_PX > 0 && PAD_PX % 2 === 0)) {
    throw new Error('PAD_PX must be an even number > 0');
}

const resize_grids = {};

/* ==== HELPER FUNCTIONS ==== */
const _gen_id_string = () => {
    return Math.random().toString(16).slice(2);
};

const _get_unique_id = () => {
    /** Generates an ID string that does not exist in the `resize_grids` keys. */
    let id = _gen_id_string();
    while (id in Object.keys(resize_grids)) {
        id = _gen_id_string();
    }
    return id;
};

const _axis_to_int = (axis) => {
    /** Converts an axis to a standardized axis integer.
     *  Args:
     *      axis: Integer or string to be parsed.
     *  Returns:
     *      "x" or 0: 0
     *      "y" or 1: 1
     *  Throws:
     *      Error if the `axis` input is invalid.
     */
    if (axis === 0 || axis === 'x') {
        return 0;
    } else if (axis === 1 || axis === 'y') {
        return 1;
    } else {
        throw new Error(`"Axis" expected (x (0), y (1)), got: ${axis}`);
    }
};

class ResizeGridHandle {
    /** The clickable "handle" between two ResizeGridItem instances. */
    visible = true;
    id = null; // unique identifier for this instance.
    pad_px = PAD_PX;
    constructor({id, parent, axis} = {}) {
        this.id = isNullOrUndefined(id) ? _gen_id_string() : id;
        this.parent = parent;
        this.elem = document.createElement('div');
        this.elem.id = id;
        this.elem.dataset.id = id;
        this.axis = _axis_to_int(axis);
        isNumberThrowError(this.axis);
        this.elem.classList.add('resize-grid--handle');
        if (this.axis === 0) {
            this.elem.classList.add("resize-grid--row-handle");
            this.elem.style.minHeight = this.pad_px + 'px';
            this.elem.style.maxHeight = this.pad_px + 'px';
        } else if (this.axis === 1) {
            this.elem.classList.add("resize-grid--col-handle");
            this.elem.style.minWidth = this.pad_px + 'px';
            this.elem.style.maxWidth = this.pad_px + 'px';
        }
    }

    destroy() {
        this.elem.remove();
    }

    show() {
        this.elem.classList.remove('hidden');
        this.visible = true;
    }

    hide() {
        this.elem.classList.add('hidden');
        this.visible = false;
    }
}

class ResizeGridItem {
    /** Grid elements. These can be axes or individual cells.
     *
     *  Attributes:
     *      id (str):
     *          A unique identifier for this instance and its element.
     *      parent (ResizeGridAxis,ResizeGrid):
     *          The container class for this item. Cannot be a base ResizeGridItem.
     *      elem (Element):
     *          The DOM element representing this item.
     *      callbacks (Object):
     *          Object specifying callbacks for various operations in this class.
     *      axis (int):
     *          The axis along which this item lies. 0: row, 1: col.
     *          This value is inferred from the passed element's class list.
     *      is_flex_grow (bool):
     *          Whether this item should auto expand along its axis.
     *          The actual elem.style.flexGrow may change independent of this variable
     *          but this variable is used to determine the default flexGrow.
     *      is_flex_shrink (bool):
     *          Whether this item should auto shrink along its axis.
     *          Same behavior as `is_flex_grow`.
     *      min_size (int):
     *          The minimum size of the element.
     *      base_size (int):
     *          The default size of the element.
     *      original_css_text (str):
     *          The elem.style.cssText string that the element had on initialization.
     *          Used during destruction of this instance to reset the element.
    */
    handle = null;
    visible = true;
    callbacks = {
        /** Allows user to modify the `size_px` value passed to setSize().
         *  Default callback just returns null. (no op).
         *  Args:
         *      item [ResizeGridItem]: This class instance.
         *      size_px [int]: The size (in px) to be overridden.
         *  Returns:
         *      int: The overridden setSize `size_px` paremeter. If null/undefined,
         *          then `size_px` does not get modified.
         */
        set_size_override: (item, size_px) => {
            return;
        },
    };
    constructor({id, parent, elem, axis, callbacks} = {}) {
        if (elem.id) {
            this.id = elem.id;
        } else {
            this.id = isNullOrUndefined(id) ? _gen_id_string() : id;
        }
        this.parent = parent;
        this.elem = elem;
        if (!isNullOrUndefined(axis)) {
            this.axis = _axis_to_int(axis);
        } else if (elem.classList.contains("resize-grid--row")) {
            this.axis = 0;
        } else if (elem.classList.contains("resize-grid--col")) {
            this.axis = 1;
        } else {
            throw new Error("Unable to infer axis from element:", elem);
        }

        // Parse user specified callback overrides.
        if (isObject(callbacks)) {
            if (isFunction(callbacks.set_size_override)) {
                this.callbacks.set_size_override = callbacks.set_size_override;
            }
        }

        this.is_flex_grow = Boolean(parseInt(this.elem.style.flexGrow));
        this.is_flex_shrink = Boolean(parseInt(this.elem.style.flexShrink));
        let flex_basis = parseInt(cssRelativeUnitToPx(this.elem.style.flexBasis));
        if (isNumber(flex_basis) && flex_basis > 0) {
            this.base_size = flex_basis;
        } else {
            const dims = this.elem.getBoundingClientRect();
            this.base_size = parseInt(this.axis === 0 ? dims.height : dims.width);
        }
        this.min_size = this.base_size;

        // If data-min-size is set, use that for the min_size instead.
        if ("minSize" in this.elem.dataset) {
            this.min_size = parseInt(cssRelativeUnitToPx(this.elem.dataset.minSize));
        }

        this.elem.dataset.id = this.id;
        this.original_css_text = this.elem.style.cssText;
    }

    render({force_flex_grow, force_flex_shrink, reset} = {}) {
        /** Sets the element's flex styles.
         *
         *  If no arguments are passed, then flexGrow is reset to default and the
         *  flexBasis is set to the current calculated size of the element
         *  (clamped to min_size).
         *
         *  Args:
         *      force_flex_grow (bool):
         *          Sets flexGrow=1 and does nothing else.
         *      force_flex_shrink (bool):
         *          Sets flexShrink=1 and does nothing else.
         *      reset (bool):
         *          Sets flexGrow and flexBasis to the instance defaults.
        */
        if (!this.visible) {
            return;
        }

        force_flex_grow = force_flex_grow === true;
        force_flex_shrink = force_flex_shrink === true;
        reset = reset === true;

        const vis_items = this.parent.items.filter(x => x.visible);
        if (vis_items.length === 1) {
            force_flex_grow = true;
            force_flex_shrink = true;
        }

        if (force_flex_grow || force_flex_shrink) {
            this.elem.style.flexGrow = force_flex_grow ? 1 : parseInt(this.elem.style.flexGrow);
            this.elem.style.flexShrink = force_flex_shrink ? 1 : parseInt(this.elem.style.flexShrink);
        } else if (reset) {
            this.elem.style.flexGrow = Number(this.is_flex_grow);
            this.elem.style.flexShrink = Number(this.is_flex_shrink);
            this.elem.style.flexBasis = this.base_size + 'px';
        } else {
            this.elem.style.flexGrow = Number(this.is_flex_grow);
            this.elem.style.flexShrink = Number(this.is_flex_shrink);
            this.elem.style.flexBasis = Math.max(this.min_size, this.getSize()) + 'px';
        }
    }

    destroy() {
        if (!isNullOrUndefined(this.handle)) {
            this.handle.destroy();
            this.handle = null;
        }
        // Revert changes to the container element.
        this.elem.style.cssText = this.original_css_text;
        if (!this.elem.style.cssText) {
            this.elem.removeAttribute('style');
        }
        delete this.elem.dataset.id;
        delete this.elem.dataset.index;
    }

    shrink(px, {limit_to_base} = {}) {
        /** Shrink size along axis by specified pixels. Returns remainder.
         *
         *  Args:
         *      px (int):
         *          The number of pixels to shrink by.
         *          If -1, then the item is shrunk to the `base_size` or `min_size`
         *          depending on the value of `limit_to_base`.
         *      limit_to_base (bool):
         *          Whether to use the `base_size` as the minimum size after shrinking.
         *          If not specified or false, then `min_size` is used.
         *
         *  Returns:
         *      int: The number of pixels remaining that could not be shrunk.
        */
        if (px <= 0) {
            return 0;
        }

        limit_to_base = limit_to_base === true;
        const target_size = limit_to_base ? this.base_size : this.min_size;
        const curr_size = this.getSize();

        if (px === -1) {
            // Shrink to target regardless of the current size.
            this.setSize(target_size);
            return 0;
        } else if (curr_size <= target_size) {
            // This can happen if using base_size instead of min_size and the item is
            // manually resized smaller than base_size. Cannot shrink, make no changes.
            return px;
        } else if (curr_size - target_size < px) {
            // Can shrink but not to the requested amount. Return remainder.
            this.setSize(target_size);
            return px - (curr_size - target_size);
        } else {
            // Can shrink the full requested amount.
            this.setSize(curr_size - px);
            return 0;
        }
    }

    grow(px, {only_if_flex} = {}) {
        /** Grows along axis and returns the amount grown in pixels.
         *
         *  Args:
         *      px (int):
         *          The number of pixels to grow by.
         *          If -1, then the item grows to fill its container.
         *      only_if_flex (bool):
         *          If true, then grow is only performed if is_flex_grow=true.
         *
         *  Returns:
         *      int: The number of pixels that the item grew by.
        */
        only_if_flex = only_if_flex === true;
        if (only_if_flex && !this.is_flex_grow) {
            return 0;
        }
        let new_size;
        const curr_size = this.getSize();
        if (px === -1) {
            // grow to fill container (only works if visible)
            // set flexGrow to 1 to expand to max width so we can calc new width.
            this.elem.style.flexGrow = 1;
            new_size = this.getSize();
            this.elem.style.flexGrow = Number(this.is_flex_grow);
        } else {
            new_size = curr_size + px;
        }
        this.setSize(new_size);
        return new_size - curr_size;
    }

    setSize(size_px) {
        /** Sets the flexBasis value for the item.
         *
         *  Prior to setting flexBasis, the `set_size_override` callback is called.
         *  If this callback returns a valid number, then we use that value instead
         *  of `size_px` for the new size.
         *
         *  Args:
         *      size_px (int): The new size (in pixels).
         *
         *  Returns:
         *      int: The size that we just set.
        */
        const new_size_px = this.callbacks.set_size_override(this, size_px);
        if (isNumber(new_size_px)) {
            size_px = new_size_px;
        }
        this.elem.style.flexBasis = parseInt(size_px) + 'px';
        return size_px;
    }

    getSize() {
        /** Returns the current size of this item.
         *
         *  Returns:
         *      int: If this item is visible in the DOM, then we return the computed
         *          dimensions of the element. Otherwise we return the element's
         *          inline style flexBasis value.
         */
        if (this.visible) {

            const dims = this.elem.getBoundingClientRect();
            return this.axis === 0 ? parseInt(dims.height) : parseInt(dims.width);
        } else {
            return parseInt(this.elem.style.flexBasis);
        }
    }

    genHandle() {
        /** Generates a ResizeGridHandle for this item and returns the new handle. */
        this.handle = new ResizeGridHandle({
            id: `${this.id}_handle`,
            parent: this.parent,
            axis: this.axis,
        });
        if (isElement(this.elem.nextElementSibling)) {
            this.elem.parentElement.insertBefore(
                this.handle.elem,
                this.elem.nextSibling
            );
        } else {
            this.elem.parentElement.appendChild(this.handle.elem);
        }
        return this.handle;
    }

    show() {
        /** Shows this item and its ResizeGridHandle. */
        this.elem.classList.remove('hidden');
        // Only show the handle if it isnt the last visible item along its axis.
        const siblings = this.parent.items;
        let sibling = siblings.slice(siblings.indexOf(this) + 1).find(x => x.visible);
        if (sibling instanceof ResizeGridItem) {
            this.handle.show();
            sibling.render();
        } else {
            sibling = siblings.slice(0, siblings.indexOf(this)).find(x => x.visible);
            if (sibling instanceof ResizeGridItem) {
                sibling.handle.show();
                sibling.render();
            }
        }
        this.visible = true;
    }

    hide() {
        /** Hides this item and its ResizeGridHandle. */
        this.elem.classList.add('hidden');
        this.handle.hide();
        this.visible = false;
    }
}

class ResizeGridAxis extends ResizeGridItem {
    /** Represents a collection of ResizeGridItems along a single axis.
     *
     *  Attributes:
     *      items (Array[ResizeGridItem]):
     *          The items contained within this axis.
     *      item_ids (Object[str, ResizeGridItem]):
     *          Mapping of item IDs to their ResizeGridItem instance.
    */
    constructor(...args) {
        super(...args);
        this.items = [];
        this.item_ids = {};
    }

    destroy() {
        this.items.forEach(item => item.destroy());
        this.items = [];
        this.item_ids = {};
        super.destroy();
    }

    addCell(id, elem, idx) {
        /** Creates a ResizeGridItem along this axis and returns the new item. */
        const item = new ResizeGridItem({
            id: id,
            parent: this,
            elem: elem,
            axis: this.axis ^ 1, // Children are along the opposite axis of this.
            callbacks: this.callbacks,
        });
        item.genHandle();
        item.elem.dataset.index = idx;
        this.items.push(item);
        this.item_ids[id] = item;
        return item;
    }

    render({force_flex_grow, reset} = {}) {
        /** Renders all children items and this item. */
        if (!this.visible) {
            return;
        }

        this.items.forEach(item => {
            item.render({force_flex_grow: force_flex_grow, reset: reset});
        });
        // If we only have one visible cell, we need to force it to grow to fill axis.
        const visible_cells = this.items.filter(x => x.visible);
        if (visible_cells.length === 1) {
            visible_cells[0].render({force_flex_grow: true});
        }

        if (!(this instanceof ResizeGrid)) {
            super.render({force_flex_grow: force_flex_grow, reset: reset});
        }

        this.updateVisibleHandles();
    }

    getById(id) {
        isStringThrowError(id);
        if (id in this.item_ids) {
            return this.item_ids[id];
        }
        throw new Error(`No matching Cell ID in ResizeGridAxis: ${id}`);
    }

    getByElem(elem) {
        isElementThrowError(elem);
        elem = elem.closest(".resize-grid--cell,.resize-grid--col,.resize-grid--row");
        isElementThrowError(elem);
        return this.getById(elem.dataset.id);
    }

    getByIdx(idx) {
        isNumberThrowError(idx);
        const res = this.items[idx];
        if (!isNullOrUndefined(res)) {
            return res;
        }
        throw new Error(`Invalid Cell Index in ResizeGridAxis: ${idx}`);
    }

    getItem({id, idx, elem} = {}) {
        if (isString(id)) {
            return this.getById(id);
        } else if (isNumber(idx)) {
            return this.getByIdx(idx);
        } else if (isElement(elem)) {
            return this.getByElem(elem);
        } else {
            // Indicates programmer error.
            throw new Error("Invalid arguments. Must specify one of [id, idx, elem].");
        }
    }

    updateVisibleHandles() {
        /** Sets the visibility of each ResizeGridHandle based on surrounding items. */
        for (const item of this.items) {
            if (item.visible) {
                item.handle.show();
            } else {
                item.handle.hide();
            }
        }
        const last_vis = this.items[this.items.findLastIndex(x => x.visible)];
        if (last_vis instanceof ResizeGridItem) {
            last_vis.handle.hide();
        }
    }

    makeRoomForItem(item, siblings, {use_base_size} = {}) {
        /** Shrinks items along this axis until the passed item can fit.
         *
         *  Args:
         *      item (ResizeGridItem):
         *          The item to be added into this axis.
         *      siblings (Array[ResizeGridItem]):
         *          An array of ResizeGridItems within the same container as `item`.
         *      use_base_size (bool):
         *          Whether to use the `item`'s base_size or current size when inserting.
         *
         *  Throws:
         *      Error: If unable to shrink items along this axis to make room for `item`.
         */
        isNullOrUndefinedThrowError(item);
        const idx = siblings.indexOf(item);
        isNumberThrowError(idx);
        use_base_size = use_base_size === true;
        // If use_base_size=false, then try to get the item's current size and use that.
        // Since the item is hidden, this will get the size saved in flexBasis which was
        // the previously set size.
        const tot = (use_base_size ? item.base_size : item.getSize()) + item.handle.pad_px;
        let rem = tot;

        // Get first visible sibling after the item we're trying to add.
        let sibling = siblings.slice(idx + 1).find(x => x.visible);
        const sibling_idx = siblings.indexOf(sibling);

        // Shrink from the sibling first.
        if (sibling instanceof ResizeGridItem) {
            rem = sibling.shrink(rem, {limit_to_base: false});
        }
        if (rem <= 0) {
            return;
        }

        // Shrink all other items next, starting from the end.
        let others = siblings.filter((x, i) => x.visible && i !== idx && i !== sibling_idx);
        for (const other of others.slice().reverse()) {
            rem = other.shrink(rem, {limit_to_base: false});
            if (rem <= 0) {
                return;
            }
        }

        // Finally, shrink from the item itself since it is too big to insert.
        rem = item.shrink(rem, {limit_to_base: false});
        if (rem <= 0) {
            return;
        }

        // This indicates a programmer error.
        throw new Error(`No space for item. tot: ${tot}, rem: ${rem}`);
    }

    growToFill(item, siblings) {
        /** Grows an item along an axis to fill its container.
         *
         *  Args:
         *      item (ResizeGridItem):
         *          The item to grow.
         *      siblings (Array[ResizeGridItem]):
         *          An array of ResizeGridItems within the same container as `item`.
        */
        const idx = siblings.indexOf(item);
        isNullOrUndefinedThrowError(item); // Indicates programmer error.
        let sibling = siblings.slice(idx + 1).find(x => x.visible);
        if (isNullOrUndefined(sibling)) {
            sibling = siblings.slice(0, idx).findLast(x => x.visible);
            isNullOrUndefinedThrowError(sibling); // Indicates programmer error.
        }

        const sibling_idx = siblings.indexOf(sibling);

        // Hide sibling's handle if sibling is the last visible item.
        if (siblings.slice(sibling_idx + 1).every(x => !x.visible) && sibling.handle.visible) {
            sibling.handle.hide();
        }

        // If we are growing sibling to fill, then just set flexGrow=1.
        const others = siblings.filter(x => x !== sibling && x !== item);
        if (siblings.length <= 2 || others.every(x => !x.visible)) {
            sibling.render({force_flex_grow: true});
        } else {
            sibling.grow(-1);
        }
    }

    resizeItem(item, size_px) {
        // Don't resize invisible items.
        if (!item.visible) {
            return;
        }
        // Don't resize item if it is the only visible item in the axis.
        if (this.items.filter(x => x.visible).length === 1) {
            return;
        }

        if (size_px < item.min_size) {
            console.error(`Requested size is too small: ${size_px} < ${item.min_size}`);
            return;
        }

        const dims = this.elem.getBoundingClientRect();
        let max_size = parseInt(this.axis === 0 ? dims.width : dims.height);
        const vis_siblings = this.items.filter(x => x.visible && x !== item);
        max_size -= vis_siblings.reduce((acc, obj) => {
            return acc + obj.min_size + (obj.handle.visible ? obj.handle.pad_px : 0);
        }, 0);

        if (size_px > max_size) {
            console.error(`Requested size is too large: ${size_px} > ${max_size}`);
            return;
        }

        // Find the direct sibling of this item.
        const idx = this.items.indexOf(item);
        isNullOrUndefinedThrowError(item); // Indicates programmer error.
        // Look after item.
        let sibling = this.items.slice(idx + 1).find(x => x.visible);
        if (isNullOrUndefined(sibling)) {
            // No valid siblings after item, look before item.
            sibling = this.items.slice(0, idx).findLast(x => x.visible);
            isNullOrUndefinedThrowError(sibling); // Indicates programmer error.
        }
        const sibling_idx = this.items.indexOf(sibling);

        const _make_room = (sibling, others, tot_px) => {
            let rem = tot_px;
            // Shrink from the sibling first.
            rem = sibling.shrink(rem, {limit_to_base: false});
            if (rem <= 0) {
                return;
            }

            // Shrink all other items next, starting from the end.
            for (const other of others.slice().reverse()) {
                rem = other.shrink(rem, {limit_to_base: false});
                if (rem <= 0) {
                    return;
                }
            }

            // This indicates a programmer error.
            throw new Error(`No space for item. tot: ${tot_px}, rem: ${rem}`);
        };

        const curr_size = item.getSize();
        if (size_px < curr_size) { // shrink
            item.shrink(curr_size - size_px, {limit_to_base: false});
            sibling.grow(-1);
        } else if (size_px > curr_size) { // grow
            const others = this.items.filter((x, i) => {
                return x.visible && i !== idx && i !== sibling_idx;
            });
            _make_room(sibling, others, size_px - curr_size);
            item.setSize(size_px);
        }
    }

    show({id, idx, elem, item} = {}) {
        /** Shows an item along this axis.
         *
         *  The arguments to this function are used to lookup the item to show.
        */
        if (!(item instanceof ResizeGridItem)) {
            item = this.getItem({id: id, idx: idx, elem: elem});
        }

        if (item.visible) {
            return;
        }

        // We are trying to show an item but the container (this) is not visible.
        // Show the container (this) first so we can show the item.
        if (item !== this && !this.visible) {
            // If any parent items are visible, then we need to make room for this item.
            if (this.parent.items.some(x => x.visible)) {
                this.makeRoomForItem(this, this.parent.items, {use_base_size: false});
            }
            super.show();
            this.parent.render();
        }

        // No items are visible in this container. Show the item and force flexGrow=1.
        if (this.items.every(x => !x.visible)) {
            item.show();
            item.handle.hide();
            item.render({force_flex_grow: true});
            return;
        }

        // Other items are visible in this container, make room for this item.
        this.makeRoomForItem(item, this.items, {use_base_size: false});
        item.show();
        this.updateVisibleHandles();
    }

    hide({id, idx, elem, item} = {}) {
        /** Hides an item along this axis.
         *
         *  The arguments to this function are used to lookup the item to hide.
        */
        if (!(item instanceof ResizeGridItem)) {
            item = this.getItem({id: id, idx: idx, elem: elem});
        }

        if (!item.visible) {
            return;
        }

        item.hide();
        // If no other items are visible, hide the container.
        if (this.items.every(x => !x.visible)) {
            super.hide();
            if (this.parent.items.every(x => !x.visible)) {
                this.parent.render();
                return;
            }
            this.parent.growToFill(this, this.parent.items);
            this.parent.render();
            return;
        }
        this.growToFill(item, this.items);
    }

    toggle({id, idx, elem, item, override} = {}) {
        /** Toggles the visibility of an item along this axis.
         *
         *  The arguments to this function are used to lookup the item to show.
         *
         *  Args:
         *      override (bool): If specified, this value is used to set visibility.
        */
        if (!(item instanceof ResizeGridItem)) {
            item = this.getItem({id: id, idx: idx, elem: elem});
        }

        let new_state = !item.visible;
        if (override === true || override === false) {
            new_state = override;
        }
        new_state ? item.parent.show({item: item}) : item.parent.hide({item: item});
    }
}

class ResizeGrid extends ResizeGridAxis {
    /** Class representing a resizable grid.
     *
     *  Attributes (the less obvious ones):
     *      added_outer_div (bool):
     *          Whether the outermost ResizeGridAxis was added during setup. This is
     *          used on destruction to revert the container element to its original state.
     *      prev_dims (Object):
     *          Generated by elem.getBoundingClientRect(). This tracks the last known
     *          dimensions of this container element between resize events.
    */
    event_abort_controller = null;
    added_outer_div = false;
    added_elem_id = false;
    setup_has_run = false;
    prev_dims = null;
    resize_observer = null;
    resize_observer_timer = null;
    constructor(id, elem, {callbacks} = {}) {
        const row_elems = Array.from(elem.querySelectorAll(":scope > .resize-grid--row"));
        const col_elems = Array.from(elem.querySelectorAll(":scope > .resize-grid--col"));
        let axis = 0;
        if (row_elems.length && col_elems.length) {
            throw new Error("Invalid grid. Cannot have rows and cols at same level.");
        } else if (row_elems.length) {
            axis = 0;
        } else if (col_elems.length) {
            axis = 1;
        } else {
            axis = 0;
        }

        super({
            id: id,
            elem: elem,
            parent: null,
            axis: axis,
            callbacks: callbacks,
        });

        if (this.elem.id !== id) {
            this.elem.id = id;
            this.added_elem_id = true;
        }
    }

    destroy() {
        this.destroyEvents();
        if (this.added_outer_div) {
            this.elem.innerHTML = this.elem.children[0].innerHTML;
            this.added_outer_div = false;
        }
        if (this.added_elem_id) {
            this.elem.removeAttribute("id");
            this.added_elem_id = false;
        }
        super.destroy();
        this.setup_has_run = false;
    }

    destroyEvents() {
        /** Destroys all event listeners and observers. */
        // We can simplify removal of event listeners by firing an AbortController
        // abort signal. Must pass the signal to any event listeners on creation.
        if (this.event_abort_controller) {
            this.event_abort_controller.abort();
        }
        if (!isNullOrUndefined(this.resize_observer)) {
            this.resize_observer.disconnect();
        }
        clearTimeout(this.resize_observer_timer);
        this.resize_observer = null;
        this.resize_observer_timer = null;
    }

    setup() {
        /** Fully prepares this instance for use. */
        // We don't want to run setup a second time without having run `destroy()`.
        if (this.setup_has_run) {
            // Indicates programmer error.
            throw new Error("Setup has already run.");
        }

        if (!this.elem.querySelector('.resize-grid--row,.resize-grid--col,.resize-grid--cell')) {
            throw new Error('Grid has no valid content from which it can build.');
        }

        this.prev_dims = this.elem.getBoundingClientRect();
        this.build();
        this.setupEvents();
        this.setup_has_run = true;
    }

    setupEvents() {
        /** Sets up all event delegators and observers for this instance. */
        this.event_abort_controller = new AbortController();
        let active_pointer_id;
        let siblings;
        let dblclick_timer;
        let last_move_time;

        const _on_pointerdown = (event) => {
            if (!isNullOrUndefined(active_pointer_id) || !event.isPrimary) {
                return;
            }

            const handle_elem = event.target.closest(".resize-grid--handle");
            if (!isElement(handle_elem)) {
                return;
            }

            if (handle_elem.closest(".resize-grid").dataset.id !== this.id) {
                return;
            }

            siblings = this.getSiblings(handle_elem);
            if (!(siblings.prev instanceof ResizeGridItem) ||
                !(siblings.handle instanceof ResizeGridHandle) ||
                !(siblings.next instanceof ResizeGridItem)
            ) {
                siblings = null;
                throw new Error(`Failed to find siblings for handle: ${handle_elem}`);
            }

            event.preventDefault();
            active_pointer_id = event.pointerId;

            // Temporarily set styles for elements. These are cleared on pointerup.
            // Also cleared if dblclick is fired.
            // See `onMove()` comments for more info.
            siblings.prev.setSize(siblings.prev.getSize());
            siblings.next.setSize(siblings.next.getSize());
            siblings.prev.elem.style.flexGrow = 0;
            siblings.next.elem.style.flexGrow = 1;
            siblings.next.elem.style.flexShrink = 1;

            document.body.classList.add('resizing');
            if (siblings.handle.axis === 0) {
                document.body.classList.add('resizing-col');
            } else {
                document.body.classList.add('resizing-row');
            }

            if (!dblclick_timer) {
                siblings.handle.elem.dataset.awaitDblClick = '';
                dblclick_timer = setTimeout(
                    (elem) => {
                        dblclick_timer = null;
                        delete elem.dataset.awaitDblClick;
                    },
                    DBLCLICK_TIME_MS,
                    siblings.handle.elem
                );
            } else if ('awaitDblClick' in siblings.handle.elem.dataset) {
                clearTimeout(dblclick_timer);
                dblclick_timer = null;
                delete siblings.handle.elem.dataset.awaitDblClick;
                siblings.handle.elem.dispatchEvent(
                    new CustomEvent('resize_grid_handle_dblclick', {
                        bubbles: true,
                        detail: this,
                    })
                );
                document.body.classList.remove('resizing', 'resizing-col', 'resizing-row');
                siblings.prev.render();
                siblings.next.render();
                siblings = null;
                active_pointer_id = null;
                return;
            }
        };

        const _on_pointermove = (event) => {
            if (event.pointerId !== active_pointer_id) {
                return;
            }

            if (isNullOrUndefined(siblings)) {
                return;
            }

            event.preventDefault();

            const now = new Date().getTime();
            if (!last_move_time || now - last_move_time > MOVE_TIME_DELAY_MS) {
                this.onMove(event, siblings.prev, siblings.handle, siblings.next);
                last_move_time = now;
            }
        };

        const _on_pointerup = (event) => {
            if (event.pointerId !== active_pointer_id) {
                return;
            }

            document.body.classList.remove('resizing', 'resizing-col', 'resizing-row');

            if (isNullOrUndefined(siblings)) {
                return;
            }

            event.preventDefault();

            // Set the new flexBasis value for the `next` element then revert
            // the style changes set in the `pointerup` event.
            siblings.next.setSize(siblings.next.getSize());
            siblings.prev.render();
            siblings.next.render();
            siblings = null;
            active_pointer_id = null;
        };

        const event_options = {signal: this.event_abort_controller.signal};
        window.addEventListener('pointerdown', _on_pointerdown, event_options);
        window.addEventListener('pointermove', _on_pointermove, event_options);
        window.addEventListener('pointerup', _on_pointerup, event_options);

        this.resize_observer = new ResizeObserver((entries) => {
            for (const entry of entries) {
                if (entry.target.id === this.id) {
                    clearTimeout(this.resize_observer_timer);
                    this.resize_observer_timer = setTimeout(() => {
                        this.onResize();
                    }, RESIZE_DEBOUNCE_TIME_MS);
                }
            }
        });
        this.resize_observer.observe(this.elem);
    }

    addAxis(id, elem, idx) {
        /** Creates a ResizeGridAxis instance and returns the new item. */
        const item = new ResizeGridAxis({
            id: id,
            parent: this,
            elem: elem,
            axis: this.axis,
            callbacks: this.callbacks,
        });
        item.elem.dataset.index = idx;
        item.genHandle();
        this.items.push(item);
        return item;
    }

    build() {
        /** Generates rows/cols based on this instance element's content. */
        // Flex direction should be opposite of container axis.
        this.elem.style.flexDirection = this.axis === 0 ? "column" : "row";
        let axis_elems;
        if (this.axis === 0) {
            axis_elems = Array.from(this.elem.querySelectorAll(":scope > .resize-grid--row"));
        } else {
            axis_elems = Array.from(this.elem.querySelectorAll(":scope > .resize-grid--col"));
        }
        if (!axis_elems.length) {
            // If we don't have any rows/cols, then make a new axis.
            const elem = document.createElement("div");
            elem.classList.add("resize-grid--row");
            elem.append(...this.elem.children);
            axis_elems = [elem];
            // track this change so we can undo it on `destroy()`.
            this.added_outer_div = true;
        }

        // If we only have a single row/col then make it fill the container.
        if (axis_elems.length === 1) {
            axis_elems[0].style.flexGrow = 1;
        }
        let id = 0;
        axis_elems.forEach((axis_elem, i) => {
            const axis = this.addAxis(id++, axis_elem, i);
            axis_elem.querySelectorAll(
                ":scope > .resize-grid--cell"
            ).forEach((cell_elem, j) => axis.addCell(id++, cell_elem, j));
        });

        // Set any rows/cols with only one element to be flexGrow=1 and flexShrink=1
        // by default. Since they don't have siblings to work around, this makes it so
        // the user doesn't have to specify these settings in HTML.
        if (this.items.length === 1) {
            this.items[0].is_flex_grow = 1;
            this.items[0].elem.style.flexGrow = 1;
            this.items[0].is_flex_shrink = 1;
            this.items[0].elem.style.flexShrink = 1;
        }

        for (const item of this.items) {
            if (item.items.length === 1) {
                item.items[0].is_flex_grow = 1;
                item.items[0].elem.style.flexGrow = 1;
                item.items[0].is_flex_shrink = 1;
                item.items[0].elem.style.flexShrink = 1;
            }
        }

        // Render all axes to force them to update flex dims.
        this.items[this.items.length - 1].handle.hide();
        this.items.forEach(axis => {
            axis.items[axis.items.length - 1].handle.hide();
        });
        this.render({reset: true});
        this.buildIdMap(this);
    }

    buildIdMap(item) {
        /** Generates a mapping from ID to ResizeGridItem instances.
         *
         *  Starts at the passed `item` as a root and recursively builds mapping
         *  from child items.
        */
        if (item instanceof ResizeGridAxis) {
            this.item_ids[item.id] = item;
            item.items.forEach(x => this.buildIdMap(x));
        } else if (item instanceof ResizeGridItem) {
            this.item_ids[item.id] = item;
        }
    }

    getSiblings(handle_elem) {
        /** Returns the nearest visible ResizeGridItems surrounding a ResizeGridHandle.
         *
         *  Args:
         *      handle_elem (Element): The handle element in the grid to lookup.
         *
         *  Returns:
         *      Object: Keys=(prev, next). Values are ResizeGridItems.
        */
        let prev = this.getItem({elem: handle_elem.previousElementSibling});
        if (!prev.visible) {
            prev = prev.parent.items.slice(0, this.items.indexOf(prev)).findLast(x => x.visible);
        }
        let next = this.getItem({elem: handle_elem.nextElementSibling});
        if (!next.visible) {
            next = next.parent.items.slice(this.items.indexOf(next) + 1).findLast(x => x.visible);
        }
        return {prev: prev, handle: prev.handle, next: next};
    }

    onMove(event, a, handle, b) {
        /** Handles pointermove events by calculating and setting new size of elements.
         *
         *  While we are dragging the handle, we only manually set size for the
         *  item before the handle. During this time, the element after the handle
         *  should have flexGrow=1 and flexShrink=1 so that it can react to the size
         *  of the element before the handle. This simplifies calculations since we
         *  only have to do math for one side of handle.
        */
        const a_dims = a.elem.getBoundingClientRect();
        const b_dims = b.elem.getBoundingClientRect();
        const a_start = Math.ceil(handle.axis === 0 ? a_dims.top : a_dims.left);
        const b_end = Math.floor(handle.axis === 0 ? b_dims.bottom : b_dims.right);
        const a_lim = a_start + a.min_size;
        const b_lim = b_end - b.min_size;
        const half_pad = Math.floor(handle.pad_px / 2);
        let pos = parseInt(handle.axis === 0 ? event.y : event.x);
        pos = Math.min(Math.max(pos, a_lim + half_pad), b_lim - half_pad);
        a.setSize((pos - half_pad) - a_start);
    }

    onResize() {
        /** Resizes grid items on resize observer events. */
        const curr_dims = this.elem.getBoundingClientRect();

        // If height and width are 0, this indicates the element is not visible anymore.
        // We don't want to do anything if the element isn't visible.
        if (curr_dims.height === 0 && curr_dims.width === 0) {
            return;
        }

        // Do nothing if the dimensions haven't changed.
        if (JSON.stringify(curr_dims) === JSON.stringify(this.prev_dims)) {
            return;
        }

        const d_w = curr_dims.width - this.prev_dims.width;
        const d_h = curr_dims.height - this.prev_dims.height;

        // If no change to size, don't proceed.
        if (d_w === 0 && d_h === 0) {
            return;
        }

        if (d_w < 0) {
            // Width decrease
            for (const item of this.items) {
                if (this.axis === 0) {
                    for (const subitem of item.items) {
                        if (subitem.getSize() > subitem.min_size) {
                            subitem.elem.style.flexShrink = 1;
                        } else {
                            subitem.elem.style.flexShrink = 0;
                            subitem.elem.style.flexBasis = subitem.min_size + "px";
                        }
                    }
                } else {
                    if (item.getSize() > item.min_size) {
                        item.elem.style.flexShrink = 1;
                    } else {
                        item.elem.style.flexShrink = 0;
                        item.elem.style.flexBasis = item.min_size + "px";
                    }
                }
            }
        } else if (d_w > 0) {
            // width increase
            if (this.axis === 0) {
                for (const item of this.items) {
                    let did_grow = false;
                    let subitem = item.items.findLast(x => x.visible && x.is_flex_grow);
                    if (subitem instanceof ResizeGridItem) {
                        did_grow = subitem.grow(-1) === 0;
                    }
                    if (!did_grow) {
                        subitem = item.items.findLast(x => x.visible && !x.is_flex_grow);
                        if (subitem instanceof ResizeGridItem) {
                            subitem.grow(-1);
                        }
                    }
                }
            }
        }

        if (d_h < 0) {
            // height decrease
            for (const item of this.items) {
                if (this.axis === 1) {
                    for (const subitem of item.items) {
                        if (subitem.getSize() > subitem.min_size) {
                            subitem.elem.style.flexShrink = 1;
                        } else {
                            subitem.elem.style.flexShrink = 0;
                            subitem.elem.style.flexBasis = subitem.min_size + "px";
                        }
                    }
                } else {
                    if (item.getSize() > item.min_size) {
                        item.elem.style.flexShrink = 1;
                    } else {
                        item.elem.style.flexShrink = 0;
                        item.elem.style.flexBasis = item.min_size + "px";
                    }
                }
            }
        } else if (d_h > 0) {
            // height increase
            if (this.axis === 1) {
                let did_grow = false;
                let item = this.items.findLast(x => x.visible && x.is_flex_grow);
                if (item instanceof ResizeGridItem) {
                    did_grow = item.grow(-1) === 0;
                }
                if (!did_grow) {
                    item = this.items.findLast(x => x.visible && !x.is_flex_grow);
                    if (item instanceof ResizeGridItem) {
                        item.grow(-1);
                    }
                }
            }
        }

        this.prev_dims = curr_dims;
    }
}

function resizeGridGetGrid(elem) {
    /** Returns the nearest ResizeGrid instance to the passed element. */
    // Need to find grid element so we can lookup by its id.
    const grid_elem = elem.closest('.resize-grid');
    if (!isElement(grid_elem)) {
        return null;
    }
    // Now try to get the actual ResizeGrid instance.
    const grid = resize_grids[grid_elem.dataset.id];
    if (isNullOrUndefined(grid)) {
        return null;
    }

    return grid;
}

function resizeGridSetup(elem, {id, callbacks} = {}) {
    /** Sets up a new ResizeGrid instance for the provided element. */
    isElementThrowError(elem);
    if (!isString(id)) {
        id = _get_unique_id();
    }
    const grid = new ResizeGrid(id, elem, {callbacks: callbacks});
    grid.setup();
    resize_grids[id] = grid;
    return grid;
}
