/** @format */

// Prevent eslint errors on functions defined in other files.
/*global
    isNullOrUndefinedThrowError,
    isNullOrUndefinedLogError,
    isNullOrUndefined,
    isString,
    cssRelativeUnitToPx,
    isNumber,
    isElementThrowError,
    isElement,
*/
/*eslint no-undef: "error"*/

// Should be between 0 and 15. Any higher and the delay becomes noticable.
// Higher values reduce computational load.
const MOVE_TIME_DELAY_MS = 15;
// Prevents handling element resize events too quickly. Lower values increase
// computational load and may lead to lag when resizing.
const RESIZE_DEBOUNCE_TIME_MS = 100;
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
    let id = _gen_id_string();
    while (id in Object.keys(resize_grids)) {
        id = _gen_id_string();
    }
    return id;
};

const _parse_array_type = (arr, type_check_fn) => {
    /** Validates that a variable is an array with members of a specified type.
     * `type_check_fn` must accept array elements as arguments and return whether
     * they match the expected type.
     */
    isNullOrUndefinedThrowError(type_check_fn);
    if (isNullOrUndefined(arr)) {
        return [];
    }
    if (!Array.isArray(arr) && type_check_fn(arr)) {
        return [arr];
    } else if (Array.isArray(arr) && arr.every((x) => type_check_fn(x))) {
        return arr;
    } else {
        throw new Error('Invalid array types:', arr);
    }
};

const _axis_to_int = (axis) => {
    /** Converts an axis to a standardized axis integer.
     *  Returns:
     *      "x" or 0: 0
     *      "y" or 1: 1
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
    /** Class defining the clickable "handle" between two grid items. */
    visible = true;
    id = null; // unique identifier for this instance.
    pad_px = PAD_PX;
    constructor({id, parent, axis, class_list} = {}) {
        this.id = isNullOrUndefined(id) ? _gen_id_string() : id;
        this.parent = parent;
        this.elem = document.createElement('div');
        this.elem.id = id;
        this.elem.classList.add('resize-grid--handle');
        _parse_array_type(class_list, isString).forEach((class_name) => {
            this.elem.classList.add(class_name);
        });

        this.axis = _axis_to_int(axis);

        if (this.axis === 0) {
            this.elem.style.minHeight = this.pad_px + 'px';
            this.elem.style.maxHeight = this.pad_px + 'px';
        } else if (this.axis === 1) {
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
    /** Class defining the cells in a grid. These can be rows or columns. */
    handle = null;
    visible = true;
    pad_px = PAD_PX;
    constructor({id, parent, elem, axis} = {}) {
        this.id = isNullOrUndefined(id) ? _gen_id_string() : id;
        this.parent = parent; // the parent class instance
        this.elem = elem;
        this.axis = _axis_to_int(axis);

        this.is_flex_grow = Boolean(parseInt(this.elem.style.flexGrow));
        this.default_is_flex_grow = this.is_flex_grow;
        let flex_basis = parseInt(cssRelativeUnitToPx(this.elem.style.flexBasis));
        // If user specifies data-min-size, then the flexBasis is just used to set
        // the initial size.
        if ('minSize' in this.elem.dataset) {
            this.min_size = parseInt(cssRelativeUnitToPx(this.elem.dataset.minSize));
            if (isNumber(flex_basis)) {
                this.base_size = flex_basis;
            } else {
                this.base_size = this.min_size;
            }
        } else if (isNumber(flex_basis)) {
            this.min_size = flex_basis;
            this.base_size = flex_basis;
        } else {
            this.min_size = 0;
            this.base_size = 0;
        }
        const dims = this.elem.getBoundingClientRect();
        this.base_size =
            this.axis === 0 ? parseInt(dims.height) : parseInt(dims.width);
        this.elem.dataset.id = this.id;
        this.original_css_text = this.elem.style.cssText;
    }

    render({force_flex_grow, reset} = {}) {
        /** Sets the element's flex styles. */
        force_flex_grow = force_flex_grow === true;
        reset = reset === true;

        this.elem.style.flexShrink = 0;
        if (reset) {
            this.elem.style.flexGrow = Number(this.is_flex_grow);
            this.elem.style.flexBasis = parseInt(this.base_size) + 'px';
        } else if (force_flex_grow) {
            this.elem.style.flexGrow = 1;
        } else {
            this.elem.style.flexGrow = Number(this.is_flex_grow);
            if (!this.is_flex_grow) {
                this.elem.style.flexBasis =
                    parseInt(Math.max(this.min_size, this.getSize())) + 'px';
            }
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
    }

    shrink(px, {limit_to_base} = {}) {
        /** Shrink size along axis by specified pixels. Returns remainder. */
        limit_to_base = limit_to_base === true;
        const target_size = limit_to_base ? this.base_size : this.min_size;
        const curr_size = this.getSize();

        if (px === -1) {
            // shrink to min_size
            this.setSize(target_size);
            return 0;
        } else if (curr_size - target_size < px) {
            this.setSize(target_size);
            return px - (curr_size - target_size);
        } else {
            this.setSize(curr_size - px);
            return 0;
        }
    }

    grow(px, {only_if_flex} = {}) {
        /** Grows along axis and returns the amount grown in pixels. */
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

    setSize(px) {
        this.elem.style.flexBasis = parseInt(px) + 'px';
        this.render();
    }

    getSize() {
        // If this item is visible, then we can use the computed dimensions.
        // Otherwise we are forced to use the flexBasis inline style.
        if (this.visible) {
            const dims = this.elem.getBoundingClientRect();
            return this.axis === 0 ? parseInt(dims.height) : parseInt(dims.width);
        } else {
            return parseInt(this.elem.style.flexBasis);
        }
    }

    genHandle(class_list) {
        /** Generates a ResizeGridHandle after this item based on the axis. */
        this.handle = new ResizeGridHandle({
            id: `${this.id}_handle`,
            parent: this.parent,
            axis: this.axis,
            class_list: class_list,
        });
        if (isElement(this.elem.nextElementSibling)) {
            this.elem.parentElement.insertBefore(
                this.handle.elem,
                this.elem.nextSibling
            );
        } else {
            this.elem.parentElement.appendChild(this.handle.elem);
        }
    }

    show() {
        /** Shows this item and its ResizeGridHandle. */
        this.elem.classList.remove('hidden');
        // Only show the handle if there is another ResizeGridItem after this one.
        if (!isNullOrUndefined(this.handle.elem.nextSibling)) {
            this.handle.show();
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

class ResizeGridContainer {
    /** Class defining a collection of ResizeGridItem and ResizeGridHandle instances. */
    constructor({id, parent, elem} = {}) {
        this.id = isNullOrUndefined(id) ? _gen_id_string() : id;
        this.parent = parent;
        this.elem = elem;
        this.original_css_text = this.elem.style.cssText;

        this.grid = [];
        this.rows = [];
        this.id_map = {};
        this.added_outer_row = false;
    }

    destroy() {
        this.rows.forEach((row) => {
            row.destroy();
        });
        this.rows = null;
        if (this.added_outer_row) {
            this.elem.innerHTML = this.elem.querySelector(
                ':scope > .resize-grid--row'
            ).innerHTML;
        }
        super.destroy();
    }

    addRow(id, elem, row_idx) {
        /** Generates a ResizeGridItem and ResizeGridHandle for a row element. */
        const row = new ResizeGridItem({
            id: id,
            parent: this,
            elem: elem,
            axis: 0,
        });
        row.genHandle('resize-grid--row-handle');
        row.elem.dataset.row = row_idx;
        this.rows.push(row);
        this.id_map[id] = row;
        return row;
    }

    addCol(id, elem, row_idx, col_idx) {
        /** Generates a ResizeGridItem and ResizeGridHandle for a column element. */
        const col = new ResizeGridItem({
            id: id,
            parent: this,
            elem: elem,
            axis: 1,
        });
        col.genHandle('resize-grid--col-handle');
        col.elem.dataset.row = row_idx;
        col.elem.dataset.col = col_idx;
        this.grid[row_idx].push(col);
        this.id_map[id] = col;
        return col;
    }

    build() {
        /** Generates rows/cols based on this instance element's content. */
        let row_elems = Array.from(this.elem.querySelectorAll('.resize-grid--row'));
        // If we do not have any rows, then we generate a single row to contain the cols.
        if (!row_elems.length) {
            const elem = document.createElement('div');
            elem.classList.add('resize-grid--row');
            elem.append(...this.elem.children);
            this.elem.replaceChildren(elem);
            row_elems = [elem];
            // track this addition so we can remove it later.
            this.added_outer_row = true;
        }

        // Make sure that if we only have one row, that it fills the container.
        if (row_elems.length === 1 && !row_elems[0].style.flexBasis) {
            row_elems[0].style.flexGrow = 1;
            row_elems[0].style.flexBasis =
                parseInt(this.elem.getBoundingClientRect().height) + 'px';
        }

        let id = 0;
        this.grid = [...Array(row_elems.length)].map((_) => []);
        row_elems.forEach((row_elem, i) => {
            this.addRow(id++, row_elem, i);
            const col_elems = row_elem.querySelectorAll('.resize-grid--col');
            col_elems.forEach((col_elem, j) => {
                this.addCol(id++, col_elem, i, j);
            });
            this.grid[i][this.grid[i].length - 1].handle.hide();
        });
        this.rows[this.rows.length - 1].handle.hide();

        // Now that all handles are added, we need to render the flex styles for each item.
        for (let i = 0; i < this.rows.length; i++) {
            this.rows[i].render({reset: true});
            for (let j = 0; j < this.grid[i].length; j++) {
                this.grid[i][j].render({reset: true});
            }
            const vis_cols = this.grid[i].filter((x) => x.visible);
            if (vis_cols.length === 1) {
                vis_cols[0].render({force_flex_grow: true});
            }
        }
        const vis_rows = this.rows.filter((x) => x.visible);
        if (vis_rows.length === 1) {
            vis_rows[0].render({force_flex_grow: true});
        }
    }

    getByElem(elem) {
        return this.id_map[elem.dataset.id];
    }

    getByIdx({row_idx, col_idx} = {}) {
        /** Returns the ResizeGridItem at the row/col index. */
        row_idx = parseInt(row_idx);
        col_idx = parseInt(col_idx);
        if (
            (!isNumber(row_idx) && !isNumber(col_idx)) ||
            (!isNumber(row_idx) && isNumber(col_idx))
        ) {
            console.error('Invalid row/col idx:', row_idx, col_idx);
            return;
        }
        if (isNumber(row_idx) && !isNumber(col_idx)) {
            if (row_idx >= this.rows.length) {
                console.error(
                    `row_idx out of range: (${row_idx} > ${this.rows.length})`
                );
                return;
            }
            return this.rows[row_idx];
        }
        if (isNumber(row_idx) && isNumber(col_idx)) {
            if (row_idx >= this.grid.length) {
                console.error(
                    `row_idx out of range: (${row_idx} > ${this.grid.length})`
                );
                return;
            }
            if (col_idx >= this.grid[row_idx].length) {
                console.error(
                    `col_idx out of range: (${col_idx} > ${this.grid[row_idx].length})`
                );
                return;
            }
            return this.grid[row_idx][col_idx];
        }
    }

    updateVisibleHandles() {
        /** Sets the visibility of each ResizeGridHandle based on surrounding items. */
        const last_vis_rows_idx = this.rows.findLastIndex((x) => x.visible);
        for (let i = 0; i < this.rows.length; i++) {
            const last_vis_grid_idx = this.grid[i].findLastIndex((x) => x.visible);
            for (let j = 0; j < this.grid[i].length; j++) {
                const item = this.getByIdx({row_idx: i, col_idx: j});
                if (isNullOrUndefined(item)) {
                    continue;
                }

                // Don't show handle if item is last column in row.
                if (this.grid[i][j].visible && j !== last_vis_grid_idx) {
                    this.grid[i][j].handle.show();
                } else {
                    this.grid[i][j].handle.hide();
                }
            }

            const item = this.getByIdx({row_idx: i});
            if (isNullOrUndefined(item)) {
                continue;
            }

            // Don't show handle if item is last row in grid.
            if (this.rows[i].visible && i !== last_vis_rows_idx) {
                this.rows[i].handle.show();
            } else {
                this.rows[i].handle.hide();
            }
        }
    }

    makeRoomForItem(item, siblings, item_idx, {use_base_size} = {}) {
        /** Shrinks items along axis until the supplied item can fit. */
        use_base_size = use_base_size === true;
        let tot = use_base_size ? item.base_size : item.getSize();
        // Get the item after this item's handle.
        let sibling = siblings.slice(item_idx + 1).find((x) => x.visible);
        if (isNullOrUndefined(sibling)) {
            // No items after this item. Instead get the item just before this item.
            sibling = siblings.slice(0, item_idx).findLast((x) => x.visible);
            isNullOrUndefinedThrowError(sibling); // Indicates programmer error.
            // Last item so we want to hide its handle.
            item.handle.hide();
            // Add previous item handle's size
            tot += sibling.handle.pad_px;
        } else {
            // Need to add handle between this item and next item.
            item.handle.show();
            tot += item.handle.pad_px;
        }

        const sibling_idx = siblings.indexOf(sibling);

        let rem = tot;
        rem = sibling.shrink(rem, {limit_to_base: use_base_size});
        if (rem <= 0) {
            return;
        }
        // Shrink from flexGrow items next starting from the end.
        if (rem > 0) {
            const others = siblings.filter(
                (x) =>
                    x.visible && x.is_flex_grow && siblings.indexOf(x) !== sibling_idx
            );
            for (const other of others.slice().reverse()) {
                rem = other.shrink(rem, {limit_to_base: use_base_size});
                if (rem <= 0) {
                    return;
                }
            }
        }
        // Now shrink from non-flexGrow items starting from the end.
        if (rem > 0) {
            const others = siblings.filter(
                (x) =>
                    x.visible && !x.is_flex_grow && siblings.indexOf(x) !== sibling_idx
            );
            for (const other of others.slice().reverse()) {
                rem = other.shrink(rem, {limit_to_base: use_base_size});
                if (rem <= 0) {
                    return;
                }
            }
        }

        // Shrink the item itself if we still don't have room.
        if (rem > 0) {
            rem = item.shrink(rem, {limit_to_base: use_base_size});
            if (rem <= 0) {
                return;
            }
        }

        // If still not enough room, try again but use the base sizes.
        if (rem > 0 && !use_base_size) {
            this.makeRoomForItem(item, siblings, item_idx, {use_base_size: true});
            return;
        }
        // If we still couldn't make room, this indicates programmer error.
        throw new Error(`No space for row. tot: ${tot}, rem: ${rem}`);
    }

    growToFill(item, siblings, item_idx, tot_px) {
        /** Expands item along axis until the axis has no remaining space. */
        // Expand the item that was attached via the hidden item's handle first.
        let sibling = siblings.slice(item_idx + 1).find((x) => x.visible);
        if (isNullOrUndefined(sibling)) {
            // Otherwise, expand the previous attached item.
            sibling = siblings.slice(0, item_idx).findLast((x) => x.visible);
            isNullOrUndefinedThrowError(sibling); // Indicates programmer error.
        } else {
            tot_px += item.pad_px;
        }
        // Hide sibling's handle if sibling is last visible item.
        if (
            siblings
                .slice(siblings.findIndex((x) => x === sibling) + 1)
                .every((x) => !x.visible)
        ) {
            if (sibling.handle.visible) {
                sibling.handle.hide();
                tot_px += sibling.pad_px;
            }
        }

        // If we are growing sibling to fill, then just set flexGrow=1.
        if (
            siblings.length <= 2 ||
            siblings.every((x) => !x.visible && x !== sibling && x !== item)
        ) {
            sibling.render({force_flex_grow: true});
        } else {
            sibling.grow(-1);
        }
    }

    showRow(row_idx, {show_empty_row} = {}) {
        /** Makes space for the row then shows it. */
        show_empty_row = show_empty_row === true;
        row_idx = parseInt(row_idx);
        const item = this.getByIdx({row_idx: row_idx});
        isNullOrUndefinedThrowError(item);

        if (item.visible) {
            return;
        }

        if (item.axis !== 0) {
            console.error('Expected row, got col:', item);
            return;
        }

        // If no columns are visible, then we can't show the row.
        if (this.grid[row_idx].every((x) => !x.visible) && !show_empty_row) {
            console.error('No visible columns in row. Cannot show.');
            return;
        }

        // All rows are hidden. We just show this row and make it fill the container.
        if (this.rows.every((x) => !x.visible)) {
            item.show();
            item.handle.hide();
            item.render({force_flex_grow: true});
        } else {
            this.makeRoomForItem(item, this.rows, row_idx);
            item.show();
            if (this.rows.slice(row_idx + 1).every((x) => !x.visible)) {
                // If this is the last visible row, hide the handle.
                item.handle.hide();
            }
            const prev_item = this.rows.slice(0, row_idx).find((x) => x.visible);
            if (!isNullOrUndefined(prev_item)) {
                prev_item.handle.show();
            }
            item.render();
        }
    }

    showCol(row_idx, col_idx) {
        /** Makes space for the column then shows it. */
        row_idx = parseInt(row_idx);
        col_idx = parseInt(col_idx);
        const item = this.getByIdx({row_idx: row_idx, col_idx: col_idx});
        isNullOrUndefinedThrowError(item);

        if (item.visible) {
            return;
        }

        if (item.axis !== 1) {
            console.error('Expected col, got row:', item);
            return;
        }

        // If the row isn't visible, we need to show it before we can show columns.
        if (!this.rows[row_idx].visible) {
            this.showRow(row_idx, {show_empty_row: true});
        }

        // All cols are hidden. We just show this col and make it fill the row.
        if (this.grid[row_idx].every((x) => !x.visible)) {
            item.show();
            item.handle.hide();
            item.render({force_flex_grow: true});
        } else {
            this.makeRoomForItem(item, this.grid[row_idx], col_idx);
            item.show();
            if (this.grid[row_idx].slice(col_idx + 1).every((x) => !x.visible)) {
                // If this is the last visible col, hide the handle.
                item.handle.hide();
            }
            const prev_item = this.grid[row_idx]
                .slice(0, col_idx)
                .find((x) => x.visible);
            if (!isNullOrUndefined(prev_item)) {
                prev_item.handle.show();
            }
            item.render();
        }
    }

    hideRow(row_idx) {
        /** Hides a row and resizes other rows to fill the gap. */
        row_idx = parseInt(row_idx);
        const item = this.getByIdx({row_idx: row_idx});
        isNullOrUndefinedThrowError(item);

        if (!item.visible) {
            return;
        }

        if (item.axis !== 0) {
            console.error('Expected row, got col:', item);
            return;
        }

        let tot_px = item.elem.getBoundingClientRect().height;
        item.hide();
        // If no other rows are visible, we don't need to do anything else.
        if (this.rows.every((x) => !x.visible)) {
            return;
        }

        this.growToFill(item, this.rows, row_idx, tot_px);
    }

    hideCol(row_idx, col_idx) {
        /** Hides a column and resizes other columns to fill the gap. */
        row_idx = parseInt(row_idx);
        col_idx = parseInt(col_idx);
        const item = this.getByIdx({row_idx: row_idx, col_idx: col_idx});
        isNullOrUndefinedThrowError(item);

        if (!item.visible) {
            return;
        }

        if (item.axis !== 1) {
            console.error('Expected col, got row:', item);
            return;
        }

        let tot_px = item.getSize();
        item.hide();
        // If no other cols are visible, hide the containing row.
        if (this.grid[row_idx].every((x) => !x.visible)) {
            this.hideRow(row_idx);
            return;
        }

        this.growToFill(item, this.grid[row_idx], col_idx, tot_px);
    }

    show({row_idx, col_idx} = {}) {
        /** Shows a row or column based on the provided row/col indices. */
        row_idx = parseInt(row_idx);
        col_idx = parseInt(col_idx);
        if (isNumber(row_idx) && !isNumber(col_idx)) {
            this.showRow(row_idx);
        } else if (isNumber(row_idx) && isNumber(col_idx)) {
            this.showCol(row_idx, col_idx);
        } else {
            throw new Error('Invalid parameters for row/col idx:', row_idx, col_idx);
        }
    }

    hide({row_idx, col_idx} = {}) {
        /** Hides a row or column based on the provided row/col indices. */
        row_idx = parseInt(row_idx);
        col_idx = parseInt(col_idx);
        if (isNumber(row_idx) && !isNumber(col_idx)) {
            this.hideRow(row_idx);
        } else if (isNumber(row_idx) && isNumber(col_idx)) {
            this.hideCol(row_idx, col_idx);
        } else {
            throw new Error('Invalid parameters for row/col idx:', row_idx, col_idx);
        }
    }

    toggle({row_idx, col_idx, override} = {}) {
        /** Toggles a row or column's visibility based on the provided row/col indices. */
        row_idx = parseInt(row_idx);
        col_idx = parseInt(col_idx);
        const item = this.getByIdx({row_idx: row_idx, col_idx: col_idx});
        isNullOrUndefinedThrowError(item);

        let new_state = !item.visible;
        if (override === true || override === false) {
            new_state = override;
        }

        if (item.axis === 0) {
            new_state ? this.showRow(row_idx) : this.hideRow(row_idx);
        } else {
            new_state ?
                this.showCol(row_idx, col_idx) :
                this.hideCol(row_idx, col_idx);
        }
    }
}

class ResizeGrid {
    /** Class representing a resizable grid. */
    event_abort_controller = null;
    added_outer_row = false;
    container = null;
    setup_has_run = false;
    prev_dims = null;
    resize_observer = null;
    resize_observer_timer;
    constructor(id, elem) {
        this.id = id;
        this.elem = elem;
        this.elem.dataset.gridId = this.id;
    }

    destroy() {
        this.destroyEvents();
        if (!isNullOrUndefined(this.container)) {
            this.container.destroy();
            this.container = null;
        }
        this.setup_has_run = false;
    }

    setup() {
        /** Fully prepares this instance for use. */
        if (!this.elem.querySelector('.resize-grid--row,.resize-grid--col')) {
            throw new Error('Container has no rows or cols.');
        }

        if (!isNullOrUndefined(this.container)) {
            this.container.destroy();
            this.container = null;
        }

        this.container = new ResizeGridContainer(this, null, this.elem);
        this.container.build();
        this.prev_dims = this.elem.getBoundingClientRect();
        this.setupEvents();
        this.setup_has_run = true;
    }

    setupEvents() {
        /** Sets up all event delegators and observers for this instance. */
        this.event_abort_controller = new AbortController();
        let prev;
        let handle;
        let next;
        let touch_count = 0;
        let dblclick_timer;
        let last_move_time;

        window.addEventListener(
            'pointerdown',
            (event) => {
                if (event.target.hasPointerCapture(event.pointerId)) {
                    event.target.releasePointerCapture(event.pointerId);
                }
                if (event.pointerType === 'mouse' && event.button !== 0) {
                    return;
                }
                if (event.pointerType === 'touch') {
                    touch_count++;
                    if (touch_count !== 1) {
                        return;
                    }
                }

                const elem = event.target.closest('.resize-grid--handle');
                if (!elem) {
                    return;
                }
                // Clicked handles will always be between two elements. If the user
                // somehow clicks an invisible handle then we have bigger problems.
                prev = this.container.getByElem(elem.previousElementSibling);
                if (!prev.visible) {
                    const row_idx = prev.elem.dataset.row;
                    const col_idx = prev.elem.dataset.col;
                    const siblings =
                        prev.axis === 0 ?
                            this.container.rows :
                            this.container.grid[row_idx];
                    const idx = prev.axis === 0 ? row_idx : col_idx;
                    prev = siblings.slice(0, idx).findLast((x) => x.visible);
                }
                handle = prev.handle;
                next = this.container.getByElem(elem.nextElementSibling);
                if (!next.visible) {
                    const row_idx = next.elem.dataset.row;
                    const col_idx = next.elem.dataset.col;
                    const siblings =
                        next.axis === 0 ?
                            this.container.rows :
                            this.container.grid[row_idx];
                    const idx = next.axis === 0 ? row_idx : col_idx;
                    next = siblings.slice(idx).find((x) => x.visible);
                }

                if (
                    isNullOrUndefinedLogError(prev) ||
                    isNullOrUndefinedLogError(handle) ||
                    isNullOrUndefinedLogError(next)
                ) {
                    prev = null;
                    handle = null;
                    next = null;
                    return;
                }

                event.preventDefault();
                event.stopPropagation();

                handle.elem.setPointerCapture(event.pointerId);

                document.body.classList.add('resizing');
                if (handle.axis === 0) {
                    document.body.classList.add('resizing-col');
                } else {
                    document.body.classList.add('resizing-row');
                }
            },
            {signal: this.event_abort_controller.signal}
        );

        window.addEventListener(
            'pointermove',
            (event) => {
                if (
                    isNullOrUndefined(prev) ||
                    isNullOrUndefined(handle) ||
                    isNullOrUndefined(next)
                ) {
                    return;
                }

                event.preventDefault();
                event.stopPropagation();

                const now = new Date().getTime();
                if (!last_move_time || now - last_move_time > MOVE_TIME_DELAY_MS) {
                    this.onMove(event, prev, handle, next);
                    last_move_time = now;
                }
            },
            {signal: this.event_abort_controller.signal}
        );

        window.addEventListener(
            'pointerup',
            (event) => {
                if (
                    isNullOrUndefined(prev) ||
                    isNullOrUndefined(handle) ||
                    isNullOrUndefined(next)
                ) {
                    return;
                }

                if (event.target.hasPointerCapture(event.pointerId)) {
                    event.target.releasePointerCapture(event.pointerId);
                }

                if (event.pointerType === 'mouse' && event.button !== 0) {
                    return;
                }
                if (event.pointerType === 'touch') {
                    touch_count--;
                }

                event.preventDefault();
                event.stopPropagation();

                handle.elem.releasePointerCapture(event.pointerId);

                document.body.classList.remove('resizing');
                document.body.classList.remove('resizing-col');
                document.body.classList.remove('resizing-row');

                if (!dblclick_timer) {
                    handle.elem.dataset.awaitDblClick = '';
                    dblclick_timer = setTimeout(
                        (elem) => {
                            dblclick_timer = null;
                            delete elem.dataset.awaitDblClick;
                        },
                        DBLCLICK_TIME_MS,
                        handle.elem
                    );
                } else if ('awaitDblClick' in handle.elem.dataset) {
                    clearTimeout(dblclick_timer);
                    dblclick_timer = null;
                    delete handle.elem.dataset.awaitDblClick;
                    handle.elem.dispatchEvent(
                        new CustomEvent('resize_handle_dblclick', {
                            bubbles: true,
                            detail: this,
                        })
                    );
                }
                prev = null;
                handle = null;
                next = null;
            },
            {signal: this.event_abort_controller.signal}
        );

        window.addEventListener(
            'pointerout',
            (event) => {
                if (event.pointerType === 'touch') {
                    touch_count--;
                }
            },
            {signal: this.event_abort_controller.signal}
        );

        this.resize_observer = new ResizeObserver((entries) => {
            for (const entry of entries) {
                if (entry.target.id === this.elem.id) {
                    clearTimeout(this.resize_observer_timer);
                    this.resize_observer_timer = setTimeout(() => {
                        this.onResize();
                    }, RESIZE_DEBOUNCE_TIME_MS);
                }
            }
        });
        this.resize_observer.observe(this.elem);
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

    onMove(event, prev, handle, next) {
        /** Handles pointermove events. */
        const par_dims = this.elem.getBoundingClientRect();
        const a_dims = prev.elem.getBoundingClientRect();
        const b_dims = next.elem.getBoundingClientRect();
        let pos = handle.axis === 0 ? parseInt(event.y) : parseInt(event.x);
        pos = Math.min(
            handle.axis === 0 ? parseInt(par_dims.height) : parseInt(par_dims.width),
            pos
        );
        const a_start =
            handle.axis === 0 ? parseInt(a_dims.top) : parseInt(a_dims.left);
        const b_end =
            handle.axis === 0 ? parseInt(b_dims.bottom) : parseInt(b_dims.right);

        let a = pos - Math.floor(handle.pad_px / 2);
        let b = pos + Math.floor(handle.pad_px / 2);

        let a_lim = a_start + prev.min_size;
        let b_lim = b_end - next.min_size;

        if (a < a_lim) {
            a = a_lim;
            b = a + handle.pad_px;
        }

        if (b > b_lim) {
            b = b_lim;
            a = b - handle.pad_px;
        }

        prev.setSize(a - a_start);
        next.setSize(b_end - b);
    }

    onResize() {
        /** Resizes grid items on resize observer events. */
        const curr_dims = this.elem.getBoundingClientRect();
        const d_w = curr_dims.width - this.prev_dims.width;
        const d_h = curr_dims.height - this.prev_dims.height;

        // If no change to size, don't proceed.
        if (d_w === 0 && d_h === 0) {
            return;
        }

        if (d_w < 0) {
            // Width decrease
            for (const row of this.container.grid) {
                let rem = Math.abs(d_w);
                for (const col of row.slice().reverse()) {
                    const flex_grow = parseInt(col.elem.style.flexGrow);
                    rem = col.shrink(rem, {limit_to_base: false});
                    // Shrink causes flexGrow to be set back to default.
                    // We want to keep the current flex grow setting so we set it back.
                    col.render({force_flex_grow: flex_grow === 0 ? false : true});
                    if (rem <= 0) {
                        break;
                    }
                }
            }
        } else if (d_w > 0) {
            // width increase
            for (const row of this.container.grid) {
                for (const col of row.slice().reverse()) {
                    const amt = col.grow(-1, {only_if_flex: true});
                    if (amt === 0) {
                        break;
                    }
                }
            }
        }

        if (d_h < 0) {
            // height decrease
            let rem = Math.abs(d_h);
            for (const row of this.container.rows.slice().reverse()) {
                const flex_grow = parseInt(row.elem.style.flexGrow);
                rem = row.shrink(rem, {limit_to_base: false});
                // Shrink causes flexGrow to be set back to default.
                // We want to keep the current flex grow setting so we set it back.
                row.render({force_flex_grow: flex_grow === 0 ? false : true});
                if (rem <= 0) {
                    break;
                }
            }
        } else if (d_h > 0) {
            // height increase
            for (const row of this.container.rows.slice().reverse()) {
                if (row.grow(-1, {only_if_flex: true}) === 0) {
                    break;
                }
            }
        }

        this.prev_dims = curr_dims;
    }

    show({row_idx, col_idx} = {}) {
        /** Show a row or column.
         * Columns require both the row_idx and col_idx.
         */
        this.container.show({row_idx: row_idx, col_idx: col_idx});
        this.container.updateVisibleHandles();
    }

    hide({row_idx, col_idx} = {}) {
        /** Hide a row or column.
         * Columns require both the row_idx and col_idx.
         */
        this.container.hide({row_idx: row_idx, col_idx: col_idx});
        this.container.updateVisibleHandles();
    }

    toggle({row_idx, col_idx, override} = {}) {
        /** Toggle visibility of a row or column.
         * Columns require both the row_idx and col_idx.
         */
        this.container.toggle({
            row_idx: row_idx,
            col_idx: col_idx,
            override: override,
        });
        this.container.updateVisibleHandles();
    }

    toggleElem(elem, override) {
        /** Toggles the nearest ResizeGridItem to the passed element. */
        isElementThrowError(elem);
        let _elem = elem.closest('.resize-grid--col');
        if (isElement(_elem)) {
            this.toggle({
                row_idx: _elem.dataset.row,
                col_idx: _elem.dataset.col,
                override: override,
            });
        } else {
            _elem = elem.closest('.resize-grid--row');
            isElementThrowError(_elem);
            this.toggle({row_idx: _elem.dataset.row, override: override});
        }
    }
}

function resizeGridGetGrid(elem) {
    /** Returns the nearest ResizeGrid to the passed element. */
    // Need to find grid element so we can lookup by its id.
    const grid_elem = elem.closest('.resize-grid');
    if (!isElement(grid_elem)) {
        return null;
    }
    // Now try to get the actual ResizeGrid instance.
    const grid = resize_grids[grid_elem.dataset.gridId];
    if (isNullOrUndefined(grid)) {
        return null;
    }

    return grid;
}

function resizeGridSetup(elem, {id} = {}) {
    /** Sets up a new ResizeGrid instance for the provided element. */
    isElementThrowError(elem);
    if (!isString(id)) {
        id = _get_unique_id();
    }
    const grid = new ResizeGrid(id, elem);
    grid.setup();
    resize_grids[id] = grid;
    return grid;
}
