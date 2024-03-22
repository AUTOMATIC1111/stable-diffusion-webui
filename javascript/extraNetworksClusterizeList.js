// Collators used for sorting.
const INT_COLLATOR = new Intl.Collator([], { numeric: true });
const STR_COLLATOR = new Intl.Collator("en", { numeric: true, sensitivity: "base" });

const isString = x => typeof x === "string" || x instanceof String;
const isStringLogError = x => {
    if (isString(x)) {
        return true;
    }
    console.error("expected string, got:", typeof x);
    return false;
};
const isNull = x => typeof x === "null" || x === null;
const isUndefined = x => typeof x === "undefined" || x === undefined;
// checks both null and undefined for simplicity sake.
const isNullOrUndefined = x => isNull(x) || isUndefined(x);
const isNullOrUndefinedLogError = x => {
    if (isNullOrUndefined(x)) {
        return true;
    }
    console.error("Variable is null/undefined.");
    return false;
};
const isElement = x => x instanceof Element;
const isElementLogError = x => {
    if (isElement(x)) {
        return true;
    }
    console.error("expected element type, got:", typeof x);
    return false;
}

const getElementByIdLogError = x => {
    let elem = gradioApp().getElementById(x);
    isElementLogError(elem);
    return elem;
};

const querySelectorLogError = x => {
    let elem = gradioApp().querySelector(x);
    isElementLogError(elem);
    return elem;
}

const getComputedPropertyDims = (elem, prop) => {
    /** Returns the top/left/bottom/right float dimensions of an element for the specified property. */
    const style = window.getComputedStyle(elem, null);
    return {
        top: parseFloat(style.getPropertyValue(`${prop}-top`)),
        left: parseFloat(style.getPropertyValue(`${prop}-left`)),
        bottom: parseFloat(style.getPropertyValue(`${prop}-bottom`)),
        right: parseFloat(style.getPropertyValue(`${prop}-right`)),
    };
}

const getComputedMarginDims = elem => {
    const dims = getComputedPropertyDims(elem, "margin");
    return {
        width: dims.left + dims.right,
        height: dims.top + dims.bottom,
    };
}

const getComputedPaddingDims = elem => {
    const dims = getComputedPropertyDims(elem, "padding");
    return {
        width: dims.left + dims.right,
        height: dims.top + dims.bottom,
    };
}

const getComputedBorderDims = elem => {
    // computed border will always start with the pixel width so thankfully
    // the parseFloat() conversion will just give us the width and ignore the rest.
    // Otherwise we'd have to use border-<pos>-width instead.
    const dims = getComputedPropertyDims(elem, "border");
    return {
        width: dims.left + dims.right,
        height: dims.top + dims.bottom,
    };
}

const getComputedDims = elem => {
    /** Returns the full width and height of an element including its margin, padding, and border. */
    const width = elem.scrollWidth;
    const height = elem.scrollHeight;
    const margin = getComputedMarginDims(elem);
    const padding = getComputedPaddingDims(elem);
    const border = getComputedBorderDims(elem);
    return {
        width: width + margin.width + padding.width + border.width,
        height: height + margin.height + padding.height + border.height,
    }
    
}

function compress(string) {
    /** Compresses a string into a base64 encoded GZipped string. */
    const cs = new CompressionStream('gzip');
    const writer = cs.writable.getWriter();

    const blobToBase64 = blob => new Promise((resolve, _) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result.split(',')[1]);
        reader.readAsDataURL(blob);
    });
    const byteArray = new TextEncoder().encode(string);
    writer.write(byteArray);
    writer.close();
    return new Response(cs.readable).blob().then(blobToBase64);
};

function decompress(base64string) {
    /** Decompresses a base64 encoded GZipped string. */
    const ds = new DecompressionStream('gzip');
    const writer = ds.writable.getWriter();
    const bytes = Uint8Array.from(atob(base64string), c => c.charCodeAt(0));
    writer.write(bytes);
    writer.close();
    return new Response(ds.readable).arrayBuffer().then(function (arrayBuffer) {
        return new TextDecoder().decode(arrayBuffer);
    });
}

const parseHtml = function (str) {
    const tmp = document.implementation.createHTMLDocument('');
    tmp.body.innerHTML = str;
    return [...tmp.body.childNodes];
}

const getComputedValue = function (container, css_property) {
    return parseInt(
        window.getComputedStyle(container, null)
            .getPropertyValue(css_property)
            .split("px")[0]
    );
};

const calcColsPerRow = function (parent, child) {
    /** Calculates the number of columns of children that can fit in a parent's visible width. */
    const parent_inner_width = parent.offsetWidth - getComputedPaddingDims(parent).width;
    return parseInt(parent_inner_width / getComputedDims(child).width);

}

const calcRowsPerCol = function (parent, child) {
    /** Calculates the number of rows of children that can fit in a parent's visible height. */
    const parent_inner_height = parent.offsetHeight - getComputedPaddingDims(parent).height;
    return parseInt(parent_inner_height / getComputedDims(child).height);
}

class ExtraNetworksClusterize {
    /** Base class for a clusterize list. Cannot be used directly. */
    constructor(
        {
            data_id,
            scroll_id,
            content_id,
            txt_search_elem,
            rows_in_block = 10,
            blocks_in_cluster = 4,
            show_no_data_row = true,
            callbacks = {},
        } = {
                rows_in_block: 10,
                blocks_in_cluster: 4,
                show_no_data_row: true,
                callbacks: {},
            }
    ) {
        // Do not continue if any of the required parameters are invalid.
        if (!isStringLogError(data_id)) { return; }
        if (!isStringLogError(scroll_id)) { return; }
        if (!isStringLogError(content_id)) { return; }
        if (!isElementLogError(txt_search_elem)) { return; }

        this.data_id = data_id;
        this.scroll_id = scroll_id;
        this.content_id = content_id;
        this.txt_search_elem = txt_search_elem;
        this.rows_in_block = rows_in_block;
        this.blocks_in_cluster = blocks_in_cluster;
        this.show_no_data_row = show_no_data_row;
        this.callbacks = callbacks;

        this.clusterize = null;

        this.data_elem = null;
        this.scroll_elem = null;
        this.content_elem = null;

        this.resize_observer = null;
        this.resize_observer_timer = null;
        this.resize_observer_timeout_ms = 250;
        this.element_observer = null;
        this.data_update_timer = null
        this.data_update_timeout_ms = 1000;

        this.enabled = false;

        this.encoded_str = "";

        this.no_data_text = "Directory is empty.";
        this.no_data_class = "nocards";

        this.n_rows = 1;
        this.n_cols = 1;

        this.data_obj = {};
        this.data_obj_keys_sorted = [];

        this.sort_fn = this.sortByDivId;
        this.sort_reverse = false;

        Promise.all([
            waitForElement(`#${this.data_id}`).then((elem) => this.data_elem = elem),
            waitForElement(`#${this.scroll_id}`).then((elem) => this.scroll_elem = elem),
            waitForElement(`#${this.content_id}`).then((elem) => this.content_elem = elem),
        ]).then(() => {
            this.setupElementObservers();
            this.setupResizeHandlers();
        });
    }

    enable(enabled) {
        if (enabled === undefined || enabled === null) {
            this.enabled = true;
        } else if (typeof enabled !== "boolean") {
            console.error("Invalid type. Expected boolean, got", typeof enabled);
        } else {
            this.enabled = enabled;
        }
    }

    load() {
        return new Promise(resolve => {
            waitForElement(`#${this.data_id}`)
                .then((elem) => this.data_elem = elem)
                .then(() => this.parseJson(this.data_elem.dataset.json))
                .then(() => { return resolve(); });
        });
    }

    parseJson(encoded_str) { /** promise */
        return new Promise(resolve => {
            // Skip parsing if the string hasnt actually updated.
            if (this.encoded_str === encoded_str) {
                return resolve();
            }
            Promise.resolve(encoded_str)
                .then(v => decompress(v))
                .then(v => JSON.parse(v))
                .then(v => this.updateJson(v))
                .then(() => this.encoded_str = encoded_str)
                .then(() => this.init())
                .then(() => this.repair())
                .then(() => this.applyFilter())
                .then(() => { return resolve(); });
        });
    }

    updateJson(json) { /** promise */
        /** Must be overridden by inherited class. */
        console.error("Base class method called. Must be overridden by child.");
        return new Promise(resolve => {return resolve();});
    }

    sortByDivId() {
        // Sort data_obj keys (div_id) as numbers.
        this.data_obj_keys_sorted = Object.keys(this.data_obj).sort((a, b) => INT_COLLATOR.compare(a, b));
    }

    applySort() {
        this.sort_fn()
        if (this.sort_reverse) {
            this.data_obj_keys_sorted = this.data_obj_keys_sorted.reverse();
        }
    }

    applyFilter() {
        /** Must be implemented by subclasses. */
        this.applySort();
        this.updateRows();
    }

    filterRows(obj) {
        var results = [];
        for (const div_id of this.data_obj_keys_sorted) {
            if (obj[div_id].active) {
                results.push(obj[div_id].element.outerHTML);
            }
        }
        return results;
    }

    updateDivContent(div_id, content) {
        /** Updates an element in the dataset. Does not call update_rows(). */
        if (!(div_id in this.data_obj)) {
            console.error("div_id not in data_obj:", div_id);
        } else if (typeof content === "object") {
            this.data_obj[div_id].element = parseHtml(content.outerHTML)[0];
            return true;
        } else if (typeof content === "string") {
            this.data_obj[div_id].element = parseHtml(content)[0];
            return true;
        } else {
            console.error("Invalid content:", div_id, content);
        }

        return false;
    }

    updateRows() {
        // If we don't have any entries in the dataset, then just clear the list and return.
        if (this.data_obj_keys_sorted.length === 0 || Object.keys(this.data_obj).length === 0) {
            return;
        }

        this.refresh(true);

        // Rebuild with `force=false` so we only rebuild if dimensions change.
        this.rebuild(false);
    }

    getMaxRowWidth() {
        // impliment in subclasses
        return;
    }

    recalculateDims() {
        let rebuild_required = false;
        let clear_before_return = false;

        if (!this.enabled) {
            // Inactive list is not displayed on screen. Would error if trying to resize.
            return false;
        }
        if (Object.keys(this.data_obj).length === 0 || this.data_obj_keys_sorted.length === 0) {
            // If there is no data then just skip.
            return false;
        }

        // If no rows exist, we need to add one so we can calculate rows/cols.
        // We remove this row before returning.
        if (this.rowCount() === 0){// || this.content_elem.innerHTML === "") {
            this.clear();
            this.update([this.data_obj[this.data_obj_keys_sorted[0]].element.outerHTML]);
            clear_before_return = true;
        }
        
        const child = this.content_elem.querySelector(":not(.clusterize-extra-row)");
        if (isNullOrUndefined(child)) {
            if (clear_before_return) {
                this.clear();
                return rebuild_required;
            }
        }
        
        // Calculate the visible rows and colums for the clusterize-content area.
        let n_cols = calcColsPerRow(this.content_elem, child);
        let n_rows = calcRowsPerCol(this.scroll_elem, child);
        n_cols = (isNaN(n_cols) || n_cols <= 0) ? 1 : n_cols;
        n_rows = (isNaN(n_rows) || n_rows <= 0) ? 1 : n_rows;

        // Add two extra rows to account for partial row visibility on top and bottom
        // of the content element view region.
        n_rows += 2;

        if (n_cols != this.n_cols || n_rows != this.n_rows) {
            // Sizes have changed. Update the instance values.
            this.n_cols = n_cols;
            this.n_rows = n_rows;
            this.rows_in_block = this.n_rows;
            rebuild_required = true;
        }

        // If we added a temporary row earlier, remove before returning.
        if (clear_before_return) {
            this.clear();
        }

        return rebuild_required;
    }

    waitForElements() {
        return new Promise(resolve => {
            Promise.all([
                waitForElement(`#${this.data_id}`),
                waitForElement(`#${this.scroll_id}`),
                waitForElement(`#${this.content_id}`),
            ]).then(() => {
                return resolve();
            });
        });
    }

    repair() {
        /** Fixes element association in DOM. Returns true if element was replaced in DOM. */
        // If association for elements is broken, replace them with instance version.
        if (!this.scroll_elem.isConnected || !this.content_elem.isConnected) {
            gradioApp().getElementById(this.scroll_id).replaceWith(this.scroll_elem);
            // Fix resize observers since they are bound to each element individually.
            if (!isNullOrUndefined(this.resize_observer)) {
                this.resize_observer.disconnect();
                this.resize_observer.observe(this.scroll_elem);
                this.resize_observer.observe(this.content_elem);
            }
            // Make sure to refresh forcefully after updating the dom.
            this.refresh(true);
            return true;
        }
        return false;
    }

    rebuild(force) {
        // Only accept boolean values for `force` parameter. Default to false.
        if (force !== true) {
            force = false;
        }

        if (isNullOrUndefined(this.clusterize)) {
            // If we have already initialized, don't do it again.
            this.init();
        } else if (this.recalculateDims() || force) {
            this.destroy();
            this.clusterize = null;
            this.init();
        } else {
            this.update();
        }
    }

    init(rows) {
        if (!isNullOrUndefined(this.clusterize)) {
            // If we have already initialized, don't do it again.
            return;
        }

        if (isNullOrUndefined(rows) && isNullOrUndefined(this.data_obj)) {
            // data hasnt been loaded yet and we arent provided any. skip.
            return;
        }

        if (isNullOrUndefined(rows)) {
            rows = this.data_obj;
        } else if (Array.isArray(rows) && !(rows.every(row => isString(row)))) {
            console.error("Invalid data type for rows. Expected array[string].");
            return;
        }

        this.clusterize = new Clusterize(
            {
                rows: this.filterRows(rows),
                scrollId: this.scroll_id,
                contentId: this.content_id,
                rows_in_block: this.rows_in_block,
                blocks_in_cluster: this.blocks_in_cluster,
                show_no_data_row: this.show_no_data_row,
                no_data_text: this.no_data_text,
                no_data_class: this.no_data_class,
                callbacks: this.callbacks,
            }
        );
    }

    onResize(elem_id) {
        this.updateRows();
    }

    onElementAdded(elem_id) {
        switch(elem_id) {
            case this.data_id:
                waitForElement(`#${this.data_id}`).then((elem) => this.data_elem = elem);
                break;
            case this.scroll_id:
                this.repair();
                break;
            case this.content_id:
                this.repair();
                break;
            default:
                break;
        }
    }

    onElementRemoved(elem_id) {
        switch(elem_id) {
            case this.data_id:
                waitForElement(`#${this.data_id}`).then((elem) => this.data_elem = elem);
                break;
            case this.scroll_id:
                this.repair();
                break;
            case this.content_id:
                this.repair();
                break;
            default:
                break;
        }
    }

    onElementUpdated(elem_id) {
        switch(elem_id) {
            case this.data_id:
                waitForElement(`#${this.data_id}`).then((elem) => this.data_elem = elem);
                break;
            case this.scroll_id:
                this.repair();
                break;
            case this.content_id:
                this.repair();
                break;
            default:
                break;
        }
    }

    onDataChanged(data) {
        this.parseJson(data);
    }

    setupElementObservers() {
        this.element_observer = new MutationObserver((mutations) => {
            // don't waste time if this object isn't enabled.
            if (!this.enabled) {
                return;
            }

            let data_elem = gradioApp().getElementById(this.data_id);
            if (data_elem && data_elem !== this.data_elem) {
                this.onElementUpdated(data_elem.id);
            } else if (data_elem && data_elem.dataset.json !== this.encoded_str) {
                // we don't want to get blasted with data updates so just wait for
                // the data to settle down before updating.
                clearTimeout(this.data_update_timer);
                this.data_update_timer = setTimeout(() => {
                    this.onDataChanged(data_elem.dataset.json);
                }, this.data_update_timeout_ms);
            }

            let scroll_elem = gradioApp().getElementById(this.scroll_id);
            if (scroll_elem && scroll_elem !== this.scroll_elem) {
                this.onElementUpdated(scroll_elem.id);
            }

            let content_elem = gradioApp().getElementById(this.content_id);
            if (content_elem && content_elem !== this.content_elem) {
                this.onElementUpdated(content_elem.id);
            }
        });
        this.element_observer.observe(gradioApp(), {subtree: true, childList: true, attributes: true});
    }

    setupResizeHandlers() {
        // Handle element resizes. Delay of `resize_observer_timeout_ms` after resize 
        // before firing an event as a way of "debouncing" resizes.
        this.resize_observer = new ResizeObserver((entries) => {
            for (const entry of entries) {
                if (entry.target.id === this.scroll_id || entry.target.id === this.content_id) {
                    clearTimeout(this.resize_observer_timer);
                    this.resize_observer_timer = setTimeout(() => this.onResize(entry.id), this.resize_observer_timeout_ms);
                }
            }
        });

        this.resize_observer.observe(this.scroll_elem);
        this.resize_observer.observe(this.content_elem);
    }

    /* ==== Clusterize.Js FUNCTION WRAPPERS ==== */
    refresh(force) {
        /** Refreshes the clusterize instance so that it can recalculate its dims. */
        if (isNullOrUndefined(this.clusterize)) {
            return;
        }

        // Only allow boolean variables. default to false.
        if (force !== true) {
            force = false;
        }
        this.clusterize.refresh(force);
    }

    rowCount() {
        /** Gets the total (not only visible) row count in the clusterize instance. */
        return this.clusterize.getRowsAmount();
    }

    clear() {
        /** Removes all rows. */
        this.clusterize.clear();
    }

    update(rows) {
        /** Adds rows from a list of element strings. */
        if (rows === undefined || rows === null) {
            // If not passed, use the default method of getting rows.
            rows = this.filterRows(this.data_obj);
        } else if (!Array.isArray(rows) || !(rows.every(row => typeof row === "string"))) {
            console.error("Invalid data type for rows. Expected array[string].");
            return;
        }
        this.clusterize.update(rows);
    }

    destroy() {
        /** Destroys a clusterize instance and removes its rows from the page. */
        // If `true` isnt passed, then clusterize dumps every row to the DOM.
        // This kills performance so we never want to do this.
        this.clusterize.destroy(true);
    }
}

class ExtraNetworksClusterizeTreeList extends ExtraNetworksClusterize {
    constructor(...args) {
        super(...args);
    }

    getBoxShadow(depth) {
        // Generates style for a multi-level box shadow for vertical indentation lines.
        let res = "";
        var style = getComputedStyle(document.body);
        let bg = style.getPropertyValue("--body-background-fill");
        let fg = style.getPropertyValue("--border-color-primary");
        let text_size = style.getPropertyValue("--button-large-text-size");
        for (let i = 1; i <= depth; i++) {
            res += `calc((${i} * ${text_size}) - (${text_size} * 0.6)) 0 0 ${bg} inset,`;
            res += `calc((${i} * ${text_size}) - (${text_size} * 0.4)) 0 0 ${fg} inset`;
            res += (i + 1 > depth) ? "" : ", ";
        }
        return res;
    }

    updateJson(json) {
        return new Promise(resolve => {
            var style = getComputedStyle(document.body);
            //let spacing_sm = style.getPropertyValue("--spacing-sm");
            let text_size = style.getPropertyValue("--button-large-text-size");
            for (const [k, v] of Object.entries(json)) {
                let div_id = k;
                let parsed_html = parseHtml(v)[0];
                // parent_id = -1 if item is at root level
                let parent_id = "parentId" in parsed_html.dataset ? parsed_html.dataset.parentId : -1;
                let expanded = "expanded" in parsed_html.dataset;
                let depth = Number(parsed_html.dataset.depth);
                parsed_html.style.paddingLeft = `calc(${depth} * ${text_size})`;
                parsed_html.style.boxShadow = this.getBoxShadow(depth);

                // Add the updated html to the data object.
                this.data_obj[div_id] = {
                    element: parsed_html,
                    active: parent_id === -1, // always show root
                    expanded: expanded || (parent_id === -1), // always expand root
                    parent: parent_id,
                    children: [], // populated later
                };
            }

            // Build list of children for each element in dataset.
            for (const [k, v] of Object.entries(this.data_obj)) {
                if (v.parent === -1) {
                    continue;
                } else if (!(v.parent in this.data_obj)) {
                    console.error("parent not in data:", v.parent);
                } else {
                    this.data_obj[v.parent].children.push(k);
                }
            }

            // Handle expanding of rows on initial load
            for (const [k, v] of Object.entries(this.data_obj)) {
                if (v.parent === -1) {
                    // Always show root level.
                    this.data_obj[k].active = true;
                } else if (this.data_obj[v.parent].expanded && this.data_obj[v.parent].active) {
                    // Parent is both active and expanded. show child
                    this.data_obj[k].active = true;
                } else {
                    this.data_obj[k].active = false;
                }
            }
            return resolve();
        });
    }

    removeChildRows(div_id) {
        for (const child_id of this.data_obj[div_id].children) {
            this.data_obj[child_id].active = false;
            this.data_obj[child_id].expanded = false;
            delete this.data_obj[child_id].element.dataset.expanded;
            this.removeChildRows(child_id);
        }
    }

    addChildRows(div_id) {
        for (const child_id of this.data_obj[div_id].children) {
            this.data_obj[child_id].active = true;
            if (this.data_obj[child_id].expanded) {
                this.addChildRows(child_id);
            }
        }
    }

    getMaxRowWidth() {
        if (!this.enabled) {
            // Inactive list is not displayed on screen. Can't calculate size.
            return false;
        }
        if (this.rowCount() === 0) {
            // If there is no data then just skip.
            return false;
        }

        let max_width = 0;
        for (let i = 0; i < this.content_elem.children.length; i += this.n_cols) {
            let row_width = 0;
            for (let j = 0; j < this.n_cols; j++) {
                const child = this.content_elem.children[i + j];
                const child_style = window.getComputedStyle(child, null);
                const prev_style = child.style.cssText;
                const n_cols = child_style.getPropertyValue("grid-template-columns").split(" ").length;
                child.style.gridTemplateColumns = `repeat(${n_cols}, max-content)`;
                row_width += getComputedDims(child).width;
                // Re-apply previous style.
                child.style.cssText = prev_style;
            }
            max_width = Math.max(max_width, row_width);
        }
        if (max_width <= 0) {
            return;
        }

        // Add the container's padding to the result.
        max_width += getComputedPaddingDims(this.content_elem).width;
        // Add the scrollbar's width to the result. Will add 0 if scrollbar isnt present.
        max_width += this.scroll_elem.offsetWidth - this.scroll_elem.clientWidth;
        return max_width;
    }
}

class ExtraNetworksClusterizeCardsList extends ExtraNetworksClusterize {
    constructor(...args) {
        super(...args);

        this.sort_mode_str = "path";
        this.sort_dir_str = "ascending";
        this.filter_str = "";
    }

    updateJson(json) {
        return new Promise(resolve => {
            for (const [k, v] of Object.entries(json)) {
                let div_id = k;
                let parsed_html = parseHtml(v)[0];
                // Add the updated html to the data object.
                this.data_obj[div_id] = {
                    element: parsed_html,
                    active: true,
                };
            }
            return resolve();
        });
    }

    filterRows(obj) {
        let filtered_rows = super.filterRows(obj);
        let res = [];
        for (let i = 0; i < filtered_rows.length; i += this.n_cols) {
            res.push(filtered_rows.slice(i, i + this.n_cols).join(""));
        }
        return res;
    }

    sortByName() {
        this.data_obj_keys_sorted = Object.keys(this.data_obj).sort((a, b) => {
            return STR_COLLATOR.compare(
                this.data_obj[a].element.dataset.sortName,
                this.data_obj[b].element.dataset.sortName,
            );
        });
    }

    sortByPath() {
        this.data_obj_keys_sorted = Object.keys(this.data_obj).sort((a, b) => {
            return STR_COLLATOR.compare(
                this.data_obj[a].element.dataset.sortPath,
                this.data_obj[b].element.dataset.sortPath,
            );
        });
    }

    sortByCreated() {
        this.data_obj_keys_sorted = Object.keys(this.data_obj).sort((a, b) => {
            return INT_COLLATOR.compare(
                this.data_obj[a].element.dataset.sortCreated,
                this.data_obj[b].element.dataset.sortCreated,
            );
        });
    }

    sortByModified() {
        this.data_obj_keys_sorted = Object.keys(this.data_obj).sort((a, b) => {
            return INT_COLLATOR.compare(
                this.data_obj[a].element.dataset.sortModified,
                this.data_obj[b].element.dataset.sortModified,
            );
        });
    }

    setSortMode(btn_sort_mode) {
        this.sort_mode_str = btn_sort_mode.dataset.sortMode.toLowerCase();
    }

    setSortDir(btn_sort_dir) {
        this.sort_dir_str = btn_sort_dir.dataset.sortDir.toLowerCase();
    }

    applySort() {
        this.sort_reverse = this.sort_dir_str === "descending";

        switch (this.sort_mode_str) {
            case "name":
                this.sort_fn = this.sortByName;
                break;
            case "path":
                this.sort_fn = this.sortByPath;
                break;
            case "created":
                this.sort_fn = this.sortByCreated;
                break;
            case "modified":
                this.sort_fn = this.sortByModified;
                break;
            default:
                this.sort_fn = this.sortByDivId;
                break;
        }
        super.applySort();
    }

    applyFilter(filter_str) {
        if (filter_str !== undefined && filter_str !== null) {
            this.filter_str = filter_str.toLowerCase();
        } else {
            this.filter_str = this.txt_search_elem.value.toLowerCase();
        }

        for (const [k, v] of Object.entries(this.data_obj)) {
            let search_only = v.element.querySelector(".search_only");
            let text = Array.prototype.map.call(v.element.querySelectorAll(".search_terms"), function (t) {
                return t.textContent.toLowerCase();
            }).join(" ");

            let visible = text.indexOf(this.filter_str) != -1;
            if (search_only && this.filter_str.length < 4) {
                visible = false;
            }
            this.data_obj[k].active = visible;
        }

        this.applySort();
        this.updateRows();
    }

    getMaxRowWidth() {
        if (!this.enabled) {
            // Inactive list is not displayed on screen. Can't calculate size.
            return false;
        }
        if (this.rowCount() === 0) {
            // If there is no data then just skip.
            return false;
        }

        let max_width = 0;
        for (let i = 0; i < this.content_elem.children.length; i += this.n_cols) {
            let row_width = 0;
            for (let j = 0; j < this.n_cols; j++) {
                row_width += getComputedDims(this.content_elem.children[i + j]).width;
            }
            max_width = Math.max(max_width, row_width);
        }
        if (max_width <= 0) {
            return;
        }

        // Add the container's padding to the result.
        max_width += getComputedPaddingDims(this.content_elem).width;
        // Add the scrollbar's width to the result. Will add 0 if scrollbar isnt present.
        max_width += this.scroll_elem.offsetWidth - this.scroll_elem.clientWidth;
        return max_width;
    }
}
