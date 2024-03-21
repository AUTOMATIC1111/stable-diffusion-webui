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

const calcColsPerRow = function (parent) {
    // Returns the number of columns in a row of a flexbox.
    //const parent = document.querySelector(selector);
    const parent_width = getComputedValue(parent, "width");
    const parent_padding_left = getComputedValue(parent, "padding-left");
    const parent_padding_right = getComputedValue(parent, "padding-right");

    const child = parent.firstElementChild;
    const child_width = getComputedValue(child, "width");
    const child_margin_left = getComputedValue(child, "margin-left");
    const child_margin_right = getComputedValue(child, "margin-right");

    var parent_width_no_padding = parent_width - parent_padding_left - parent_padding_right;
    const child_width_with_margin = child_width + child_margin_left + child_margin_right;
    parent_width_no_padding += child_margin_left + child_margin_right;

    return parseInt(parent_width_no_padding / child_width_with_margin);
}

const calcRowsPerCol = function (container, parent) {
    // Returns the number of columns in a row of a flexbox.
    //const parent = document.querySelector(selector);
    const parent_height = getComputedValue(container, "height");
    const parent_padding_top = getComputedValue(container, "padding-top");
    const parent_padding_bottom = getComputedValue(container, "padding-bottom");

    const child = parent.firstElementChild;
    const child_height = getComputedValue(child, "height");
    const child_margin_top = getComputedValue(child, "margin-top");
    const child_margin_bottom = getComputedValue(child, "margin-bottom");

    var parent_height_no_padding = parent_height - parent_padding_top - parent_padding_bottom;
    const child_height_with_margin = child_height + child_margin_top + child_margin_bottom;
    parent_height_no_padding += child_margin_top + child_margin_bottom;

    return parseInt(parent_height_no_padding / child_height_with_margin);
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
        return waitForElement(`#${this.data_id}`)
            .then((elem) => this.data_elem = elem)
            .then(() => this.parseJson(this.data_elem.dataset.json))
            .then(() => this.init())
            .then(() => this.repair())
            .then(() => this.applyFilter());
    }

    parseJson(encoded_str) { /** promise */
        return new Promise(resolve => {
            // Skip parsing if the string hasnt actually updated.
            if (this.encoded_str === encoded_str) {
                console.log("no change");
                return resolve();
            }
            return resolve(
                Promise.resolve(encoded_str)
                    .then(v => decompress(v))
                    .then(v => JSON.parse(v))
                    .then(v => this.updateJson(v))
                    .then(() => {console.log("parse json done"); this.encoded_str = encoded_str; })
            );
        });
    }

    updateJson(json) { /** promise */
        /** Must be overridden by inherited class. */
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

        this.refresh();

        // Rebuild with `force=false` so we only rebuild if dimensions change.
        this.rebuild(false);
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
        
        // Calculate the visible rows and colums for the clusterize-content area.
        let n_cols = calcColsPerRow(this.content_elem);
        let n_rows = calcRowsPerCol(this.scroll_elem, this.content_elem);
        n_cols = (isNaN(n_cols) || n_cols <= 0) ? 1 : n_cols;
        n_rows = (isNaN(n_rows) || n_rows <= 0) ? 1 : n_rows;

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
            console.log("rebuild:: init");
            this.init();
        } else if (this.recalculateDims() || force) {
            console.log("rebuild:: full", this.scroll_id);
            this.destroy();
            this.clusterize = null;
            this.init();
        } else {
            console.log("rebuild:: update", this.scroll_id);
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
        console.log("element resized:", elem_id);
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
        console.log("onElementAdded::", elem_id, document.body.contains(this.scroll_elem));
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
        console.log("onElementRemoved::", elem_id, document.body.contains(this.scroll_elem));
    }

    onDataChanged(data) {
        console.log("onDataChanged::", this.data_id);
        this.parseJson(data);
    }

    setupElementObservers() {
        this.element_observer = new MutationObserver((mutations) => {
            for (const mutation of mutations) {
                if (mutation.type === "childList") {
                    // added
                    if (mutation.addedNodes.length > 0) {
                        for (const node of mutation.addedNodes) {
                            if (node.id === this.data_id || node.id === this.scroll_id || node.id === this.content_id) {
                                this.onElementAdded(node.id);
                            }
                        }
                    }
                    // removed
                    if (mutation.removedNodes.length > 0) {
                        for (const node of mutation.removedNodes) {
                            if (node.id === this.data_id || node.id === this.scroll_id || node.id === this.content_id) {
                                this.onElementRemoved(node.id);
                            }
                        }
                    }
                } else if (mutation.type === "attributes") {
                    if (mutation.target.id === this.data_id && mutation.attributeName === "data-json") {
                        this.onDataChanged(mutation.target.dataset.json);
                    }
                }
            }
        });
        this.element_observer.observe(gradioApp(), {subtree: true, childList: true, attributes: true});
    }

    setupResizeHandlers() {
        this.resize_observer = new ResizeObserver((entries) => {
            for (const entry of entries) {
                console.log("resizeObserver:", entry.target.id);
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
            //this.applyFilter();
            console.log("updateJson:: done", this.scroll_id);
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
            //this.applyFilter();
            console.log("updateJson:: done", this.scroll_id);
            if (this.scroll_id.includes("textual")) { console.log(this.data_obj); }
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
}
