// Prevent eslint errors on functions defined in other files.
/*global
    isString,
    isStringLogError,
    isNull,
    isUndefined,
    isNullOrUndefined,
    isNullOrUndefinedLogError,
    isElement,
    isElementLogError,
    getElementByIdLogError,
    querySelectorLogError,
    waitForElement,
    Clusterize
*/
/*eslint no-undef: "error"*/

const JSON_UPDATE_DEBOUNCE_TIME_MS = 250;
const RESIZE_DEBOUNCE_TIME_MS = 250;
// Collators used for sorting.
const INT_COLLATOR = new Intl.Collator([], {numeric: true});
const STR_COLLATOR = new Intl.Collator("en", {numeric: true, sensitivity: "base"});

class InvalidCompressedJsonDataError extends Error {
    constructor(message, options) {
        super(message, options);
    }
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
};

const getComputedMarginDims = elem => {
    /** Returns the width/height of the computed margin of an element. */
    const dims = getComputedPropertyDims(elem, "margin");
    return {
        width: dims.left + dims.right,
        height: dims.top + dims.bottom,
    };
};

const getComputedPaddingDims = elem => {
    /** Returns the width/height of the computed padding of an element. */
    const dims = getComputedPropertyDims(elem, "padding");
    return {
        width: dims.left + dims.right,
        height: dims.top + dims.bottom,
    };
};

const getComputedBorderDims = elem => {
    /** Returns the width/height of the computed border of an element. */
    // computed border will always start with the pixel width so thankfully
    // the parseFloat() conversion will just give us the width and ignore the rest.
    // Otherwise we'd have to use border-<pos>-width instead.
    const dims = getComputedPropertyDims(elem, "border");
    return {
        width: dims.left + dims.right,
        height: dims.top + dims.bottom,
    };
};

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
    };
};

async function decompress(base64string) {
    /** Decompresses a base64 encoded ZLIB compressed string. */
    try {
        if (isNullOrUndefined(base64string) || base64string === "") {
            throw new Error("invalid base64 string");
        }
        const ds = new DecompressionStream("deflate");
        const writer = ds.writable.getWriter();
        const bytes = Uint8Array.from(atob(base64string), c => c.charCodeAt(0));
        writer.write(bytes);
        writer.close();
        const arrayBuffer = await new Response(ds.readable).arrayBuffer();
        return await JSON.parse(new TextDecoder().decode(arrayBuffer));
    } catch (error) {
        throw new InvalidCompressedJsonDataError(error);
    }
}

const htmlStringToElement = function(str) {
    /** Converts an HTML string into an Element type. */
    let parser = new DOMParser();
    let tmp = parser.parseFromString(str, "text/html");
    return tmp.body.firstElementChild;
};

const calcColsPerRow = function(parent, child) {
    /** Calculates the number of columns of children that can fit in a parent's visible width. */
    const parent_inner_width = parent.offsetWidth - getComputedPaddingDims(parent).width;
    return parseInt(parent_inner_width / getComputedDims(child).width);

};

const calcRowsPerCol = function(parent, child) {
    /** Calculates the number of rows of children that can fit in a parent's visible height. */
    const parent_inner_height = parent.offsetHeight - getComputedPaddingDims(parent).height;
    return parseInt(parent_inner_height / getComputedDims(child).height);
};

class ExtraNetworksClusterize {
    /** Base class for a clusterize list. Cannot be used directly. */
    constructor(
        {
            tabname,
            extra_networks_tabname,
            scroll_id,
            content_id,
            data_request_callback,
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
        if (!isStringLogError(tabname)) {
            return;
        }
        if (!isStringLogError(extra_networks_tabname)) {
            return;
        }
        if (!isStringLogError(scroll_id)) {
            return;
        }
        if (!isStringLogError(content_id)) {
            return;
        }
        if (!isFunctionLogError(data_request_callback)) {
            return;
        }

        this.tabname = tabname;
        this.extra_networks_tabname = extra_networks_tabname;
        this.scroll_id = scroll_id;
        this.content_id = content_id;
        this.data_request_callback = data_request_callback;
        this.rows_in_block = rows_in_block;
        this.blocks_in_cluster = blocks_in_cluster;
        this.show_no_data_row = show_no_data_row;
        this.callbacks = callbacks;

        this.clusterize = null;

        this.scroll_elem = null;
        this.content_elem = null;

        this.element_observer = null;

        this.resize_observer = null;
        this.resize_observer_timer = null;

        // Used to control logic. Many functions immediately return when disabled.
        this.enabled = false;

        // Stores the current encoded string so we can compare against future versions.
        this.encoded_str = "";

        this.no_data_text = "No results.";
        this.no_data_class = "clusterize-no-data";

        this.n_rows = this.rows_in_block;
        this.n_cols = 1;

        this.data_obj = {};
        this.data_obj_keys_sorted = [];

        this.sort_fn = this.sortByDivId;
        this.sort_reverse = false;
    }

    reset() {
        /** Destroy clusterize instance and set all instance variables to defaults. */
        this.destroy();

        this.teardownElementObservers();
        this.teardownResizeObservers();

        this.clusterize = null;
        this.scroll_elem = null;
        this.content_elem = null;
        this.enabled = false;
        this.encoded_str = "";
        this.n_rows = this.rows_in_block;
        this.n_cols = 1;
        this.data_obj = {};
        this.data_obj_keys_sorted = [];
        this.sort_fn = this.sortByDivId;
        this.sort_reverse = false;
    }

    async setup() {
        return new Promise(resolve => {
            // Setup our event handlers only after our elements exist in DOM.
            Promise.all([
                waitForElement(`#${this.scroll_id}`).then((elem) => this.scroll_elem = elem),
                waitForElement(`#${this.content_id}`).then((elem) => this.content_elem = elem),
            ]).then(() => {
                this.setupElementObservers();
                this.setupResizeObservers();
                return this.fetchData();
            }).then(encoded_str => {
                if (isNullOrUndefined(encoded_str)) {
                    // no new data to load. break from chain.
                    return resolve();
                }
                return this.parseJson(encoded_str);
            }).then(json => {
                if (isNullOrUndefined(json)) {
                    return resolve();
                }
                this.clear();
                this.updateJson(json);
            }).then(() => {
                this.sortData();
            }).then(() => {
                // since calculateDims manually adds an element from our data_obj,
                // we don't need clusterize initialzied to calculate dims.
                this.calculateDims();
                this.applyFilter();
                this.rebuild(this.getFilteredRows());
                return resolve();
            }).catch(error => {
                console.error("setup:: error in promise:", error);
                return resolve();
            });
        });
    }

    async load() {
        return new Promise(resolve => {
            if (isNullOrUndefined(this.clusterize)) {
                // This occurs whenever we click on a tab before initialization and setup
                // have fully completed for this instance.
                return resolve(this.setup());
            }
            if (this.calculateDims()) {
                // Since dimensions updated, we need to apply the filter and rebuild.
                this.applyFilter();
                this.rebuild(this.getFilteredRows());
            } else {
                this.refresh(true);
            }
            return resolve();
        });
    }

    async fetchData() {
        let encoded_str = await this.data_request_callback(
            this.tabname,
            this.extra_networks_tabname,
            this.constructor.name,
        );
        if (this.encoded_str === encoded_str) {
            // no change to the data since last call. ignore.
            return null;
        }
        this.encoded_str = encoded_str;
        return this.encoded_str;
    }

    async parseJson(encoded_str) { /** promise */
        /** Parses a base64 encoded and gzipped JSON string and sets up a clusterize instance. */
        return new Promise((resolve, reject) => {
            Promise.resolve(encoded_str)
                .then(v => decompress(v))
                .then(v => resolve(v))
                .catch(error => {
                    return reject(error);
                });
        });
    }

    calculateDims() {
        let res = false;
        // Cannot calculate dims if not enabled since our elements won't be visible.
        if (!this.enabled) {
            return res;
        }

        // Cannot do anything if we have no data.
        if (this.data_obj_keys_sorted.length <= 0) {
            return res;
        }

        // Repair before anything else so we can actually get dimensions.
        this.repair();

        // Add an element to the container manually so we can calculate dims.
        const child = htmlStringToElement(this.data_obj[this.data_obj_keys_sorted[0]].html);
        this.content_elem.prepend(child);

        let n_cols = calcColsPerRow(this.content_elem, child);
        let n_rows = calcRowsPerCol(this.scroll_elem, child);
        n_cols = (isNaN(n_cols) || n_cols <= 0) ? 1 : n_cols;
        n_rows = (isNaN(n_rows) || n_rows <= 0) ? 1 : n_rows;
        n_rows += 2;
        if (n_cols != this.n_cols || n_rows != this.n_rows) {
            // Sizes have changed. Update the instance values.
            this.n_cols = n_cols;
            this.n_rows = n_rows;
            this.rows_in_block = this.n_rows;
            res = true;
        }

        // Remove the temporary element from DOM.
        child.remove();

        return res;
    }

    sortData() {
        /** Sorts the rows using the instance's `sort_fn`.
                 *
                 *  It is expected that a subclass will override this function to update the
                 *  instance's `sort_fn` then call `super.sortData()` to apply the sorting.
                 */
        this.sort_fn();
        if (this.sort_reverse) {
            this.data_obj_keys_sorted = this.data_obj_keys_sorted.reverse();
        }
    }

    applyFilter() {
        /** Should be overridden by child class. */
        this.sortData();
        if (!isNullOrUndefined(this.clusterize)) {
            this.update(this.getFilteredRows());
        }
        //this.rebuild(this.getFilteredRows());
    }

    getFilteredRows() {
        let rows = [];
        let active_keys = this.data_obj_keys_sorted.filter(k => this.data_obj[k].active);
        for (let i = 0; i < active_keys.length; i += this.n_cols) {
            rows.push(
                active_keys.slice(i, i + this.n_cols)
                    .map(k => this.data_obj[k].html)
                    .join("")
            );
        }
        return rows;
    }

    rebuild(rows) {
        if (!isNullOrUndefined(this.clusterize)) {
            this.clusterize.destroy(true);
            this.clusterize = null;
        }

        if (isNullOrUndefined(rows) || !Array.isArray(rows)) {
            rows = [];
        }

        this.clusterize = new Clusterize(
            {
                rows: rows,
                scrollId: this.scroll_id,
                contentId: this.content_id,
                rows_in_block: this.rows_in_block,
                tag: "div",
                blocks_in_cluster: this.blocks_in_cluster,
                show_no_data_row: this.show_no_data_row,
                no_data_text: this.no_data_text,
                no_data_class: this.no_data_class,
                callbacks: this.callbacks,
            }
        );
    }

    enable(enabled) {
        /** Enables or disables this instance. */
        // All values other than `true` for `enabled` result in this.enabled=false.
        this.enabled = !(enabled !== true);
    }

    updateJson(json) { /** promise */
        console.error("Base class method called. Must be overridden by subclass.");
        return new Promise(resolve => {
            return resolve();
        });
    }

    sortByDivId() {
        /** Sort data_obj keys (div_id) as numbers. */
        this.data_obj_keys_sorted = Object.keys(this.data_obj).sort((a, b) => INT_COLLATOR.compare(a, b));
    }

    updateDivContent(div_id, content) {
        /** Updates an element's html in the dataset.
         *
         *  NOTE: This function only updates the dataset. Calling function must call
         *  rebuild() to apply these changes to the view. Adding this call to this
         *  function would be very slow in the case where many divs need their content
         *  updated at the same time.
        */
        if (!(div_id in this.data_obj)) {
            console.error("div_id not in data_obj:", div_id);
        } else if (isElement(content)) {
            this.data_obj[div_id].html = content.outerHTML;
            return true;
        } else if (isString(content)) {
            this.data_obj[div_id].html = content;
            return true;
        } else {
            console.error("Invalid content:", div_id, content);
        }

        return false;
    }

    getMaxRowWidth() {
        console.error("getMaxRowWidth:: Not implemented in base class. Must be overridden.");
        return;
    }

    repair() {
        /** Fixes element association in DOM. Returns whether a fix was performed. */
        if (!this.enabled) {
            return false;
        }
        if (!isElement(this.scroll_elem) || !isElement(this.content_elem)) {
            return false;
        }
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

    onResize(elem_id) {
        /** Callback whenever one of our visible elements is resized. */
        if (!this.enabled) {
            return;
        }
        this.refresh(true);
        if (this.calculateDims()) {
            this.rebuild(this.getFilteredRows());
        }
    }

    onElementDetached(elem_id) {
        /** Callback whenever one of our elements has become detached from the DOM. */
        switch (elem_id) {
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

    setupElementObservers() {
        /** Listens for changes to the data, scroll, and content elements.
         *
         *  During testing, the scroll/content elements would frequently get removed from
         *  the DOM. Our clusterize instance stores a reference to these elements
         *  which breaks whenever these elements are removed from the DOM. To fix this,
         *  we need to check for these changes and re-attach our stores elements by
         *  replacing the ones in the DOM with the ones in our clusterize instance.
         *
         *  We also use an observer to detect whenever the data element gets a new set
         *  of JSON data so that we can update our dataset.
         */
        this.element_observer = new MutationObserver((mutations) => {
            // don't waste time if this object isn't enabled.
            if (!this.enabled) {
                return;
            }

            let scroll_elem = gradioApp().getElementById(this.scroll_id);
            if (scroll_elem && scroll_elem !== this.scroll_elem) {
                this.onElementDetached(scroll_elem.id);
            }

            let content_elem = gradioApp().getElementById(this.content_id);
            if (content_elem && content_elem !== this.content_elem) {
                this.onElementDetached(content_elem.id);
            }
        });
        this.element_observer.observe(gradioApp(), {subtree: true, childList: true, attributes: true});
    }

    teardownElementObservers() {
        if (!isNullOrUndefined(this.element_observer)) {
            this.element_observer.takeRecords();
            this.element_observer.disconnect();
        }
        this.element_observer = null;
    }

    setupResizeObservers() {
        /** Handles any updates to the size of both the Scroll and Content elements. */
        this.resize_observer = new ResizeObserver((entries) => {
            for (const entry of entries) {
                if (entry.target.id === this.scroll_id || entry.target.id === this.content_id) {
                    // debounce the event
                    clearTimeout(this.resize_observer_timer);
                    this.resize_observer_timer = setTimeout(() => this.onResize(entry.id), RESIZE_DEBOUNCE_TIME_MS);
                }
            }
        });

        this.resize_observer.observe(this.scroll_elem);
        this.resize_observer.observe(this.content_elem);
    }

    teardownResizeObservers() {
        if (!isNullOrUndefined(this.resize_observer)) {
            this.resize_observer.disconnect();
        }
        if (!isNullOrUndefined(this.resize_observer_timer)) {
            clearTimeout(this.resize_observer_timer);
            this.resize_observer_timer = null;
        }
        this.resize_observer = null;
        this.resize_observer_timer = null;
    }

    /* ==== Clusterize.Js FUNCTION WRAPPERS ==== */
    update(rows) {
        /** Updates the clusterize rows. */
        if (isNullOrUndefined(rows) || !Array.isArray(rows)) {
            rows = this.getFilteredRows();
        }
        this.clusterize.update(rows);
    }

    clear() {
        /** Clears the clusterize list and this instance's data. */
        if (!isNullOrUndefined(this.clusterize)) {
            this.clusterize.clear();
            this.data_obj = {};
            this.data_obj_keys_sorted = [];
            if (isElement(this.content_elem)) {
                this.content_elem.innerHTML = "<div class='clusterize-no-data'>Loading...</div>";
            }
        }
    }

    refresh(force) {
        /** Refreshes the clusterize instance so that it can recalculate its dims.
         * `force` [boolean]: If true, tells clusterize to refresh regardless of whether
         *      its dimensions have changed.
        */
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

    destroy() {
        /** Destroys a clusterize instance and removes its rows from the page. */
        // Passing `true` prevents clusterize from dumping every row in its dataset
        // to the DOM. This kills performance so we never want to do this.
        if (!isNullOrUndefined(this.clusterize)) {
            this.clusterize.destroy(true);
            this.clusterize = null;
        }
        this.data_obj = {};
        this.data_obj_keys_sorted = [];
        if (isElement(this.content_elem)) {
            this.content_elem.innerHTML = "<div class='clusterize-no-data'>Loading...</div>";
        }
    }
}

class ExtraNetworksClusterizeTreeList extends ExtraNetworksClusterize {
    /** Subclass used to display a directories/files in the Tree View. */
    constructor(...args) {
        super(...args);

        this.no_data_text = "No directories/files";
        this.selected_div_id = null;
    }

    reset() {
        this.selected_div_id = null;
        super.reset();
    }

    getBoxShadow(depth) {
        /** Generates style for a multi-level box shadow for vertical indentation lines.
         * This is used to indicate the depth of a directory/file within a directory tree.
        */
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
        /** Processes JSON object and adds each entry to our data object. */
        return new Promise(resolve => {
            var style = getComputedStyle(document.body);
            let text_size = style.getPropertyValue("--button-large-text-size");
            for (const [k, v] of Object.entries(json)) {
                let div_id = k;
                let parsed_html = htmlStringToElement(v);
                // parent_id = -1 if item is at root level
                let parent_id = "parentId" in parsed_html.dataset ? parsed_html.dataset.parentId : -1;
                let expanded = "expanded" in parsed_html.dataset;
                let selected = "selected" in parsed_html.dataset;
                let depth = Number(parsed_html.dataset.depth);
                parsed_html.style.paddingLeft = `calc(${depth} * ${text_size})`;
                parsed_html.style.boxShadow = this.getBoxShadow(depth);

                // Add the updated html to the data object.
                this.data_obj[div_id] = {
                    html: parsed_html.outerHTML,
                    active: parent_id === -1, // always show root
                    expanded: expanded || (parent_id === -1), // always expand root
                    selected: selected,
                    parent: parent_id,
                    children: [], // populated later
                };

                // maybe not necessary.
                parsed_html = null;
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
        /** Removes rows from the list that are children of the passed div.
         * The rows aren't removed from the data object, just set to active=false
         * so they aren't displayed.
        */
        for (const child_id of this.data_obj[div_id].children) {
            this.data_obj[child_id].active = false;
            if (this.data_obj[child_id].selected) {
                // deselect the child only if it is selected.
                let elem = htmlStringToElement(this.data_obj[child_id].html);
                delete elem.dataset.selected;
                this.data_obj[child_id].selected = false;
                this.updateDivContent(child_id, elem);
            }
            this.removeChildRows(child_id);
        }
    }

    addChildRows(div_id) {
        /** Adds rows to the list that are children of the passed div.
         * The rows aren't added to the data object, just set to active=true
         * so they are displayed.
        */
        for (const child_id of this.data_obj[div_id].children) {
            this.data_obj[child_id].active = true;
            if (this.data_obj[child_id].expanded) {
                this.addChildRows(child_id);
            }
        }
    }

    onRowExpandClick(div_id, elem) {
        /** Toggles expand/collapse of a row's children. */
        if ("expanded" in elem.dataset) {
            this.data_obj[div_id].expanded = false;
            delete elem.dataset.expanded;
            this.removeChildRows(div_id);
        } else {
            this.data_obj[div_id].expanded = true;
            elem.dataset.expanded = "";
            this.addChildRows(div_id);
        }
        this.updateDivContent(div_id, elem);
        if (this.calculateDims()) {
            this.rebuild(this.getFilteredRows());
        } else {
            this.update(this.getFilteredRows());
        }
    }

    _setRowSelectedState(div_id, elem, new_state) {
        if (new_state) {
            elem.dataset.selected = "";
        } else {
            delete elem.dataset.selected;
        }
        this.data_obj[div_id].selected = new_state;
        this.updateDivContent(div_id, elem);
    }

    onRowSelected(div_id, elem, override) {
        /** Selects a row and deselects all others. */
        if (!isElementLogError(elem)) {
            return;
        }
        if (!(div_id in this.data_obj)) {
            console.error("div_id not in dataset:", div_id);
            return;
        }

        if (!isNullOrUndefined(override)) {
            override = (override === true);
        }

        if (!isNullOrUndefined(this.selected_div_id) && div_id !== this.selected_div_id) {
            // deselect the current selected row
            let prev_elem = htmlStringToElement(this.data_obj[this.selected_div_id].html);
            this._setRowSelectedState(this.selected_div_id, prev_elem, false);

            // select the new row
            this._setRowSelectedState(div_id, elem, true);
            this.selected_div_id = div_id;
        } else {
            // toggle the passed row's selected state.
            if (this.data_obj[div_id].selected) {
                this._setRowSelectedState(div_id, elem, false);
                this.selected_div_id = null;
            } else {
                this._setRowSelectedState(div_id, elem, true);
                this.selected_div_id = div_id;
            }
        }

        this.update(this.getFilteredRows());
    }

    getMaxRowWidth() {
        /** Calculates the width of the widest row in the list. */
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
                row_width += child.scrollWidth;
                // Restore previous style.
                child.style.cssText = prev_style;
            }
            max_width = Math.max(max_width, row_width);
        }
        if (max_width <= 0) {
            return;
        }

        // Adds the scroll element's border and the scrollbar's width to the result.
        // If scrollbar isn't visible, then only the element border is added.
        max_width += this.scroll_elem.offsetWidth - this.scroll_elem.clientWidth;
        return max_width;
    }
}

class ExtraNetworksClusterizeCardsList extends ExtraNetworksClusterize {
    /** Subclass used to display cards in the Cards View. */
    constructor(...args) {
        super(...args);

        this.no_data_text = "No files matching filter.";
        this.default_sort_mode_str = "path";
        this.default_sort_dir_str = "ascending";
        this.default_filter_str = "";

        this.sort_mode_str = this.default_sort_mode_str;
        this.sort_dir_str = this.default_sort_dir_str;
        this.filter_str = this.default_filter_str;
    }

    reset() {
        this.sort_mode_str = this.default_sort_mode_str;
        this.sort_dir_str = this.default_sort_dir_str;
        this.filter_str = this.default_filter_str;
        super.reset();
    }

    updateJson(json) {
        /** Processes JSON object and adds each entry to our data object. */
        return new Promise(resolve => {
            this.data_obj = {};
            for (const [i, [k, v]] of Object.entries(Object.entries(json))) {
                let div_id = k;
                let parsed_html = htmlStringToElement(v);
                let search_only = isElement(parsed_html.querySelector(".search_only"));
                let search_terms_elem = parsed_html.querySelector(".search_terms");
                let search_terms = "";
                if (isElement(search_terms_elem)) {
                    search_terms = Array.prototype.map.call(
                        parsed_html.querySelectorAll(".search_terms"),
                        (elem) => {
                            return elem.textContent.toLowerCase();
                        }
                    ).join(" ");
                }

                // Add the updated html to the data object.
                this.data_obj[div_id] = {
                    active: true,
                    html: v,
                    sort_name: parsed_html.dataset.sortName,
                    sort_path: parsed_html.dataset.sortPath,
                    sort_created: parsed_html.dataset.sortCreated,
                    sort_modified: parsed_html.dataset.sortModified,
                    search_only: search_only,
                    search_terms: search_terms,
                };

                // maybe not necessary
                parsed_html = null;
            }
            return resolve();
        });
    }

    sortByName() {
        this.data_obj_keys_sorted = Object.keys(this.data_obj).sort((a, b) => {
            return STR_COLLATOR.compare(
                this.data_obj[a].sort_name,
                this.data_obj[b].sort_name,
            );
        });
    }

    sortByPath() {
        this.data_obj_keys_sorted = Object.keys(this.data_obj).sort((a, b) => {
            return STR_COLLATOR.compare(
                this.data_obj[a].sort_path,
                this.data_obj[b].sort_path,
            );
        });
    }

    sortByCreated() {
        this.data_obj_keys_sorted = Object.keys(this.data_obj).sort((a, b) => {
            return INT_COLLATOR.compare(
                this.data_obj[a].sort_created,
                this.data_obj[b].sort_created,
            );
        });
    }

    sortByModified() {
        this.data_obj_keys_sorted = Object.keys(this.data_obj).sort((a, b) => {
            return INT_COLLATOR.compare(
                this.data_obj[a].sort_modified,
                this.data_obj[b].sort_modified,
            );
        });
    }

    setSortMode(sort_mode_str) {
        this.sort_mode_str = sort_mode_str;
    }

    setSortDir(sort_dir_str) {
        this.sort_dir_str = sort_dir_str;
    }

    sortData() {
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
        super.sortData();
    }

    applyFilter(filter_str) {
        /** Filters our data object by setting each member's `active` attribute then sorts the result. */
        if (!isNullOrUndefined(filter_str)) {
            this.filter_str = filter_str.toLowerCase();
        } else if (isNullOrUndefined(this.filter_str)) {
            this.filter_str = this.default_filter_str;
        }

        for (const [k, v] of Object.entries(this.data_obj)) {
            let visible = v.search_terms.indexOf(this.filter_str) != -1;
            if (v.search_only && this.filter_str.length < 4) {
                visible = false;
            }
            this.data_obj[k].active = visible;
        }

        super.applyFilter()
    }

    getMaxRowWidth() {
        /** Calculates the width of the widest pseudo-row in the list. */
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
