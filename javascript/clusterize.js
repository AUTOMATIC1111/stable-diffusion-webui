/* eslint-disable */
/*
    Heavily modified Clusterize.js v1.0.0.
    Original: http://NeXTs.github.com/Clusterize.js/

    This has been modified to allow for an asynchronous data loader implementation.
    This differs from the original Clusterize.js which would store the entire dataset
    in an array and load from that; this caused a large memory overhead in the client.
*/

// Many operations can be lenghty. Try to limit their frequency by debouncing.
const SCROLL_DEBOUNCE_TIME_MS = 50;
const RESIZE_OBSERVER_DEBOUNCE_TIME_MS = 100; // should be <= refresh debounce time
const ELEMENT_OBSERVER_DEBOUNCE_TIME_MS = 100;
const REFRESH_DEBOUNCE_TIME_MS = 100;

class Clusterize {
    scroll_elem = null;
    content_elem = null;
    scroll_id = null;
    content_id = null;
    config = {
        block_height: null,
        block_width: null,
        cols_in_block: 1,
        cluster_height: null,
        cluster_width: null,
        is_mac: navigator.userAgent.toLowerCase().indexOf("mac") > -1,
        item_height: null,
        item_width: null,
        max_items: 0,
        max_rows: 0,
        rows_in_block: 50,
        rows_in_cluster: 50 * 5, // default is rows_in_block * blocks_in_cluster
    };
    options = {
        blocks_in_cluster: 5,
        tag: "div",
        no_data_class: "clusterize-no-data",
        no_data_html: "No Data",
        error_class: "clusterize-error",
        error_html: "Data Error",
        keep_parity: true,
        callbacks: {},
    };
    state = {
        cache: {},
        curr_cluster: 0,
        enabled: false,
        scroll_top: 0,
        setup_has_run: false,
        pointer_events_set: false,
    };
    #scroll_debounce_timer = 0;
    #refresh_debounce_timer = null;
    #resize_observer = null;
    #resize_observer_timer = null;
    #element_observer = null;
    #element_observer_timer = null;
    #on_scroll_bound;

    constructor(args) {
        for (const option of Object.keys(this.options)) {
            if (keyExists(args, option)) {
                this.options[option] = args[option];
            }
        }

        if (isNullOrUndefined(this.options.callbacks.initData)) {
            this.options.callbacks.initData = this.initDataDefaultCallback.bind(this);
        }
        if (isNullOrUndefined(this.options.callbacks.fetchData)) {
            this.options.callbacks.fetchData = this.fetchDataDefaultCallback.bind(this);
        }
        if (isNullOrUndefined(this.options.callbacks.sortData)) {
            this.options.callbacks.sortData = this.sortDataDefaultCallback.bind(this);
        }
        if (isNullOrUndefined(this.options.callbacks.filterData)) {
            this.options.callbacks.filterData = this.filterDataDefaultCallback.bind(this);
        }

        this.scroll_elem = args["scrollId"] ? document.getElementById(args["scrollId"]) : args["scrollElem"];
        isElementThrowError(this.scroll_elem);
        this.scroll_id = this.scroll_elem.id;

        this.content_elem = args["contentId"] ? document.getElementById(args["contentId"]) : args["contentElem"];
        isElementThrowError(this.content_elem);
        this.content_id = this.content_elem.id;

        if (!this.content_elem.hasAttribute("tabindex")) {
            this.content_elem.setAttribute("tabindex", 0);
        }

        this.state.scroll_top = this.scroll_elem.scrollTop;

        this.config.max_items = args.max_items;

        this.#on_scroll_bound = this.#onScroll.bind(this);
    }

    // ==== PUBLIC FUNCTIONS ====
    enable(state) {
        // if no state is passed, we enable by default.
        this.state.enabled = state !== false;
    }

    async setup() {
        if (this.state.setup_has_run || !this.state.enabled) {
            return;
        }

        this.#fixElementReferences();

        // We use margins to control the scrollbar's size and float our content.
        this.content_elem.style.marginTop = "0px";
        this.content_elem.style.marginBottom = "0px";

        await this.#insertToDOM();

        this.#setupEvent("scroll", this.scroll_elem, this.#on_scroll_bound);
        this.#setupElementObservers();
        this.#setupResizeObservers();

        this.state.setup_has_run = true;
    }

    clear() {
        if (!this.state.setup_has_run || !this.state.enabled) {
            return;
        }

        this.#html(this.#generateEmptyRow().join(""));
    }

    destroy() {        
        this.#teardownEvent("scroll", this.scroll_elem, this.#on_scroll_bound);
        this.#teardownElementObservers();
        this.#teardownResizeObservers();

        this.#html(this.#generateEmptyRow().join(""));
        
        this.state.setup_has_run = false;
    }

    async refresh(force) {
        if (!this.state.setup_has_run || !this.state.enabled) {
            return;
        }

        // Refresh can be a longer operation so we want to debounce it to
        // avoid refreshing too often.
        clearTimeout(this.#refresh_debounce_timer);
        this.#refresh_debounce_timer = setTimeout(
            async () => {
                if (!isElement(this.content_elem.offsetParent)) {
                    return;
                }
        
                if (this.#recalculateDims() || force) {
                    await this.update()
                }
            },
            REFRESH_DEBOUNCE_TIME_MS,
        )
    }

    async update() {
        if (!this.state.setup_has_run || !this.state.enabled) {
            return;
        }

        this.state.scroll_top = this.scroll_elem.scrollTop;
        // fixes #39
        if (this.config.max_rows * this.config.item_height < this.state.scroll_top) {
            this.scroll_elem.scrollTop = 0;
            this.state.curr_cluster = 0;
        }

        await this.#insertToDOM();
    }

    getRowsAmount() {
        return this.config.max_rows;
    }

    getScrollProgress() {
        return this.state.scroll_top / (this.config.max_rows * this.config.item_height) * 100 || 0;
    }

    async setMaxItems(max_items) {
        /** Sets the new max number of items.
         *
         *  This is used to control the scroll bar's length.
         *
         *  Returns whether the number of max items changed.
         */
        if (!this.state.setup_has_run || !this.state.enabled) {
            this.config.max_items = max_items;
            return this.config.max_items !== max_items;
        }

        this.config.max_items = max_items;
    }

    // ==== PRIVATE FUNCTIONS ====
    initDataDefaultCallback() {
        return Promise.resolve({});
    }

    async initData() {
        if (!this.state.enabled) {
            return;
        }
        return await this.options.callbacks.initData();
    }

    fetchDataDefaultCallback() {
        return Promise.resolve([]);
    }

    async fetchData(idx_start, idx_end) {
        if (!this.state.enabled) {
            return;
        }
        try {
            return await this.options.callbacks.fetchData(idx_start, idx_end);
        } catch (error) {
            throw error;
        }
    }

    sortDataDefaultCallback() {
        return Promise.resolve();
    }

    async sortData() {
        if (!this.state.setup_has_run || !this.state.enabled) {
            return;
        }

        this.#fixElementReferences();

        // Sort is applied to the filtered data.
        await this.options.callbacks.sortData();
        this.#recalculateDims();
        await this.#insertToDOM();
    }

    filterDataDefaultCallback() {
        return Promise.resolve(0);
    }

    async filterData() {
        if (!this.state.setup_has_run || !this.state.enabled) {
            return;
        }

        // Filter is applied to entire dataset.
        const max_items = await this.options.callbacks.filterData();
        await this.setMaxItems(max_items);
        await this.refresh(true);
        await this.sortData();
    }

    #exploreEnvironment(rows, cache) {
        this.options.content_tag = this.content_elem.tagName.toLowerCase();
        if (isNullOrUndefined(rows) || !rows.length) {
            return;
        }
        // Temporarily add one row so that we can calculate row dimensions.
        if (this.content_elem.children.length <= 1) {
            cache.data = this.#html(rows[0]);
        }
        if (!this.options.tag) {
            this.options.tag = this.content_elem.children[0].tagName.toLowerCase();
        }
        this.#recalculateDims();
    }

    #recalculateDims() {
        const prev_config = JSON.stringify(this.config);

        this.config.block_height = 0;
        this.config.block_width = 0;
        this.config.cluster_height = 0;
        this.config.cluster_width = 0;
        this.config.item_height = 0;
        this.config.item_width = 0;

        if (!this.config.max_items) {
            return;
        }

        // Get the first element that isn't one of our placeholder rows.
        const node = this.content_elem.querySelector(
            `:scope > :not(.clusterize-extra-row,.${this.options.no_data_class})`
        );
        if (!isElement(node)) {
            // dont attempt to compute dims if we have no data.
            return;
        }

        const node_dims = getComputedDims(node);
        this.config.item_height = node_dims.height;
        this.config.item_width = node_dims.width;
        
        // consider table's browser spacing
        if (this.options.tag === "tr" && getComputedProperty(this.content_elem, "borderCollapse") !== "collapse") {
            const spacing = parseInt(getComputedProperty(this.content_elem, "borderSpacing"), 10) || 0;
            this.config.item_height += spacing;
            this.config.item_width += spacing;
        }

        // Update rows in block to match the number of elements that can fit in the view.
        const content_padding = getComputedPaddingDims(this.content_elem);
        const column_gap = parseFloat(getComputedProperty(this.content_elem, "column-gap"));
        const row_gap = parseFloat(getComputedProperty(this.content_elem, "row-gap"));
        if (isNumber(column_gap)) {
            this.config.item_width += column_gap;
        }
        if (isNumber(row_gap)) {
            this.config.item_height += row_gap;
        }

        const inner_width = this.scroll_elem.clientWidth - content_padding.width;
        const inner_height = this.scroll_elem.clientHeight - content_padding.height;
        // Since we don't allow horizontal scrolling, we want to round down for columns.
        const cols_in_block = Math.floor(inner_width / this.config.item_width);
        // Round up for rows so that we don't cut rows off from the view.
        const rows_in_block = Math.ceil(inner_height / this.config.item_height);

        // Always need at least 1 row/col in block
        this.config.cols_in_block = Math.max(1, cols_in_block);
        this.config.rows_in_block = Math.max(1, rows_in_block);

        this.config.block_height = this.config.item_height * this.config.rows_in_block;
        this.config.block_width = this.config.item_width * this.config.cols_in_block;
        this.config.rows_in_cluster = this.options.blocks_in_cluster * this.config.rows_in_block;
        this.config.cluster_height = this.options.blocks_in_cluster * this.config.block_height;
        this.config.cluster_width = this.config.block_width;

        this.config.max_rows = Math.ceil(this.config.max_items / this.config.cols_in_block);
        this.config.max_clusters = Math.ceil(this.config.max_rows / this.config.rows_in_cluster);

        return prev_config !== JSON.stringify(this.config);
    }

    #generateEmptyRow({is_error}={}) {
        const row = document.createElement(is_error ? "div" : this.options.tag);
        row.className = is_error ? this.options.error_class : this.options.no_data_class;
        if (this.options.tag === "tr") {
            const td = document.createElement("td");
            td.colSpan = 100;
            td.innerHTML = is_error ? this.options.error_html : this.options.no_data_html;
            row.appendChild(td);
        } else {
            row.innerHTML = is_error ? this.options.error_html : this.options.no_data_html;
        }
        return [row.outerHTML];
    }

    #getClusterNum() {
        this.state.scroll_top = this.scroll_elem.scrollTop;
        const cluster_divider = this.config.cluster_height - this.config.block_height;
        const current_cluster = Math.floor(this.state.scroll_top / cluster_divider);
        return Math.min(current_cluster, this.config.max_clusters);
    }

    async #generate() {
        const rows_start = Math.max(0, (this.config.rows_in_cluster - this.config.rows_in_block) * this.#getClusterNum());
        const rows_end = rows_start + this.config.rows_in_cluster;
        const top_offset = Math.max(0, rows_start * this.config.item_height);
        const bottom_offset = Math.max(0, (this.config.max_rows - rows_end) * this.config.item_height);
        const rows_above = top_offset < 1 ? rows_start + 1 : rows_start;

        const idx_start = Math.max(0, rows_start * this.config.cols_in_block);
        const idx_end = Math.min(this.config.max_items, rows_end * this.config.cols_in_block);

        let this_cluster_rows = await this.fetchData(idx_start, idx_end);
        if (!Array.isArray(this_cluster_rows) || !this_cluster_rows.length) {
            console.error(`Failed to fetch data for idx range (${idx_start},${idx_end})`);
            this_cluster_rows = [];
        }

        if (this_cluster_rows.length < this.config.rows_in_block) {
            return {
                top_offset: 0,
                bottom_offset: 0,
                rows_above: 0,
                rows: this_cluster_rows.length ? this_cluster_rows : this.#generateEmptyRow({is_error: true}),
            };
        }

        return {
            top_offset: top_offset,
            bottom_offset: bottom_offset,
            rows_above: rows_above,
            rows: this_cluster_rows,
        };
    }

    async #insertToDOM() {
        if (!this.config.cluster_height || !this.config.cluster_width) {
            // We need to fetch a single item so that we can calculate the dimensions
            // for our list.
            const rows = await this.fetchData(0, 1);
            if (!Array.isArray(rows) || !rows.length) {
                // This implies there is no data for this list. Not an error.
                // Errors should be handled in the fetchData callback, not here.
                this.#html(this.#generateEmptyRow().join(""));
                return;
            } else {
                this.#html(rows.join(""));
                this.#exploreEnvironment(rows, this.state.cache);
                // Remove the temporary item from the data since we calculated its size.
                this.#html(this.#generateEmptyRow().join(""));
            }
        }

        const data = await this.#generate();
        let this_cluster_rows = [];
        for (let i = 0; i < data.rows.length; i += this.config.cols_in_block) {
            const new_row = data.rows.slice(i, i + this.config.cols_in_block).join("");
            this_cluster_rows.push(new_row);
        }
        this_cluster_rows = this_cluster_rows.join("");
        const this_cluster_content_changed = this.#checkChanges("data", this_cluster_rows, this.state.cache);
        const top_offset_changed = this.#checkChanges("top", data.top_offset, this.state.cache);
        const only_bottom_offset_changed = this.#checkChanges("bottom", data.bottom_offset, this.state.cache);
        const layout = [];

        if (this_cluster_content_changed || top_offset_changed) {
            if (this.options.callbacks.clusterWillChange) {
                this.options.callbacks.clusterWillChange();
            }

            if (data.top_offset && this.options.keep_parity) {
                layout.push(this.#renderExtraTag("keep-parity"));
            }
            layout.push(this_cluster_rows)
            this.#html(layout.join(""));
            if (this.options.content_tag === "ol") {
                this.content_elem.setAttribute("start", data.rows_above);
            }
            this.content_elem.style["counter-increment"] = `clusterize-counter ${data.rows_above - 1}`;
            if (this.options.callbacks.clusterChanged) {
                this.options.callbacks.clusterChanged();
            }
        }
        // Update the margins to fix scrollbar position.
        this.content_elem.style.marginTop = `${data.top_offset}px`;
        this.content_elem.style.marginBottom = `${data.bottom_offset}px`;
    }

    #html(data) {
        this.content_elem.innerHTML = data;

        // Parse items flagged as containing Shadow DOM entries.
        convertElementShadowDOM(this.content_elem, "[data-parse-as-shadow-dom]");

        return this.content_elem.innerHTML;
    }

    #renderExtraTag(class_name, height) {
        const tag = document.createElement(this.options.tag);
        const clusterize_prefix = "clusterize-";
        tag.className = [
            `${clusterize_prefix}extra-row`,
            `${clusterize_prefix}${class_name}`,
        ].join(" ");
        if (isNumber(height)) {
            tag.style.height = `${height}px`;
        }
        return tag.outerHTML;
    }

    #getChildNodes(tag) {
        const child_nodes = tag.children;
        const nodes = [];
        for (let i = 0, j = child_nodes.length; i < j; i++) {
            nodes.push(child_nodes[i]);
        }
        return nodes;
    }

    #checkChanges(type, value, cache) {
        const changed = value !== cache[type];
        cache[type] = value;
        return changed;
    }

    // ==== EVENT HANDLERS ====

    async #onScroll() {
        if (this.config.is_mac) {
            if (!this.state.pointer_events_set) {
                this.content_elem.style.pointerEvents = "none";
                this.state.pointer_events_set = true;
                clearTimeout(this.#scroll_debounce_timer);
                this.#scroll_debounce_timer = setTimeout(() => {
                    this.content_elem.style.pointerEvents = "auto";
                    this.state.pointer_events_set = false;
                }, SCROLL_DEBOUNCE_TIME_MS);
            }
        }
        if (this.state.curr_cluster !== (this.state.curr_cluster = this.#getClusterNum())) {
            await this.#insertToDOM();
        }
        if (this.options.callbacks.scrollingProgress) {
            this.options.callbacks.scrollingProgress(this.getScrollProgress());
        }
    }

    async #onResize() {
        await this.refresh();
    }

    #fixElementReferences() {
        if (!isElement(this.scroll_elem) || !isElement(this.content_elem)) {
            return;
        }

        // Element is already in DOM. Don't need to do anything.
        if (isElement(this.content_elem.offsetParent)) {
            return;
        }

        // If association for elements is broken, replace them with instance version.
        document.getElementById(this.scroll_id).replaceWith(this.scroll_elem);
    }

    #setupElementObservers() {
        /** Listens for changes to the scroll and content elements.
         *
         *  During testing, the scroll/content elements would frequently get removed from
         *  the DOM. This instance stores a reference to these elements
         *  which breaks whenever these elements are removed from the DOM. To fix this,
         *  we need to check for these changes and re-attach our stores elements by
         *  replacing the ones in the DOM with the ones in our clusterize instance.
         */

        this.#element_observer = new MutationObserver((mutations) => {
            const scroll_elem = document.getElementById(this.scroll_id);
            if (isElement(scroll_elem) && scroll_elem !== this.scroll_elem) {
                clearTimeout(this.#element_observer_timer);
                this.#element_observer_timer = setTimeout(
                    this.#fixElementReferences,
                    ELEMENT_OBSERVER_DEBOUNCE_TIME_MS,
                );
            }

            const content_elem = document.getElementById(this.content_id);
            if (isElement(content_elem) && content_elem !== this.content_elem) {
                clearTimeout(this.#element_observer_timer);
                this.#element_observer_timer = setTimeout(
                    this.#fixElementReferences,
                    ELEMENT_OBSERVER_DEBOUNCE_TIME_MS,
                );
            }
        });
        const options = { subtree: true, childList: true, attributes: true };
        this.#element_observer.observe(document, options);
    }

    #teardownElementObservers() {
        if (!isNullOrUndefined(this.#element_observer)) {
            this.#element_observer.takeRecords();
            this.#element_observer.disconnect();
        }
        this.#element_observer = null;
    }

    #setupResizeObservers() {
        /** Handles any updates to the size of both the Scroll and Content elements. */
        this.#resize_observer = new ResizeObserver((entries) => {
            for (const entry of entries) {
                if (entry.target.id === this.scroll_id || entry.target.id === this.content_id) {
                    // debounce the event
                    clearTimeout(this.#resize_observer_timer);
                    this.#resize_observer_timer = setTimeout(
                        () => this.#onResize(),
                        RESIZE_OBSERVER_DEBOUNCE_TIME_MS,
                    );
                }
            }
        });

        this.#resize_observer.observe(this.scroll_elem);
        this.#resize_observer.observe(this.content_elem);
    }

    #teardownResizeObservers() {
        if (!isNullOrUndefined(this.#resize_observer)) {
            this.#resize_observer.disconnect();
        }

        if (!isNullOrUndefined(this.#resize_observer_timer)) {
            clearTimeout(this.#resize_observer_timer);
        }

        this.#resize_observer = null;
        this.#resize_observer_timer = null;
    }

    // ==== HELPER FUNCTIONS ====

    #setupEvent(type, elem, listener) {
        if (elem.addEventListener) {
            return elem.addEventListener(type, listener, false);
        } else {
            return elem.attachEvent(`on${type}`, listener);
        }
    }

    #teardownEvent(type, elem, listener) {
        if (elem.removeEventListener) {
            return elem.removeEventListener(type, listener, false);
        } else {
            return elem.detachEvent(`on${type}`, listener);
        }
    }
}
