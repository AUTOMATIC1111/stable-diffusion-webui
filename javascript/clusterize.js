/* eslint-disable */
/*
    Heavily modified Clusterize.js v1.0.0.
    Original: http://NeXTs.github.com/Clusterize.js/

    This has been modified to allow for an asynchronous data loader implementation.
    This differs from the original Clusterize.js which would store the entire dataset
    in an array and load from that; this caused a large memory overhead in the client.
*/

const SCROLL_DEBOUNCE_TIME_MS = 50;
const RESIZE_OBSERVER_DEBOUNCE_TIME_MS = 100;
const ELEMENT_OBSERVER_DEBOUNCE_TIME_MS = 100;

class Clusterize {
    scroll_elem = null;
    content_elem = null;
    scroll_id = null;
    content_id = null;
    options = {
        rows_in_block: 50,
        cols_in_block: 1,
        blocks_in_cluster: 5,
        tag: null,
        show_no_data_row: true,
        no_data_class: "clusterize-no-data",
        no_data_text: "No data",
        keep_parity: true,
        callbacks: {},
    };
    setup_has_run = false;
    #is_mac = null;
    #ie = null;
    #max_items = null;
    #max_rows = null;
    #cache = {};
    #scroll_top = 0;
    #last_cluster = false;
    #scroll_debounce = 0;
    #resize_observer = null;
    #resize_observer_timer = null;
    #element_observer = null;
    #element_observer_timer = null;
    #pointer_events_set = false;

    constructor(args) {
        for (const option of Object.keys(this.options)) {
            if (keyExists(args, option)) {
                this.options[option] = args[option];
            }
        }

        if (isNullOrUndefined(this.options.callbacks.initData)) {
            this.options.callbacks.initData = this.initDataDefaultCallback;
        }
        if (isNullOrUndefined(this.options.callbacks.fetchData)) {
            this.options.callbacks.fetchData = this.fetchDataDefaultCallback;
        }
        if (isNullOrUndefined(this.options.callbacks.sortData)) {
            this.options.callbacks.sortData = this.sortDataDefaultCallback;
        }
        if (isNullOrUndefined(this.options.callbacks.filterData)) {
            this.options.callbacks.filterData = this.filterDataDefaultCallback;
        }

        // detect ie9 and lower
        // https://gist.github.com/padolsey/527683#comment-786682
        this.#ie = (function () {
            for (var v = 3,
                el = document.createElement("b"),
                all = el.all || [];
                el.innerHTML = `<!--[if gt IE ${++v}]><i><![endif]-->`,
                all[0];
            ) { }
            return v > 4 ? v : document.documentMode;
        }())
        this.#is_mac = navigator.platform.toLowerCase().indexOf("mac") + 1;

        this.scroll_elem = args["scrollId"] ? document.getElementById(args["scrollId"]) : args["scrollElem"];
        isElementThrowError(this.scroll_elem);
        this.scroll_id = this.scroll_elem.id;

        this.content_elem = args["contentId"] ? document.getElementById(args["contentId"]) : args["contentElem"];
        isElementThrowError(this.content_elem);
        this.content_id = this.content_elem.id;

        if (!this.content_elem.hasAttribute("tabindex")) {
            this.content_elem.setAttribute("tabindex", 0);
        }

        this.#scroll_top = this.scroll_elem.scrollTop;

        this.#max_items = args.max_items;
    }

    // ==== PUBLIC FUNCTIONS ====
    async setup() {
        if (this.setup_has_run) {
            return;
        }

        await this.#insertToDOM();
        this.scroll_elem.scrollTop = this.#scroll_top;

        this.#setupEvent("scroll", this.scroll_elem, this.#onScroll);
        this.#setupElementObservers();
        this.#setupResizeObservers();

        this.setup_has_run = true;
    }

    clear() {
        if (!this.setup_has_run) {
            return;
        }

        this.#html(this.#generateEmptyRow().join(""));
    }

    destroy() {
        this.#teardownEvent("scroll", this.scroll_elem, this.#onScroll);
        this.#teardownElementObservers();
        this.#teardownResizeObservers();
        this.clear();

        this.setup_has_run = false;
    }

    async refresh(force) {
        if (!this.setup_has_run) {
            return;
        }

        if (this.#getRowsHeight() || force) {
            await this.update()
        }
    }

    async update() {
        if (!this.setup_has_run) {
            return;
        }

        this.#scroll_top = this.scroll_elem.scrollTop;
        // fixes #39
        if (this.#max_rows * this.options.item_height < this.#scroll_top) {
            this.scroll_elem.scrollTop = 0;
            this.#last_cluster = 0;
        }

        await this.#insertToDOM();
        this.scroll_elem.scrollTop = this.#scroll_top;
    }

    getRowsAmount() {
        return this.#max_rows;
    }

    getScrollProgress() {
        return this.options.scroll_top / (this.#max_rows * this.options.item_height) * 100 || 0;
    }

    async setMaxItems(max_items) {
        if (!this.setup_has_run) {
            this.#max_items = max_items;
            return;
        }

        if (max_items === this.#max_items) {
            // No change. do nothing.
            return;
        }

        // If the number of items changed, we need to update the cluster.
        this.#max_items = max_items;
        await this.refresh(true);

        // Apply sort to the updated data.
        await this.sortData();
    }

    // ==== PRIVATE FUNCTIONS ====
    initDataDefaultCallback() {
        return Promise.resolve({});
    }

    async initData() {
        return await this.options.callbacks.initData.call(this);
    }

    fetchDataDefaultCallback() {
        return Promise.resolve([]);
    }

    async fetchData(idx_start, idx_end) {
        return await this.options.callbacks.fetchData.call(this, idx_start, idx_end);
    }

    sortDataDefaultCallback() {
        return Promise.resolve();
    }

    async sortData() {
        if (!this.setup_has_run) {
            return;
        }

        // Sort is applied to the filtered data.
        await this.options.callbacks.sortData.call(this);
        await this.#insertToDOM();
    }

    filterDataDefaultCallback() {
        return Promise.resolve(0);
    }

    async filterData() {
        if (!this.setup_has_run) {
            return;
        }

        // Filter is applied to entire dataset.
        const max_items = await this.options.callbacks.filterData.call(this);
        await this.setMaxItems(max_items);
    }

    #exploreEnvironment(rows, cache) {
        this.options.content_tag = this.content_elem.tagName.toLowerCase();
        if (isNullOrUndefined(rows) || !rows.length) {
            return;
        }
        if (this.#ie && this.#ie <= 9 && !this.options.tag) {
            this.options.tag = rows[0].match(/<([^>\s/]*)/)[1].toLowerCase();
        }
        if (this.content_elem.children.length <= 1) {
            cache.data = this.#html(rows[0] + rows[0] + rows[0]);
        }
        if (!this.options.tag) {
            this.options.tag = this.content_elem.children[0].tagName.toLowerCase();
        }
        this.#getRowsHeight();
    }

    #getRowsHeight() {
        const prev_item_height = this.options.item_height;
        const prev_item_width = this.options.item_width;
        const prev_rows_in_block = this.options.rows_in_block;
        const prev_cols_in_block = this.options.cols_in_block;

        this.options.cluster_height = 0;
        this.options.cluster_width = 0;
        if (!this.#max_items) {
            return;
        }

        const rows = this.content_elem.children;
        if (!rows.length) {
            return;
        }
        const nodes = rows[0].children;
        if (!nodes.length) {
            return;
        }

        const node = nodes[Math.floor(nodes.length / 2)];
        const node_dims = getComputedDims(node);
        this.options.item_height = node_dims.height;
        this.options.item_width = node_dims.width;
        // consider table's browser spacing
        if (this.options.tag === "tr" && getStyle("borderCollapse", this.content_elem) !== "collapse") {
            const spacing = parseInt(getStyle("borderSpacing", this.content_elem), 10) || 0;
            this.options.item_height += spacing;
            this.options.item_width += spacing;
        }
        // consider margins and margins collapsing
        if (this.options.tag !== "tr") {
            const margin_top = parseInt(getStyle("marginTop", node), 10) || 0;
            const margin_right = parseInt(getStyle("marginRight", node), 10) || 0;
            const margin_bottom = parseInt(getStyle("marginBottom", node), 10) || 0;
            const margin_left = parseInt(getStyle("marginLeft", node), 10) || 0;
            this.options.item_height += Math.max(margin_top, margin_bottom);
            this.options.item_width += Math.max(margin_left, margin_right);
        }

        // Update rows in block to match the number of elements that can fit in the scroll element view.
        this.options.rows_in_block = calcRowsPerCol(this.scroll_elem, node);
        this.options.cols_in_block = calcColsPerRow(this.content_elem, node);
        console.log("HERE:", this.scroll_elem, this.content_elem, node, this.options.rows_in_block, this.options.cols_in_block);

        this.options.block_height = this.options.item_height * this.options.rows_in_block;
        this.options.block_width = this.options.item_width * this.options.cols_in_block;
        this.options.rows_in_cluster = this.options.blocks_in_cluster * this.options.rows_in_block;
        this.options.cluster_height = this.options.blocks_in_cluster * this.options.block_height;
        this.options.cluster_width = this.options.block_width;

        this.#max_rows = parseInt(this.#max_items / this.options.cols_in_block, 10);

        return (
            prev_item_height !== this.options.item_height ||
            prev_item_width !== this.options.item_width ||
            prev_rows_in_block !== this.options.rows_in_block ||
            prev_cols_in_block !== this.options.cols_in_block
        );
    }

    #getClusterNum() {
        this.options.scroll_top = this.scroll_elem.scrollTop;
        const cluster_divider = this.options.cluster_height - this.options.block_height;
        const current_cluster = Math.floor(this.options.scroll_top / cluster_divider);
        const max_cluster = Math.floor((this.#max_rows * this.options.item_height) / cluster_divider);
        return Math.min(current_cluster, max_cluster);
    }

    #generateEmptyRow() {
        if (!this.options.tag || !this.options.show_no_data_row) {
            return [];
        }

        const empty_row = document.createElement(this.options.tag);
        const no_data_content = document.createTextNode(this.options.no_data_text);
        empty_row.className = this.options.no_data_class;
        if (this.options.tag === "tr") {
            const td = document.createElement("td");
            // fixes #53
            td.colSpan = 100;
            td.appendChild(no_data_content);
            empty_row.appendChild(td);
        } else {
            empty_row.appendChild(no_data_content);
        }
        return [empty_row.outerHTML];
    }

    async #generate() {
        const rows_start = Math.max(0, (this.options.rows_in_cluster - this.options.rows_in_block) * this.#getClusterNum());
        const rows_end = rows_start + this.options.rows_in_cluster;
        const top_offset = Math.max(0, rows_start * this.options.item_height);
        const bottom_offset = Math.max(0, (this.#max_rows - rows_end) * this.options.item_height);
        const rows_above = top_offset < 1 ? rows_start + 1 : rows_start;

        const idx_start = Math.max(0, rows_start * this.options.cols_in_block);
        const idx_end = rows_end * this.options.cols_in_block;
        const this_cluster_rows = await this.fetchData(idx_start, idx_end);
        return {
            top_offset: top_offset,
            bottom_offset: bottom_offset,
            rows_above: rows_above,
            rows: Array.isArray(this_cluster_rows) ? this_cluster_rows : [],
        };
    }

    async #insertToDOM() {
        if (!this.options.cluster_height || !this.options.cluster_width) {
            console.log("HERE");
            const rows = await this.fetchData(0, 1);
            this.#exploreEnvironment(rows, this.#cache);
        }

        const data = await this.#generate();
        let this_cluster_rows = [];
        for (let i = 0; i < data.rows.length; i += this.options.cols_in_block) {
            const new_row = data.rows.slice(i, i + this.options.cols_in_block).join("");
            this_cluster_rows.push(`<div class="clusterize-row">${new_row}</div>`);
        }
        this_cluster_rows = this_cluster_rows.join("");
        const this_cluster_content_changed = this.#checkChanges("data", this_cluster_rows, this.#cache);
        const top_offset_changed = this.#checkChanges("top", data.top_offset, this.#cache);
        const only_bottom_offset_changed = this.#checkChanges("bottom", data.bottom_offset, this.#cache);
        const layout = [];

        if (this_cluster_content_changed || top_offset_changed) {
            if (data.top_offset) {
                this.options.keep_parity && layout.push(this.#renderExtraTag("keep-parity"));
                layout.push(this.#renderExtraTag("top-space", data.top_offset));
            }
            layout.push(this_cluster_rows);
            data.bottom_offset && layout.push(this.#renderExtraTag("bottom-space", data.bottom_offset));
            this.options.callbacks.clusterWillChange && this.options.callbacks.clusterWillChange();
            this.#html(layout.join(""));
            this.options.content_tag === "ol" && this.content_elem.setAttribute("start", data.rows_above);
            this.content_elem.style["counter-increment"] = `clusterize-counter ${data.rows_above - 1}`;
            this.options.callbacks.clusterChanged && this.options.callbacks.clusterChanged();
        } else if (only_bottom_offset_changed) {
            this.content_elem.lastChild.style.height = `${data.bottom_offset}px`;
        }
    }

    #html(data) {
        const content_elem = this.content_elem;
        if (this.#ie && this.#ie <= 9 && this.options.tag === "tr") {
            const div = document.createElement("div");
            let last;
            div.innerHTML = `<table><tbody>${data}</tbody></table>`;
            while ((last = content_elem.lastChild)) {
                content_elem.removeChild(last);
            }
            const rows_nodes = this.#getChildNodes(div.firstChild.firstChild);
            while (rows_nodes.length) {
                content_elem.appendChild(rows_nodes.shift());
            }
        } else {
            content_elem.innerHTML = data;
        }
    }

    #renderExtraTag(class_name, height) {
        const tag = document.createElement(this.options.tag);
        const clusterize_prefix = "clusterize-";
        tag.className = [
            `${clusterize_prefix}extra-row`,
            `${clusterize_prefix}${class_name}`,
        ].join(" ");
        height && (tag.style.height = `${height}px`);
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
        if (this.#is_mac) {
            if (!this.#pointer_events_set) {
                this.content_elem.style.pointerEvents = "none";
                this.#pointer_events_set = true;
                clearTimeout(this.#scroll_debounce);
                this.#scroll_debounce = setTimeout(() => {
                    this.content_elem.style.pointerEvents = "auto";
                    this.#pointer_events_set = false;
                }, SCROLL_DEBOUNCE_TIME_MS);
            }
        }
        if (this.#last_cluster !== (this.#last_cluster = this.#getClusterNum())) {
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

        // If association for elements is broken, replace them with instance version.
        if (!this.scroll_elem.isConnected || !this.content_elem.isConnected) {
            document.getElementByid(this.scroll_id).replaceWith(this.scroll_elem);
            // refresh since sizes may have changed.
            this.refresh(true);
        }
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
            return elem.addEventListener(type, event => listener.call(this), false);
        } else {
            return elem.attachEvent(`on${type}`, event => listener.call(this));
        }
    }

    #teardownEvent(type, elem, listener) {
        if (elem.removeEventListener) {
            return elem.removeEventListener(type, event => listener.call(this), false);
        } else {
            return elem.detachEvent(`on${type}`, event => listener.call(this));
        }
    }
}
