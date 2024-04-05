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
    #options = {};
    #is_mac = null;
    #ie = null;
    #n_rows = null;
    #cache = {};
    #scroll_top = 0;
    #last_cluster = false;
    #scroll_debounce = 0;
    #resize_observer = null;
    #resize_observer_timer = null;
    #element_observer = null;
    #element_observer_timer = null;
    #pointer_events_set = false;
    #sort_mode = "";
    #sort_dir = "";

    constructor(args) {
        const defaults = {
            rows_in_block: 50,
            blocks_in_cluster: 4,
            tag: null,
            show_no_data_row: true,
            no_data_class: 'clusterize-no-data',
            no_data_text: 'No data',
            keep_parity: true,
            callbacks: {}
        };

        const options = [
            'rows_in_block',
            'blocks_in_cluster',
            'show_no_data_row',
            'no_data_class',
            'no_data_text',
            'keep_parity',
            'tag',
            'callbacks',
        ];

        // detect ie9 and lower
        // https://gist.github.com/padolsey/527683#comment-786682
        this.#ie = (function () {
            for (var v = 3,
                el = document.createElement('b'),
                all = el.all || [];
                el.innerHTML = '<!--[if gt IE ' + (++v) + ']><i><![endif]-->',
                all[0];
            ) { }
            return v > 4 ? v : document.documentMode;
        }())
        this.#is_mac = navigator.platform.toLowerCase().indexOf('mac') + 1;

        for (let i = 0, option; option = options[i]; i++) {
            this.#options[option] = !isNullOrUndefined(args[option]) ? args[option] : defaults[option];
        }

        this.scroll_elem = args["scrollId"] ? document.getElementById(args["scrollId"]) : args["scrollElem"];
        if (!isElement(this.scroll_elem)) {
            throw new Error("Error! Could not find scroll element");
        }
        this.scroll_id = this.scroll_elem.id;

        this.content_elem = args["contentId"] ? document.getElementById(args["contentId"]) : args["contentElem"];
        if (!isElement(this.content_elem)) {
            throw new Error("Error! Could not find content element");
        }
        this.content_id = this.content_elem.id;

        if (!this.content_elem.hasAttribute("tabindex")) {
            this.content_elem.setAttribute("tabindex", 0);
        }

        this.#scroll_top = this.scroll_elem.scrollTop;

        if (!isNumber(args.n_rows)) {
            throw new Error("Invalid argument. n_rows expected number, got:", typeof args.n_rows);
        }
        this.#n_rows = args.n_rows;

        if (!this.#options.callbacks.fetchData) {
            this.#options.callbacks.fetchData = this.#fetchDataDefault;
        }
        if (!this.#options.callbacks.sortData) {
            this.#options.callbacks.sortData = this.#sortDataDefault;
        }
        if (!this.#options.callbacks.filterData) {
            this.#options.callbacks.filterData = this.#filterDataDefault;
        }
    }

    // ==== PUBLIC FUNCTIONS ====
    async setup() {
        await this.#insertToDOM();
        this.scroll_elem.scrollTop = this.#scroll_top;

        this.#setupEvent("scroll", this.scroll_elem, this.#onScroll);
        this.#setupElementObservers();
        this.#setupResizeObservers();
    }

    destroy() {
        this.#teardownEvent("scroll", this.scroll_elem, this.#onScroll);
        this.#teardownElementObservers();
        this.#teardownResizeObservers();
        this.#html(this.#generateEmptyRow().join(""));
    }

    refresh(force) {
        if (this.#getRowsHeight() || force) {
            this.update()
        }
    }

    async update() {
        this.#scroll_top = this.scroll_elem.scrollTop;
        // fixes #39
        if (this.#n_rows * this.#options.item_height < this.#scroll_top) {
            this.scroll_elem.scrollTop = 0;
            this.#last_cluster = 0;
        }

        await this.#insertToDOM();
        this.scroll_elem.scrollTop = this.#scroll_top;
    }

    getRowsAmount() {
        return this.#n_rows;
    }

    getScrollProgress() {
        return this.#options.scroll_top / (this.#n_rows * this.#options.item_height) * 100 || 0;
    }

    async filterData(filter) {
        // Filter is applied to entire dataset.
        const n_rows = await this.#options.callbacks.filterData(filter);
        // If the number of rows changed after filter, we need to update the cluster.
        if (n_rows !== this.#n_rows) {
            this.#n_rows = n_rows;
            this.refresh(true);
        }
        // Apply sort to the new filtered data.
        await this.sortData(this.#sort_mode, this.#sort_dir);
    }

    async sortData(mode, dir) {
        // Sort is applied to the filtered data.

        // update instance sort settings to the passed values.
        this.#sort_mode = mode;
        this.#sort_dir = dir;

        await this.#options.callbacks.sortData(this.#sort_mode, this.#sort_dir === "descending");
        await this.#insertToDOM();
    }

    // ==== PRIVATE FUNCTIONS ====

    #fetchDataDefault() {
        return Promise.resolve([]);
    }

    #sortDataDefault() {
        return Promise.resolve([]);
    }

    #filterDataDefault() {
        return Promise.resolve([]);
    }

    #exploreEnvironment(rows, cache) {
        this.#options.content_tag = this.content_elem.tagName.toLowerCase();
        if (!rows.length) {
            return;
        }
        if (this.#ie && this.#ie <= 9 && !this.#options.tag) {
            this.#options.tag = rows[0].match(/<([^>\s/]*)/)[1].toLowerCase();
        }
        if (this.content_elem.children.length <= 1) {
            cache.data = this.#html(rows[0] + rows[0] + rows[0]);
        }
        if (!this.#options.tag) {
            this.#options.tag = this.content_elem.children[0].tagName.toLowerCase();
        }
        this.#getRowsHeight();
    }

    #getRowsHeight() {
        const prev_item_height = this.#options.item_height;
        const prev_rows_in_block = this.#options.rows_in_block;

        this.#options.cluster_height = 0;
        if (!this.#n_rows) {
            return;
        }

        const nodes = this.content_elem.children;
        if (!nodes.length) {
            return;
        }
        const node = nodes[Math.floor(nodes.length / 2)];
        this.#options.item_height = node.offsetHeight;
        // consider table's browser spacing
        if (this.#options.tag === "tr" && getStyle("borderCollapse", this.content_elem) !== "collapse") {
            this.#options.item_height += parseInt(getStyle("borderSpacing", this.content_elem), 10) || 0;
        }
        // consider margins and margins collapsing
        if (this.#options.tag !== "tr") {
            const margin_top = parseInt(getStyle("marginTop", node), 10) || 0;
            const margin_bottom = parseInt(getStyle("marginBottom", node), 10) || 0;
            this.#options.item_height += Math.max(margin_top, margin_bottom);
        }

        // Update rows in block to match the number of elements that can fit in the scroll element view.
        this.#options.rows_in_block = parseInt(this.scroll_elem.clientHeight / this.#options.item_height);

        this.#options.block_height = this.#options.item_height * this.#options.rows_in_block;
        this.#options.rows_in_cluster = this.#options.blocks_in_cluster * this.#options.rows_in_block;
        this.#options.cluster_height = this.#options.blocks_in_cluster * this.#options.block_height;
        return prev_item_height !== this.#options.item_height || prev_rows_in_block !== this.#options.rows_in_block;
    }

    #getClusterNum() {
        this.#options.scroll_top = this.scroll_elem.scrollTop;
        const cluster_divider = this.#options.cluster_height - this.#options.block_height;
        const current_cluster = Math.floor(this.#options.scroll_top / cluster_divider);
        const max_cluster = Math.floor((this.#n_rows * this.#options.item_height) / cluster_divider);
        return Math.min(current_cluster, max_cluster);
    }

    #generateEmptyRow() {
        if (!this.#options.tag || !this.#options.show_no_data_row) {
            return [];
        }

        const empty_row = document.createElement(this.#options.tag);
        const no_data_content = document.createTextNode(this.#options.no_data_text);
        empty_row.className = this.#options.no_data_class;
        if (this.#options.tag === "tr") {
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
        const items_start = Math.max((this.#options.rows_in_cluster - this.#options.rows_in_block) * this.#getClusterNum(), 0);
        const items_end = items_start + this.#options.rows_in_cluster;
        const top_offset = Math.max(items_start * this.#options.item_height, 0);
        const bottom_offset = Math.max((this.#n_rows - items_end) * this.#options.item_height, 0);
        const rows_above = top_offset < 1 ? items_start + 1 : items_start;

        const this_cluster_rows = await this.#options.callbacks.fetchData(items_start, items_end);
        return {
            top_offset: top_offset,
            bottom_offset: bottom_offset,
            rows_above: rows_above,
            rows: this_cluster_rows,
        };
    }

    async #insertToDOM() {
        if (!this.#options.cluster_height) {
            const rows = await this.#options.callbacks.fetchData(0, 1);
            this.#exploreEnvironment(rows, this.#cache);
        }

        const data = await this.#generate();
        const this_cluster_rows = data.rows.join("");
        const this_cluster_content_changed = this.#checkChanges("data", this_cluster_rows, this.#cache);
        const top_offset_changed = this.#checkChanges("top", data.top_offset, this.#cache);
        const only_bottom_offset_changed = this.#checkChanges("bottom", data.bottom_offset, this.#cache);
        const layout = [];

        if (this_cluster_content_changed || top_offset_changed) {
            if (data.top_offset) {
                this.#options.keep_parity && layout.push(this.#renderExtraTag("keep-parity"));
                layout.push(this.#renderExtraTag("top-space", data.top_offset));
            }
            layout.push(this_cluster_rows);
            data.bottom_offset && layout.push(this.#renderExtraTag("bottom-space", data.bottom_offset));
            this.#options.callbacks.clusterWillChange && this.#options.callbacks.clusterWillChange();
            this.#html(layout.join(""));
            this.#options.content_tag === "ol" && this.content_elem.setAttribute("start", data.rows_above);
            this.content_elem.style["counter-increment"] = `clusterize-counter ${data.rows_above - 1}`;
            this.#options.callbacks.clusterChanged && this.#options.callbacks.clusterChanged();
        } else if (only_bottom_offset_changed) {
            this.content_elem.lastChild.style.height = `${data.bottom_offset}px`;
        }
    }

    #html(data) {
        const content_elem = this.content_elem;
        if (this.#ie && this.#ie <= 9 && this.#options.tag === "tr") {
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
        const tag = document.createElement(this.#options.tag);
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
        if (this.#options.callbacks.scrollingProgress) {
            this.#options.callbacks.scrollingProgress(this.getScrollingProgress());
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
