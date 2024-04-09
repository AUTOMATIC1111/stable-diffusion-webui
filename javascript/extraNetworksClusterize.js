class NotImplementedError extends Error {
    constructor(...params) {
        super(...params);
        
        if (Error.captureStackTrace) {
            Error.captureStackTrace(this, NotImplementedError);
        }

        this.name = "NotImplementedError";
    }
}

class ExtraNetworksClusterize extends Clusterize {
    data_obj = {};
    data_obj_keys_sorted = [];
    sort_reverse = false;
    default_sort_fn = this.sortByDivId;
    sort_fn = this.default_sort_fn; 
    tabname = "";
    extra_networks_tabname = "";
    enabled = false;

    // Override base class defaults
    default_sort_mode_str = "divId";
    default_sort_dir_str = "ascending";
    default_filter_str = "";
    sort_mode_str = this.default_sort_mode_str;
    sort_dir_str = this.default_sort_dir_str;
    filter_str = this.default_filter_str;

    constructor(args) {
        super(args);
        this.tabname = getValueThrowError(args, "tabname");
        this.extra_networks_tabname = getValueThrowError(args, "extra_networks_tabname");
    }

    sortByDivId(data) {
        /** Sort data_obj keys (div_id) as numbers. */
        return Object.keys(data).sort(INT_COLLATOR.compare);
    }

    async reinitData() {
        await this.initData();
        // can't use super class' sort since it relies on setup being run first.
        // but we do need to make sure to sort the new data before continuing.
        await this.options.callbacks.sortData.call(this);
        await this.setMaxItems(Object.keys(this.data_obj).length);
    }

    async setup() {
        if (this.setup_has_run) {
            return;
        }

        await this.reinitData();

        if (this.enabled) {
            await super.setup();
        }
    }

    async load(force_init_data) {
        if (!this.enabled) {
            return;
        }

        if (!this.setup_has_run) {
            await this.setup();
        } else if (force_init_data) {
            await this.reinitData();
        } else {
            await this.refresh(true);
        }
    }

    enable(state) {
        // if no state is passed, we enable by default.
        this.enabled = state !== false;
    }

    clear() {
        this.data_obj = {};
        this.data_obj_keys_sorted = [];
        super.clear();
    }

    setSortMode(sort_mode_str) {
        if (this.sort_mode_str === sort_mode_str) {
            return;
        }

        this.sort_mode_str = sort_mode_str;
        this.sortData();
    }

    setSortDir(sort_dir_str) {
        const reverse = (sort_dir_str === "descending");
        if (this.sort_reverse === reverse) {
            return;
        }

        this.sort_dir_str = sort_dir_str;
        this.sort_reverse = reverse;
        this.sortData();
    }

    setFilterStr(filter_str) {
        if (isString(filter_str) && this.filter_str !== filter_str.toLowerCase()) {
            this.filter_str = filter_str.toLowerCase();
        } else if (isNullOrUndefined(this.filter_str)) {
            this.filter_str = this.default_filter_str;
        } else {
            return;
        }

        this.filterData();
    }

    async initDataDefaultCallback() {
        throw new NotImplementedError();
    }

    async fetchDataDefaultCallback() {
        throw new NotImplementedError();
    }

    async sortDataDefaultCallback() {
        this.data_obj_keys_sorted = this.sort_fn(this.data_obj);
        if (this.sort_reverse) {
            this.data_obj_keys_sorted = this.data_obj_keys_sorted.reverse();
        }
    }

    async filterDataDefaultCallback() {
        throw new NotImplementedError();
    }
}

class ExtraNetworksClusterizeTreeList extends ExtraNetworksClusterize {
    selected_div_id = null;

    constructor(args) {
        super({
            ...args,
            no_data_text: "Directory is empty.",
        });

    }

    clear() {
        this.selected_div_id = null;
        super.clear();
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

    #setVisibility(div_id, visible) {
        /** Recursively sets the visibility of a div_id and its children. */
        for (const child_id of this.data_obj[div_id].children) {
            this.data_obj[child_id].visible = visible;
            if (visible) {
                if (this.data_obj[div_id].expanded) {
                    this.#setVisibility(child_id, visible);
                }
            } else {
                if (this.selected_div_id === child_id) {
                    this.selected_div_id = null;
                }
                this.#setVisibility(child_id, visible);
            }
        }
    }

    onRowSelected(div_id, elem, override) {
        /** Selects a row and deselects all others. */
        if (!isElementLogError(elem)) {
            return;
        }

        if (!keyExistsLogError(this.data_obj, div_id)) {
            return;
        }

        override = override === true;

        if (!isNullOrUndefined(this.selected_div_id) && div_id !== this.selected_div_id) {
            // deselect current selection if exists on page
            const prev_elem = this.content_elem.querySelector(`div[data-div-id="${this.selected_div_id}"]`);
            if (isElement(prev_elem)) {
                delete prev_elem.dataset.selected;
            }
        }

        elem.toggleAttribute("data-selected");
        this.selected_div_id = "selected" in elem.dataset ? div_id : null;
    }

    getMaxRowWidth() {
        /** Calculates the width of the widest row in the list. */
        if (!this.enabled) {
            // Inactive list is not displayed on screen. Can't calculate size.
            return false;
        }
        if (this.content_elem.children.length === 0) {
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

    async onRowExpandClick(div_id, elem) {
        /** Expands or collapses a row to show/hide children. */
        if (!keyExistsLogError(this.data_obj, div_id)){
            return;
        }

        // Toggle state
        this.data_obj[div_id].expanded = !this.data_obj[div_id].expanded;

        const visible = this.data_obj[div_id].expanded;
        for (const child_id of this.data_obj[div_id].children) {
            this.#setVisibility(child_id, visible)
        }
        this.#setVisibility()

        await this.setMaxItems(Object.values(this.data_obj).filter(v => v.visible).length);
    }

    async initData() {
        /*Expects an object like the following:
            {
                parent: null or div_id,
                children: array of div_id's,
                visible: bool,
                expanded: bool,
            }
        */
        this.data_obj = await this.options.callbacks.initData.call(
            this,
            this.tabname,
            this.extra_networks_tabname,
            this.constructor.name,
        );
    }

    async fetchData(idx_start, idx_end) {
        if (!this.enabled) {
            return [];
        }
        const n_items = idx_end - idx_start;
        const div_ids = [];
        for (const div_id of this.data_obj_keys_sorted.slice(idx_start)) {
            if (this.data_obj[div_id].visible) {
                div_ids.push(div_id);
            }
            if (div_ids.length >= n_items) {
                break;
            }
        }

        const data = await this.options.callbacks.fetchData.call(
            this,
            this.constructor.name,
            this.extra_networks_tabname,
            div_ids,
        );

        // we have to calculate the box shadows here since the element is on the page
        // at this point and we can get its computed styles.
        const style = getComputedStyle(document.body);
        const text_size = style.getPropertyValue("--button-large-text-size");

        const res = [];
        for (const [div_id, html_str] of Object.entries(data)) {
            const parsed_html = htmlStringToElement(html_str);
            const depth = Number(parsed_html.dataset.depth);
            parsed_html.style.paddingLeft = `calc(${depth} * ${text_size})`;
            parsed_html.style.boxShadow = this.getBoxShadow(depth);
            if (this.data_obj[div_id].expanded) {
                parsed_html.dataset.expanded = "";
            }
            if (div_id === this.selected_div_id) {
                parsed_html.dataset.selected = "";
            }
            res.push(parsed_html.outerHTML);
        }

        return res;
    }

    async filterDataDefaultCallback() {
        // just return the number of visible objects in our data.
        return Object.values(this.data_obj).filter(v => v.visible).length;
    }
}

class ExtraNetworksClusterizeCardsList extends ExtraNetworksClusterize {
    constructor(args) {
        super({
            ...args,
            no_data_text: "No files matching filter.",
        });
    }

    sortByName(data) {
        return Object.keys(data).sort((a, b) => {
            return STR_COLLATOR.compare(data[a].sort_name, data[b].sort_name);
        });
    }

    sortByPath(data) {
        return Object.keys(data).sort((a, b) => {
            return STR_COLLATOR.compare(data[a].sort_path, data[b].sort_path);
        });
    }

    sortByDateCreated(data) {
        return Object.keys(data).sort((a, b) => {
            return INT_COLLATOR.compare(data[a].sort_date_created, data[b].sort_date_created);
        });
    }

    sortByDateModified(data) {
        return Object.keys(data).sort((a, b) => {
            return INT_COLLATOR.compare(data[a].sort_date_modified, data[b].sort_date_modified);
        });
    }

    async initData() {
        /*Expects an object like the following:
            {
                search_keys: array of strings,
                sort_<mode>: string, (for various sort modes)
            }
        */
        this.data_obj = await this.options.callbacks.initData.call(
            this,
            this.tabname,
            this.extra_networks_tabname,
            this.constructor.name,
        );
    }

    async fetchData(idx_start, idx_end) {
        if (!this.enabled) {
            return;
        }
        const n_items = idx_end - idx_start;
        const div_ids = [];
        for (const div_id of this.data_obj_keys_sorted.slice(idx_start)) {
            if (this.data_obj[div_id].visible) {
                div_ids.push(div_id);
            }
            if (div_ids.length >= n_items) {
                break;
            }
        }
        
        const data = await this.options.callbacks.fetchData.call(
            this,
            this.constructor.name,
            this.extra_networks_tabname,
            div_ids,
        );

        return Object.values(data);
    }

    async sortData() {
        switch (this.sort_mode_str) {
            case "name":
                this.sort_fn = this.sortByName;
                break;
            case "path":
                this.sort_fn = this.sortByPath;
                break;
            case "date_created":
                this.sort_fn = this.sortByDateCreated;
                break;
            case "date_modified":
                this.sort_fn = this.sortByDateModified;
                break;
            default:
                this.sort_fn = this.default_sort_fn;
                break;
        }
        await super.sortData()
    }

    async filterDataDefaultCallback() {
        /** Filters data by a string and returns number of items after filter. */
        let n_visible = 0;
        for (const [div_id, v] of Object.entries(this.data_obj)) {
            let visible = v.search_terms.indexOf(this.filter_str) != -1;
            if (v.search_only && this.filter_str.length < 4) {
                visible = false;
            }
            this.data_obj[div_id].visible = visible;
            if (visible) {
                n_visible++;
            }
        }

        return n_visible;
    }
}