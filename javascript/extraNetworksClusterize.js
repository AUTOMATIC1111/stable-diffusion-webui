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

    // Override base class defaults
    default_sort_mode_str = "divId";
    default_sort_dir_str = "ascending";
    default_filter_str = "";
    sort_mode_str = this.default_sort_mode_str;
    sort_dir_str = this.default_sort_dir_str;
    filter_str = this.default_filter_str;

    constructor(...args) {
        super(...args);

        // finish initialization
        this.tabname = getValueThrowError(...args, "tabname");
        this.extra_networks_tabname = getValueThrowError(...args, "extra_networks_tabname");
    }

    sortByDivId() {
        /** Sort data_obj keys (div_id) as numbers. */
        this.data_obj_keys_sorted = Object.keys(this.data_obj).sort(INT_COLLATOR.compare);
    }

    clear() {
        this.data_obj = {};
        this.data_obj_keys_sorted = [];
        super.clear();
    }

    async initDataDefault() {
        /**Fetches the initial data.
         * 
         * This data should be minimal and only contain div IDs and other necessary
         * information such as sort keys and terms for filtering.
         */
        throw new NotImplementedError();
    }

    async fetchDataDefault(idx_start, idx_end) {
        throw new NotImplementedError();
    }

    async sortDataDefault(sort_mode_str, sort_dir_str) {
        this.sort_mode_str = sort_mode_str;
        this.sort_dir_str = sort_dir_str;
        this.sort_reverse = sort_dir_str === "descending";

        this.data_obj_keys_sorted = this.sort_fn(this.data_obj);
        if (this.sort_reverse) {
            this.data_obj_keys_sorted = this.data_obj_keys_sorted.reverse();
        }
    }

    async filterDataDefault(filter_str) {
        throw new NotImplementedError();
    }
}

class ExtraNetworksClusterizeTreeList extends ExtraNetworksClusterize {
    selected_div_id = null;

    constructor(args) {
        args.no_data_text = "Directory is empty.";
        super(args);

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

    async initDataDefault() {
        /*Expects an object like the following:
            {
                parent: null or div_id,
                children: array of div_id's,
                visible: bool,
                expanded: bool,
            }
        */
        console.log("BLAH:", this.options.callbacks.initData);
        this.data_obj = await this.options.callbacks.initData(this.constructor.name);
    }

    async fetchDataDefault(idx_start, idx_end) {
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

        const data = await this.options.callbacks.fetchData(
            this.constructor.name,
            this.extra_networks_tabname,
            div_ids,
        );

        // we have to calculate the box shadows here since the element is on the page
        // at this point and we can get its computed styles.
        const style = getComputedStyle(document.body);
        const text_size = style.getPropertyValue("--button-large-text-size");

        const res = [];
        for (const [div_id, item] of Object.entries(data)) {
            const parsed_html = htmlStringToElement(item);
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

        return rows;
    }

    async sortDataDefault(sort_mode, sort_dir) {
        throw new NotImplementedError();
    }

    async filterDataDefault(filter_str) {
        // just return the number of visible objects in our data.
        return Object.values(this.data_obj).filter(v => v.visible).length;
    }
}

class ExtraNetworksClusterizeCardsList extends ExtraNetworksClusterize {
    constructor(args) {
        args.no_data_text = "No files matching filter.";
        super(args);
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

    async initDataDefault() {
        /*Expects an object like the following:
            {
                search_keys: array of strings,
                sort_<mode>: string, (for various sort modes)
            }
        */
        console.log("HERE:", this.options.callbacks);
        this.data_obj = await this.options.callbacks.initData(this.constructor.name);
    }

    async fetchDataDefault(idx_start, idx_end) {
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
        
        const data = await this.options.callbacks.fetchData(
            this.constructor.name,
            this.extra_networks_tabname,
            div_ids,
        );

        return Object.values(data);
    }

    async sortDataDefault(sort_mode_str, sort_dir_str) {
        switch (sort_mode_str) {
            case "name":
                this.sort_fn = this.sortByName;
                break;
            case "path":
                this.sort_fn = this.sortByPath;
                break;
            case "created":
                this.sort_fn = this.sortByDateCreated;
                break;
            case "modified":
                this.sort_fn = this.sortByDateModified;
                break;
            default:
                this.sort_fn = this.default_sort_fn;
                break;
        }
        await super.sortDataDefault(sort_mode_str, sort_dir_str)
    }

    async filterDataDefault(filter_str) {
        /** Filters data by a string and returns number of items after filter. */
        if (isString(filter_str)) {
            this.filter_str = filter_str.toLowerCase();
        } else if (isNullOrUndefined(this.filter_str)) {
            this.filter_str = this.default_filter_str;
        }

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