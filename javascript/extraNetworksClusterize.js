// Prevent eslint errors on functions defined in other files.
/*global
    Clusterize,
    getValueThrowError,
    INT_COLLATOR,
    STR_COLLATOR,
    LRUCache,
    isString,
    isNullOrUndefined,
    isNullOrUndefinedLogError,
    isElement,
    isElementLogError,
    keyExistsLogError,
    htmlStringToElement,
*/
/*eslint no-undef: "error"*/

// number of list html items to store in cache.
const EXTRA_NETWORKS_CLUSTERIZE_LRU_CACHE_SIZE = 1000;

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
    lru = null;
    sort_reverse = false;
    default_sort_fn = this.sortByDivId;
    sort_fn = this.default_sort_fn;
    tabname = "";
    extra_networks_tabname = "";

    // Override base class defaults
    default_sort_mode_str = "divId";
    default_sort_dir_str = "ascending";
    default_filter_str = "";
    default_directory_filter_str = "";
    sort_mode_str = this.default_sort_mode_str;
    sort_dir_str = this.default_sort_dir_str;
    filter_str = this.default_filter_str;
    directory_filter_str = this.default_directory_filter_str;

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
        await this.setMaxItems(Object.keys(this.data_obj).length);
        await this.refresh(true);
        await this.options.callbacks.sortData();
    }

    async setup() {
        if (this.setup_has_run || !this.enabled) {
            return;
        }

        if (this.lru instanceof LRUCache) {
            this.lru.clear();
        } else {
            this.lru = new LRUCache(EXTRA_NETWORKS_CLUSTERIZE_LRU_CACHE_SIZE);
        }

        await this.reinitData();

        if (this.enabled) {
            await super.setup();
        }
    }

    destroy() {
        this.data_obj = {};
        this.data_obj_keys_sorted = [];
        if (this.lru instanceof LRUCache) {
            this.lru.destroy();
            this.lru = null;
        }
        super.destroy();
    }

    clear() {
        this.data_obj = {};
        this.data_obj_keys_sorted = [];
        if (this.lru instanceof LRUCache) {
            this.lru.clear();
        }
        super.clear("Loading...");
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
            await this.refresh();
        }
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
        if (isString(filter_str) && this.filter_str !== filter_str) {
            this.filter_str = filter_str;
        } else if (isNullOrUndefined(filter_str)) {
            this.filter_str = this.default_filter_str;
        }
        this.filterData();
    }

    setDirectoryFilterStr(filter_str) {
        if (isString(filter_str) && this.directory_filter_str !== filter_str) {
            this.directory_filter_str = filter_str;
        } else if (isNullOrUndefined(filter_str)) {
            this.directory_filter_str = this.default_directory_filter_str;
        }
        this.filterData();
    }

    async initDataDefaultCallback() {
        throw new NotImplementedError();
    }

    idxRangeToDivIds(idx_start, idx_end) {
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
        return div_ids;
    }

    async fetchDivIds(div_ids) {
        if (isNullOrUndefinedLogError(this.lru)) {
            return [];
        }
        if (Object.keys(this.data_obj).length === 0) {
            return [];
        }
        const lru_keys = Array.from(this.lru.cache.keys());
        const cached_div_ids = div_ids.filter(x => lru_keys.includes(x));
        const missing_div_ids = div_ids.filter(x => !lru_keys.includes(x));

        const data = {};
        // Fetch any div IDs not in the LRU Cache using our callback.
        if (missing_div_ids.length !== 0) {
            const fetched_data = await this.options.callbacks.fetchData(missing_div_ids);
            if (Object.keys(fetched_data).length !== missing_div_ids.length) {
                // expected data. got nothing.
                return {};
            }
            Object.assign(data, fetched_data);
        }

        // Now load any cached IDs from the LRU Cache
        for (const div_id of cached_div_ids) {
            if (!keyExistsLogError(this.data_obj, div_id)) {
                continue;
            }
            if (this.data_obj[div_id].visible) {
                data[div_id] = this.lru.get(div_id);
            }
        }

        return data;
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
        super.clear("Loading...");
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
        const this_obj = this.data_obj[div_id];
        this_obj.visible = visible;
        if (!visible && div_id === this.selected_div_id) {
            this.selected_div_id = null;
        }
        for (const child_id of this_obj.children) {
            this.#setVisibility(child_id, visible && this_obj.expanded);
        }
    }

    onRowSelected(div_id, elem, override) {
        /** Selects a row and deselects all others.
         *
         *  If both `div_id` and `elem` are null/undefined, then we deselect all rows.
         *  `override` allows us to manually set the new state instead of toggling.
        */
        if (isNullOrUndefined(div_id) && isNullOrUndefined(elem)) {
            if (!isNullOrUndefined(this.selected_div_id) && keyExistsLogError(this.data_obj, this.selected_div_id)) {
                this.selected_div_id = null;
                for (const elem of this.content_elem.children) {
                    delete elem.dataset.selected;
                }
            }
            return;
        }

        if (!isElementLogError(elem)) {
            return;
        }

        if (!keyExistsLogError(this.data_obj, div_id)) {
            return;
        }

        if (!isNullOrUndefined(this.selected_div_id) && div_id !== this.selected_div_id) {
            const prev_elem = this.content_elem.querySelector(`[data-div-id="${this.selected_div_id}"]`);
            // deselect current selection if exists on page
            if (isElement(prev_elem)) {
                delete prev_elem.dataset.selected;
                this.selected_div_id = null;
            }
        }
        elem.toggleAttribute("data-selected", override);
        this.selected_div_id = "selected" in elem.dataset ? div_id : null;
    }

    getMaxRowWidth() {
        /** Calculates the width of the widest row in the list. */
        if (!this.enabled) {
            // Inactive list is not displayed on screen. Can't calculate size.
            return;
        }
        if (this.content_elem.children.length === 0) {
            // If there is no data then just skip.
            return;
        }

        let max_width = 0;
        for (let i = 0; i < this.content_elem.children.length; i += this.options.cols_in_block) {
            let row_width = 0;
            for (let j = 0; j < this.options.cols_in_block; j++) {
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

    async onExpandAllClick(div_id) {
        if (!keyExistsLogError(this.data_obj, div_id)) {
            return;
        }

        const _expand = (parent_id) => {
            const this_obj = this.data_obj[parent_id];
            this_obj.visible = true;
            this_obj.expanded = true;
            for (const child_id of this_obj.children) {
                _expand(child_id);
            }
        };

        this.data_obj[div_id].expanded = true;
        for (const child_id of this.data_obj[div_id].children) {
            _expand(child_id);
        }

        const new_len = Object.values(this.data_obj).filter(v => v.visible).length;
        await this.setMaxItems(new_len);
        await this.refresh(true);
        await this.sortData();
    }

    async onCollapseAllClick(div_id) {
        if (!keyExistsLogError(this.data_obj, div_id)) {
            return;
        }

        const _collapse = (parent_id) => {
            const this_obj = this.data_obj[parent_id];
            this_obj.visible = false;
            this_obj.expanded = false;
            for (const child_id of this_obj.children) {
                _collapse(child_id);
            }
        };

        this.data_obj[div_id].expanded = false;
        for (const child_id of this.data_obj[div_id].children) {
            _collapse(child_id);
        }

        const new_len = Object.values(this.data_obj).filter(v => v.visible).length;
        await this.setMaxItems(new_len);
        await this.refresh(true);
        await this.sortData();
    }

    async onRowExpandClick(div_id, elem) {
        /** Expands or collapses a row to show/hide children. */
        if (!keyExistsLogError(this.data_obj, div_id)) {
            return;
        }

        // Toggle state
        this.data_obj[div_id].expanded = !this.data_obj[div_id].expanded;

        const visible = this.data_obj[div_id].expanded;
        for (const child_id of this.data_obj[div_id].children) {
            this.#setVisibility(child_id, visible);
        }

        const new_len = Object.values(this.data_obj).filter(v => v.visible).length;
        await this.setMaxItems(new_len);
        await this.refresh(true);
        await this.sortData();
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
        this.data_obj = await this.options.callbacks.initData();
    }

    async fetchData(idx_start, idx_end) {
        if (!this.enabled) {
            return [];
        }

        if (Object.keys(this.data_obj).length === 0) {
            return [];
        }

        const data = await this.fetchDivIds(this.idxRangeToDivIds(idx_start, idx_end));
        const data_ids_sorted = Object.keys(data).sort((a, b) => {
            return this.data_obj_keys_sorted.indexOf(a) - this.data_obj_keys_sorted.indexOf(b);
        });

        // we have to calculate the box shadows here since the element is on the page
        // at this point and we can get its computed styles.
        const style = getComputedStyle(document.body);
        const text_size = style.getPropertyValue("--button-large-text-size");

        const res = [];
        for (const div_id of data_ids_sorted) {
            if (!keyExistsLogError(this.data_obj, div_id)) {
                continue;
            }
            const html_str = data[div_id];
            const parsed_html = isElement(html_str) ? html_str : htmlStringToElement(html_str);
            const depth = Number(parsed_html.dataset.depth);
            parsed_html.style.paddingLeft = `calc(${depth} * ${text_size})`;
            parsed_html.style.boxShadow = this.getBoxShadow(depth);
            // Roots come expanded by default. Need to delete if it exists.
            delete parsed_html.dataset.expanded;
            if (this.data_obj[div_id].expanded) {
                parsed_html.dataset.expanded = "";
            }

            // Only allow one item to have `data-selected`.
            delete parsed_html.dataset.selected;
            if (div_id === this.selected_div_id) {
                parsed_html.dataset.selected = "";
            }

            res.push(parsed_html.outerHTML);
            this.lru.set(String(div_id), parsed_html);
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
            // Wrap the paths in File objects to allow for proper sorting of filepaths.
            const a_file = new File([""], data[a].sort_path);
            const b_file = new File([""], data[b].sort_path);
            return a_file - b_file;
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
                search_only: bool,
                sort_<mode>: string, (for various sort modes)
            }
        */
        this.data_obj = await this.options.callbacks.initData();
    }

    updateCard(elem, new_html) {
        const parsed_html = htmlStringToElement(new_html);

        // replace the element in DOM with our new element
        elem.replaceWith(parsed_html);

        // update the internal cache with the new html
        this.lru.set(String(elem.dataset.divId), new_html);
    }

    async fetchData(idx_start, idx_end) {
        if (!this.enabled) {
            return [];
        }

        const data = await this.fetchDivIds(this.idxRangeToDivIds(idx_start, idx_end));
        const data_ids_sorted = Object.keys(data).sort((a, b) => {
            return this.data_obj_keys_sorted.indexOf(a) - this.data_obj_keys_sorted.indexOf(b);
        });

        const res = [];
        for (const div_id of data_ids_sorted) {
            res.push(data[div_id]);
            this.lru.set(div_id, data[div_id]);
        }

        return res;
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
        await super.sortData();
    }

    async filterDataDefaultCallback() {
        /** Filters data by a string and returns number of items after filter. */
        let n_visible = 0;
        for (const [div_id, v] of Object.entries(this.data_obj)) {
            let visible = true;
            // Filtering as directory only shows direct children. Case sensitive
            // comparison against the relative directory of each object.
            if (this.directory_filter_str && this.directory_filter_str !== v.rel_parent_dir) {
                this.data_obj[div_id].visible = false;
                continue;
            }

            if (v.search_only && this.filter_str.length >= 4) {
                // Custom filter for items marked search_only=true.
                // TODO: Not ideal. This disregards any search_terms set on the model.
                // However the search terms are currently set up in a way that would
                // reveal hidden models if the user searches for any visible parent
                // directories. For example, searching for "Lora" would reveal a hidden
                // model in "Lora/.hidden/model.safetensors" since that full path is
                // included in the search terms.
                visible = v.rel_parent_dir.toLowerCase().indexOf(this.filter_str.toLowerCase()) !== -1;
            } else {
                // All other filters treated case insensitive.
                visible = v.search_terms.toLowerCase().indexOf(this.filter_str.toLowerCase()) !== -1;
            }

            this.data_obj[div_id].visible = visible;
            if (visible) {
                n_visible++;
            }
        }
        return n_visible;
    }
}
