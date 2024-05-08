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
    initial_load = false;

    // Override base class defaults
    default_sort_mode_str = "divId";
    default_sort_dir_str = "ascending";
    default_filter_str = "";
    sort_mode_str = this.default_sort_mode_str;
    sort_dir_str = this.default_sort_dir_str;
    filter_str = this.default_filter_str;
    directory_filters = {};

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
        const max_items = Object.keys(this.data_obj).filter(k => this.data_obj[k].visible).length;
        await this.setMaxItems(max_items);
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
        this.initial_load = false;
        this.data_obj = {};
        this.data_obj_keys_sorted = [];
        if (this.lru instanceof LRUCache) {
            this.lru.destroy();
            this.lru = null;
        }
        super.destroy();
    }

    clear() {
        this.initial_load = false;
        this.data_obj = {};
        this.data_obj_keys_sorted = [];
        if (this.lru instanceof LRUCache) {
            this.lru.clear();
        }
        super.clear();
    }

    async load(force_init_data) {
        if (!this.enabled) {
            return;
        }

        this.initial_load = true;
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

    setDirectoryFilters(filters) {
        if (isNullOrUndefined(filters)) {
            this.directory_filters = {};
            return;
        }
        this.directory_filters = JSON.parse(JSON.stringify(filters));
    }

    addDirectoryFilter(div_id, filter_str, recurse) {
        this.directory_filters[div_id] = {filter_str: filter_str, recurse: recurse};
    }

    removeDirectoryFilter(div_id) {
        delete this.directory_filters[div_id];
    }

    clearDirectoryFilters({excluded_div_ids} = {}) {
        if (isString(excluded_div_ids)) {
            excluded_div_ids = [excluded_div_ids];
        }

        if (!Array.isArray(excluded_div_ids)) {
            excluded_div_ids = [];
        }

        for (const div_id of Object.keys(this.directory_filters)) {
            if (excluded_div_ids.includes(div_id)) {
                continue;
            }
            delete this.directory_filters[div_id];
        }
    }

    getDirectoryFilters() {
        return this.directory_filters;
    }

    async initDataDefaultCallback() {
        throw new NotImplementedError();
    }

    idxRangeToDivIds(idx_start, idx_end) {
        return this.data_obj_keys_sorted.slice(idx_start, idx_end);
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
        // we want to apply the sort to the visible items only.
        const filtered = Object.fromEntries(
            Object.entries(this.data_obj).filter(([k, v]) => v.visible)
        );
        this.data_obj_keys_sorted = this.sort_fn(filtered);
        if (this.sort_reverse) {
            this.data_obj_keys_sorted = this.data_obj_keys_sorted.reverse();
        }
    }

    async filterDataDefaultCallback() {
        throw new NotImplementedError();
    }

    updateHtml(elem, new_html) {
        const existing = this.lru.get(String(elem.dataset.divId));
        if (new_html) {
            if (existing === new_html) {
                return;
            }
            const parsed_html = htmlStringToElement(new_html);

            // replace the element in DOM with our new element
            elem.replaceWith(parsed_html);

            // update the internal cache with the new html
            this.lru.set(String(elem.dataset.divId), new_html);
        } else {
            if (existing === elem.outerHTML) {
                return;
            }
            this.lru.set(String(elem.dataset.divId), elem.outerHTML);
        }
    }
}

class ExtraNetworksClusterizeTreeList extends ExtraNetworksClusterize {
    prev_selected_div_id = null;

    constructor(args) {
        super({...args});
        this.selected_div_ids = new Set();
    }

    clear() {
        this.prev_selected_div_id = null;
        this.selected_div_ids.clear();
        super.clear();
    }

    setRowSelected(elem) {
        if (!isElement(elem)) {
            return;
        }

        this.updateHtml(elem);
        this.selected_div_ids.add(elem.dataset.divId);
        this.prev_selected_div_id = elem.dataset.divId;
    }

    setRowDeselected(elem) {
        if (!isElement(elem)) {
            return;
        }

        this.updateHtml(elem);
        this.selected_div_ids.delete(elem.dataset.divId);
        this.prev_selected_div_id = null;
    }

    clearSelectedRows({excluded_div_ids} = {}) {
        if (isString(excluded_div_ids)) {
            excluded_div_ids = [excluded_div_ids];
        }

        if (!Array.isArray(excluded_div_ids)) {
            excluded_div_ids = [];
        }

        this.selected_div_ids.clear();
        for (const div_id of excluded_div_ids) {
            this.selected_div_ids.add(div_id);
        }
        if (!excluded_div_ids.includes(this.prev_selected_div_id)) {
            this.prev_selected_div_id = null;
        }
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

    async expandAllRows(div_id) {
        /** Recursively expands all directories below the passed div_id. */
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

    async collapseAllRows(div_id) {
        /** Recursively collapses all directories below the passed div_id. */
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

        // Deselect current selected div id if it was just hidden.
        if (this.selected_div_ids.has(div_id) && !this.data_obj[div_id].visible) {
            this.selected_div_ids.delete(div_id);
            if (this.prev_selected_div_id === div_id) {
                this.prev_selected_div_id = null;
            }
        }

        const new_len = Object.values(this.data_obj).filter(v => v.visible).length;
        await this.setMaxItems(new_len);
        await this.refresh(true);
        await this.sortData();
    }

    getChildrenDivIds(div_id, {recurse} = {}) {
        const res = JSON.parse(JSON.stringify(this.data_obj[div_id].children));
        if (recurse === true) {
            for (const child_id of this.data_obj[div_id].children) {
                res.push(...this.getChildrenDivIds(child_id, {recurse: recurse}));
            }
        }
        return res;
    }

    async toggleRowExpanded(div_id) {
        /** Toggles a row between expanded and collapses states. */
        if (!keyExistsLogError(this.data_obj, div_id)) {
            return;
        }

        // Toggle state
        this.data_obj[div_id].expanded = !this.data_obj[div_id].expanded;

        const _set_visibility = (parent_id, visible) => {
            const this_obj = this.data_obj[parent_id];
            this_obj.visible = visible;
            for (const child_id of this_obj.children) {
                _set_visibility(child_id, visible && this_obj.expanded);
            }
        };

        for (const child_id of this.data_obj[div_id].children) {
            _set_visibility(child_id, this.data_obj[div_id].expanded);
        }

        // Deselect current selected div id if it was just hidden.
        if (this.selected_div_ids.has(div_id) && !this.data_obj[div_id].visible) {
            this.selected_div_ids.delete(div_id);
            if (this.prev_selected_div_id === div_id) {
                this.prev_selected_div_id = null;
            }
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

        const res = [];
        for (const div_id of data_ids_sorted) {
            if (!keyExistsLogError(this.data_obj, div_id)) {
                continue;
            }
            const html_str = data[div_id];
            const elem = isElement(html_str) ? html_str : htmlStringToElement(html_str);

            // Roots come expanded by default. Need to delete if it exists.
            delete elem.dataset.expanded;
            if (this.data_obj[div_id].expanded) {
                elem.dataset.expanded = "";
            }

            delete elem.dataset.selected;
            if (this.selected_div_ids.has(div_id)) {
                elem.dataset.selected = "";
            }

            this.lru.set(String(div_id), elem.outerHTML);
            res.push(elem.outerHTML);
        }

        return res;
    }

    async filterDataDefaultCallback() {
        // just return the number of visible objects in our data.
        return Object.values(this.data_obj).filter(v => v.visible).length;
    }
}

class ExtraNetworksClusterizeCardList extends ExtraNetworksClusterize {
    constructor(args) {
        super({...args});
    }

    sortByPath(data) {
        return Object.keys(data).sort((a, b) => {
            return INT_COLLATOR.compare(data[a].sort_path, data[b].sort_path);
        });
    }

    sortByName(data) {
        return Object.keys(data).sort((a, b) => {
            return INT_COLLATOR.compare(data[a].sort_name, data[b].sort_name);
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

            // Apply the directory filters.
            if (!Object.keys(this.directory_filters).length) {
                v.visible = true;
            } else {
                v.visible = Object.values(this.directory_filters).some((filter) => {
                    if (filter.recurse) {
                        return v.rel_parent_dir.startsWith(filter.filter_str);
                    } else {
                        return v.rel_parent_dir === filter.filter_str;
                    }
                });
            }
            if (!v.visible) {
                continue;
            }

            // Narrow the filtered items based on the search string.
            // Custom filter for items marked search_only=true.
            if (v.search_only) {
                if (Object.keys(this.directory_filters).length || this.filter_str.length >= 4) {
                    visible = v.search_terms.toLowerCase().indexOf(this.filter_str.toLowerCase()) !== -1;
                } else {
                    visible = false;
                }
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
