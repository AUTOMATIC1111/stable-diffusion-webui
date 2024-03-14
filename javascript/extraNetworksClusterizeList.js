// Collators used for sorting.
const INT_COLLATOR = new Intl.Collator([], {numeric: true});
const STR_COLLATOR = new Intl.Collator("en", {numeric: true, sensitivity: "base"});

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

const parseHtml = function(str) {
    const tmp = document.implementation.createHTMLDocument('');
    tmp.body.innerHTML = str;
    return [...tmp.body.childNodes];
}

const getComputedValue = function(container, css_property) {
    return parseInt(
        window.getComputedStyle(container, null)
            .getPropertyValue(css_property)
            .split("px")[0]
    );
};

const calcColsPerRow = function(parent) {
    // Returns the number of columns in a row of a flexbox.
    //const parent = document.querySelector(selector);
    const parent_width = getComputedValue(parent, "width");
    const parent_padding_left = getComputedValue(parent,"padding-left");
    const parent_padding_right = getComputedValue(parent,"padding-right");

    const child = parent.firstElementChild;
    const child_width = getComputedValue(child,"width");
    const child_margin_left = getComputedValue(child,"margin-left");
    const child_margin_right = getComputedValue(child,"margin-right");

    var parent_width_no_padding = parent_width - parent_padding_left - parent_padding_right;
    const child_width_with_margin = child_width + child_margin_left + child_margin_right;
    parent_width_no_padding += child_margin_left + child_margin_right;

    return parseInt(parent_width_no_padding / child_width_with_margin);
}

const calcRowsPerCol = function(container, parent) {
    // Returns the number of columns in a row of a flexbox.
    //const parent = document.querySelector(selector);
    const parent_height = getComputedValue(container, "height");
    const parent_padding_top = getComputedValue(container,"padding-top");
    const parent_padding_bottom = getComputedValue(container,"padding-bottom");

    const child = parent.firstElementChild;
    const child_height = getComputedValue(child,"height");
    const child_margin_top = getComputedValue(child,"margin-top");
    const child_margin_bottom = getComputedValue(child,"margin-bottom");

    var parent_height_no_padding = parent_height - parent_padding_top - parent_padding_bottom;
    const child_height_with_margin = child_height + child_margin_top + child_margin_bottom;
    parent_height_no_padding += child_margin_top + child_margin_bottom;

    return parseInt(parent_height_no_padding / child_height_with_margin);
}

class ExtraNetworksClusterize {
    constructor(
        {
            scroll_id,
            content_id,
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
        console.log(`scroll_id: ${scroll_id}, content_id: ${content_id}`);
        if (scroll_id === undefined) {
            console.error("scroll_id is undefined!");
        }
        if (content_id === undefined) {
            console.error("content_id is undefined!");
        }

        this.scroll_id = scroll_id;
        this.content_id = content_id;
        this.rows_in_block = rows_in_block;
        this.blocks_in_cluster = blocks_in_cluster;
        this.show_no_data_row = show_no_data_row;
        this.callbacks = callbacks;

        this.enabled = false;

        this.encoded_str = "";

        this.no_data_text = "Directory is empty.";
        this.no_data_class = "nocards";

        this.scroll_elem = document.getElementById(this.scroll_id);
        this.content_elem = document.getElementById(this.content_id);

        this.n_rows = 1;
        this.n_cols = 1;

        this.sort_fn = this.sortByDivId;
        this.sort_reverse = false;

        this.data_obj = {};
        this.data_obj_keys_sorted = [];

        this.clusterize = new Clusterize(
            {
                rows: [],
                scrollId: this.scroll_id,
                contentId: this.content_id,
                rows_in_block: this.rows_in_block,
                blocks_in_cluster: this.blocks_in_cluster,
                show_no_data_row: this.show_no_data_row,
                callbacks: this.callbacks,
            }
        );
    }

    enable() {
        this.enabled = true;
    }

    disable() {
        this.enabled = false;
    }

    parseJson(encoded_str) {
        if (this.encoded_str === encoded_str) {
            return;
        }

        Promise.resolve(encoded_str)
            .then(v => decompress(v))
            .then(v => JSON.parse(v))
            .then(v => this.updateJson(v))
            .then(() => this.encoded_str = encoded_str);
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
        this.updateRows();
    }

    applyFilter() {
        // the base class filter just sorts the values and updates the rows.
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
        this.clusterize.update(this.filterRows(this.data_obj));
        this.clusterize.refresh();
    }

    nrows() {
        return this.clusterize.getRowsAmount();
    }

    updateItemDims() {
        if (!this.enabled) {
            // Inactive list is not displayed on screen. Would error if trying to resize.
            return;
        }
        if (this.nrows() <= 0) {
            // If there is no data then just skip.
            return;
        }
        // Calculate the visible rows and colums for the clusterize-content area.
        let n_cols = calcColsPerRow(this.content_elem);
        let n_rows = calcRowsPerCol(this.content_elem.parentElement, this.content_elem);

        n_cols = isNaN(n_cols) || n_cols <= 0 ? 1 : n_cols;
        n_rows = isNaN(n_rows) || n_rows <= 0 ? 1 : n_rows;

        if (n_cols != this.n_cols || n_rows != this.n_rows) {
            // Sizes have changed. Update the instance values.
            this.n_cols = n_cols;
            this.n_rows = n_rows;
            this.rows_in_block = this.n_rows;
        }
    }

    rebuild() {
        this.clusterize.destroy();

        // Get new references to elements since they may have changed.
        this.scroll_elem = document.getElementById(this.scroll_id);
        this.content_elem = document.getElementById(this.content_id);
        this.clusterize = new Clusterize(
            {
                rows: this.filterRows(this.data_obj),
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

        // Apply existing sort mode.
        this.applyFilter();

        this.updateItemDims();
        this.updateRows();
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
            res += (i+1 > depth) ? "" : ", ";
        }
        return res;
    }

    updateJson(json) {
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

        this.applyFilter();
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
        for (const [k, v] of Object.entries(json)) {
            let div_id = k;
            let parsed_html = parseHtml(v)[0];
            // Add the updated html to the data object.
            this.data_obj[div_id] = {
                element: parsed_html,
                active: true,
            };
        }

        this.applyFilter();
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

        switch(this.sort_mode_str) {
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
        if (filter_str !== undefined) {
            this.filter_str = filter_str.toLowerCase();
        }

        for (const [k, v] of Object.entries(this.data_obj)) {
            let search_only = v.element.querySelector(".search_only");
            let text = Array.prototype.map.call(v.element.querySelectorAll(".search_terms"), function(t) {
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