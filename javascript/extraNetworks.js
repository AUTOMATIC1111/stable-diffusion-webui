// Prevent eslint errors on functions defined in other files.
/*global
    selectCheckpoint,
    ExtraNetworksClusterizeTreeList,
    ExtraNetworksClusterizeCardsList,
    waitForElement,
    isString,
    isElement,
    isElementThrowError,
    fetchWithRetryAndBackoff,
    isElementLogError,
    isNumber,
    waitForKeyInObject,
    isNullOrUndefined,
    debounce,
    waitForBool,
    copyToClipboard,
*/
/*eslint no-undef: "error"*/

const SEARCH_INPUT_DEBOUNCE_TIME_MS = 250;
const EXTRA_NETWORKS_REFRESH_INTERNAL_DEBOUNCE_TIMEOUT_MS = 200;
const EXTRA_NETWORKS_WAIT_FOR_PAGE_READY_TIMEOUT_MS = 60000;
const EXTRA_NETWORKS_INIT_DATA_TIMEOUT_MS = 60000;
const EXTRA_NETWORKS_FETCH_DATA_TIMEOUT_MS = 60000;

const re_extranet = /<([^:^>]+:[^:]+):[\d.]+>(.*)/;
const re_extranet_g = /<([^:^>]+:[^:]+):[\d.]+>/g;
const re_extranet_neg = /\(([^:^>]+:[\d.]+)\)/;
const re_extranet_g_neg = /\(([^:^>]+:[\d.]+)\)/g;
var globalPopup = null;
var globalPopupInner = null;
const storedPopupIds = {};
const extraPageUserMetadataEditors = {};
const extra_networks_tabs = {};
var extra_networks_refresh_internal_debounce_timer;

/** Boolean flags used along with utils.js::waitForBool(). */
// Set true when we first load the UI options.
const initialUiOptionsLoaded = {state: false};

const _debounce = (handler, timeout_ms) => {
    /** Debounces a function call.
     *
     *  NOTE: This will NOT work if called from within a class.
     *  It will drop `this` from scope.
     *
     *  Repeated calls to the debounce handler will not call the handler until there are
     *  no new calls to the debounce handler for timeout_ms time.
     *
     *  Example:
     *  function add(x, y) { return x + y; }
     *  let debounce_handler = debounce(add, 5000);
     *  let res;
     *  for (let i = 0; i < 10; i++) {
     *      res = debounce_handler(i, 100);
     *  }
     *  console.log("Result:", res);
     *
     *  This example will print "Result: 109".
     */
    let timer = null;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => handler(...args), timeout_ms);
    };
};

class ExtraNetworksError extends Error {
    constructor(...args) {
        super(...args);
        this.name = this.constructor.name;
    }
}
class ExtraNetworksPageReadyError extends Error {
    constructor(...args) {
        super(...args);
    }
}

class ExtraNetworksDataReadyError extends Error {
    constructor(...args) {
        super(...args);
    }
}

class ExtraNetworksTab {
    tabname;
    extra_networks_tabname;
    tabname_full; // {tabname}_{extra_networks_tabname}
    tree_list;
    cards_list;
    container_elem;
    controls_elem;
    txt_search_elem;
    prompt_container_elem;
    prompts_elem;
    prompt_row_elem;
    neg_prompt_row_elem;
    txt_prompt_elem;
    txt_neg_prompt_elem;
    active_prompt_elem;
    sort_mode_str = "";
    sort_dir_str = "";
    filter_str = "";
    directory_filter_str = "";
    directory_filter_recurse = false;
    show_prompt = true;
    show_neg_prompt = true;
    compact_prompt_en = false;
    refresh_in_progress = false;
    dirs_view_en = false;
    tree_view_en = false;
    constructor({tabname, extra_networks_tabname}) {
        this.tabname = tabname;
        this.extra_networks_tabname = extra_networks_tabname;
        this.tabname_full = `${tabname}_${extra_networks_tabname}`;
    }

    async setup(pane, controls_div) {
        this.container_elem = pane;

        // get page elements
        await Promise.all([
            waitForElement(`#${this.tabname_full}_pane .extra-network-controls`).then(elem => this.controls_elem = elem),
            waitForElement(`#${this.tabname}_prompt_container`).then(elem => this.prompt_container_elem = elem),
            waitForElement(`#${this.tabname_full}_prompts`).then(elem => this.prompts_elem = elem),
            waitForElement(`#${this.tabname}_prompt_row`).then(elem => this.prompt_row_elem = elem),
            waitForElement(`#${this.tabname}_neg_prompt_row`).then(elem => this.neg_prompt_row_elem = elem),
            waitForElement(`#${this.tabname_full}_tree_list_scroll_area`),
            waitForElement(`#${this.tabname_full}_tree_list_content_area`),
            waitForElement(`#${this.tabname_full}_cards_list_scroll_area`),
            waitForElement(`#${this.tabname_full}_cards_list_content_area`),
        ]);

        this.txt_search_elem = this.controls_elem.querySelector(".extra-network-control--search-text");
        this.dirs_view_en = "selected" in this.controls_elem.querySelector(
            ".extra-network-control--dirs-view"
        ).dataset;
        this.tree_view_en = "selected" in this.controls_elem.querySelector(
            ".extra-network-control--tree-view"
        ).dataset;

        // determine whether compact prompt mode is enabled.
        // cannot await this since it may not exist on page depending on user setting.
        this.compact_prompt_en = isElement(gradioApp().querySelector(".toprow-compact-tools"));

        // setup this tab's controls
        this.controls_elem.id = `${this.tabname_full}_controls`;
        controls_div.insertBefore(this.controls_elem, null);

        await Promise.all([this.setupTreeList(), this.setupCardsList()]);

        const sort_mode_elem = this.controls_elem.querySelector(".extra-network-control--sort-mode[data-selected]");
        isElementThrowError(sort_mode_elem);
        const sort_dir_elem = this.controls_elem.querySelector(".extra-network-control--sort-dir");
        isElementThrowError(sort_dir_elem);

        this.setSortMode(sort_mode_elem.dataset.sortMode);
        this.setSortDir(sort_dir_elem.dataset.sortDir);
        this.applyDirectoryFilter();
        this.applyFilter();

        this.registerPrompt();

        if (this.container_elem.style.display === "none") {
            this.hideControls();
        } else {
            this.showControls();
        }
    }

    destroy() {
        this.unload();
        this.tree_list.destroy();
        this.cards_list.destroy();
        this.tree_list = null;
        this.cards_list = null;
        this.container_elem = null;
        this.controls_elem = null;
        this.txt_search_elem = null;
        this.prompt_container_elem = null;
        this.prompts_elem = null;
        this.prompt_row_elem = null;
        this.neg_prompt_row_elem = null;
        this.txt_prompt_elem = null;
        this.txt_neg_prompt_elem = null;
        this.active_prompt_elem = null;
        this.refresh_in_progress = false;
        this.tree_view_en = false;
        this.dirs_view_en = false;
    }

    async registerPrompt() {
        await Promise.all([
            waitForElement(`#${this.tabname}_prompt > label > textarea`).then(elem => this.txt_prompt_elem = elem),
            waitForElement(`#${this.tabname}_neg_prompt > label > textarea`).then(elem => this.txt_neg_prompt_elem = elem),
        ]);
        this.active_prompt_elem = this.txt_prompt_elem;
        this.txt_prompt_elem.addEventListener("focus", () => this.active_prompt_elem = this.txt_prompt_elem);
        this.txt_neg_prompt_elem.addEventListener("focus", () => this.active_prompt_elem = this.txt_neg_prompt_elem);
    }

    async setupTreeList() {
        if (this.tree_list instanceof ExtraNetworksClusterizeTreeList) {
            this.tree_list.destroy();
        }
        this.tree_list = new ExtraNetworksClusterizeTreeList({
            tabname: this.tabname,
            extra_networks_tabname: this.extra_networks_tabname,
            scrollId: `${this.tabname_full}_tree_list_scroll_area`,
            contentId: `${this.tabname_full}_tree_list_content_area`,
            tag: "button",
            callbacks: {
                initData: this.onInitTreeData.bind(this),
                fetchData: this.onFetchTreeData.bind(this),
            },
        });
        await this.tree_list.setup();
    }

    async setupCardsList() {
        if (this.cards_list instanceof ExtraNetworksClusterizeCardsList) {
            this.cards_list.destroy();
        }
        this.cards_list = new ExtraNetworksClusterizeCardsList({
            tabname: this.tabname,
            extra_networks_tabname: this.extra_networks_tabname,
            scrollId: `${this.tabname_full}_cards_list_scroll_area`,
            contentId: `${this.tabname_full}_cards_list_content_area`,
            tag: "div",
            callbacks: {
                initData: this.onInitCardsData.bind(this),
                fetchData: this.onFetchCardsData.bind(this),
            },
        });
        await this.cards_list.setup();
    }

    setSortMode(sort_mode_str) {
        this.sort_mode_str = sort_mode_str;
        this.cards_list.setSortMode(this.sort_mode_str);
    }

    setSortDir(sort_dir_str) {
        this.sort_dir_str = sort_dir_str;
        this.cards_list.setSortDir(this.sort_dir_str);
    }

    setFilterStr(filter_str) {
        this.filter_str = filter_str;
        this.cards_list.setFilterStr(this.filter_str);
    }

    setDirectoryFilterStr(filter_str, recurse) {
        this.directory_filter_str = filter_str;
        this.directory_filter_recurse = recurse;
        this.cards_list.setDirectoryFilterStr(this.directory_filter_str, this.directory_filter_recurse);
    }

    movePrompt(show_prompt = true, show_neg_prompt = true) {
        // This function only applies when compact prompt mode is enabled.
        if (!this.compact_prompt_en) {
            return;
        }

        if (show_neg_prompt) {
            this.prompts_elem.insertBefore(this.neg_prompt_row_elem, this.prompts_elem.firstChild);
        }

        if (show_prompt) {
            this.prompts_elem.insertBefore(this.prompt_row_elem, this.prompts_elem.firstChild);
        }

        this.prompts_elem.classList.toggle("extra-page-prompts-active", show_neg_prompt || show_prompt);
    }

    refreshSingleCard(elem) {
        requestGet(
            "./sd_extra_networks/get-single-card",
            {
                tabname: this.tabname,
                extra_networks_tabname: this.extra_networks_tabname,
                name: elem.dataset.name,
                div_id: elem.dataset.divId,
            },
            (data) => {
                if (data && data.html) {
                    this.cards_list.updateHtml(elem, data.html);
                }
            },
        );
    }

    showControls() {
        this.controls_elem.classList.remove("hidden");
    }

    hideControls() {
        this.controls_elem.classList.add("hidden");
    }

    async #refresh() {
        try {
            await this.waitForServerPageReady();
        } catch (error) {
            console.error(`refresh error: ${error.message}`);
            return;
        }
        const btn_dirs_view = this.controls_elem.querySelector(".extra-network-control--dirs-view");
        const btn_tree_view = this.controls_elem.querySelector(".extra-network-control--tree-view");
        const div_dirs = this.container_elem.querySelector(".extra-network-content--dirs-view");
        // We actually want to select the tree view's column in the resize-handle-row.
        // This is what we actually show/hide, not the inner elements.
        const div_tree = this.tree_list.scroll_elem.closest(".resize-handle-col");

        this.dirs_view_en = "selected" in btn_dirs_view.dataset;
        this.tree_view_en = "selected" in btn_tree_view.dataset;
        // Remove "hidden" class if button is enabled, otherwise add it.
        div_dirs.classList.toggle("hidden", !this.dirs_view_en);
        div_tree.classList.toggle("hidden", !this.tree_view_en);



        // Apply the current resize handle classes.
        const resize_handle_row = this.tree_list.scroll_elem.closest(".resize-handle-row");
        resize_handle_row.classList.toggle("resize-handle-hidden", div_tree.classList.contains("hidden"));

        await Promise.all([this.setupTreeList(), this.setupCardsList()]);
        this.tree_list.enable(true);
        this.cards_list.enable(true);
        await Promise.all([this.tree_list.load(true), this.cards_list.load(true)]);
        // apply the previous sort/filter options
        this.setSortMode(this.sort_mode_str);
        this.setSortDir(this.sort_dir_str);
        this.applyDirectoryFilter(this.directory_filter_str, this.directory_filter_recurse);
        this.applyFilter(this.filter_str);
    }

    async refresh() {
        if (this.refresh_in_progress) {
            return;
        }
        this.refresh_in_progress = true;
        await this.#refresh();
        this.refresh_in_progress = false;
    }

    async load(show_prompt, show_neg_prompt) {
        this.movePrompt(show_prompt, show_neg_prompt);
        this.showControls();
        this.tree_list.enable(true);
        this.cards_list.enable(true);
        await Promise.all([this.tree_list.load(), this.cards_list.load()]);
    }

    unload() {
        this.movePrompt(false, false);
        this.hideControls();
        this.tree_list.enable(false);
        this.cards_list.enable(false);
    }

    applyDirectoryFilter(filter_str, recurse) {
        filter_str = isString(filter_str) ? filter_str : "";
        recurse = recurse === true || recurse === false ? recurse : false;
        this.setDirectoryFilterStr(filter_str, recurse);
    }

    applyFilter(filter_str) {
        filter_str = !isString(filter_str) ? this.txt_search_elem.value : filter_str;
        this.setFilterStr(filter_str);
    }

    async waitForServerPageReady(timeout_ms) {
        /** Waits for a page on the server to be ready.
         *
         *  We need to wait for the page to be ready before we can fetch any data.
         *  It is possible to click on a tab before the server has any data ready for us.
         *  Since clicking on tabs triggers a data request, there will be an error from
         *  the server since the data isn't ready. This function allows us to wait for
         *  the server to tell us that it is ready for data requests.
         *
         *  Args:
         *      max_attempts [int]: The max number of requests that will be attempted
         *                          before giving up. If set to 0, will attempt forever.
         */
        timeout_ms = timeout_ms || EXTRA_NETWORKS_WAIT_FOR_PAGE_READY_TIMEOUT_MS;
        const response_handler = (response) => new Promise((resolve, reject) => {
            if (!response.ok) {
                return reject(response);
            }

            response.json().then(json => {
                if (!json.ready) {
                    return reject(`page not ready: ${this.extra_networks_tabname}`);
                }
                return resolve(json);
            });
        });

        const url = "./sd_extra_networks/page-is-ready";
        const payload = {extra_networks_tabname: this.extra_networks_tabname};
        const opts = {timeout_ms: timeout_ms, response_handler: response_handler};
        return await fetchWithRetryAndBackoff(url, payload, opts);
    }

    async onInitCardsData() {
        try {
            await this.waitForServerPageReady();
        } catch (error) {
            console.error(`onInitCardsData error: ${error.message}`);
            return {};
        }

        const response_handler = (response) => new Promise((resolve, reject) => {
            if (!response.ok) {
                return reject(response);
            }
            response.json().then(json => {
                if (!json.ready) {
                    return reject(`data not ready: ${this.extra_networks_tabname}`);
                }
                return resolve(json);
            });
        });

        const url = "./sd_extra_networks/init-cards-data";
        const payload = {tabname: this.tabname, extra_networks_tabname: this.extra_networks_tabname};
        const timeout_ms = EXTRA_NETWORKS_INIT_DATA_TIMEOUT_MS;
        const opts = {timeout_ms: timeout_ms, response_handler: response_handler};
        try {
            const response = await fetchWithRetryAndBackoff(url, payload, opts);
            return response.data;
        } catch (error) {
            console.error(`onInitCardsData error: ${error.message}`);
            return {};
        }
    }

    async onInitTreeData() {
        try {
            await this.waitForServerPageReady();
        } catch (error) {
            console.error(`onInitTreeData error: ${error.message}`);
            return {};
        }

        const response_handler = (response) => new Promise((resolve, reject) => {
            if (!response.ok) {
                return reject(response);
            }
            response.json().then(json => {
                if (!json.ready) {
                    return reject(`data not ready: ${this.extra_networks_tabname}`);
                }
                return resolve(json);
            });
        });

        const url = "./sd_extra_networks/init-tree-data";
        const payload = {tabname: this.tabname, extra_networks_tabname: this.extra_networks_tabname};
        const timeout_ms = EXTRA_NETWORKS_INIT_DATA_TIMEOUT_MS;
        const opts = {timeout_ms: timeout_ms, response_handler: response_handler};
        try {
            const response = await fetchWithRetryAndBackoff(url, payload, opts);
            return response.data;
        } catch (error) {
            console.error(`onInitTreeData error: ${error.message}`);
            return {};
        }
    }

    async onFetchCardsData(div_ids) {
        const url = "./sd_extra_networks/fetch-cards-data";
        const payload = {extra_networks_tabname: this.extra_networks_tabname, div_ids: div_ids};
        const timeout_ms = EXTRA_NETWORKS_FETCH_DATA_TIMEOUT_MS;
        const opts = {timeout_ms: timeout_ms};
        try {
            const response = await fetchWithRetryAndBackoff(url, payload, opts);
            if (response.missing_div_ids.length) {
                console.warn(`Failed to fetch multiple div_ids: ${response.missing_div_ids}`);
            }
            return response.data;
        } catch (error) {
            console.error(`onFetchCardsData error: ${error.message}`);
            return {};
        }
    }

    async onFetchTreeData(div_ids) {
        const url = "./sd_extra_networks/fetch-tree-data";
        const payload = {extra_networks_tabname: this.extra_networks_tabname, div_ids: div_ids};
        const timeout_ms = EXTRA_NETWORKS_FETCH_DATA_TIMEOUT_MS;
        const opts = {timeout_ms: timeout_ms};
        try {
            const response = await fetchWithRetryAndBackoff(url, payload, opts);
            if (response.missing_div_ids.length) {
                console.warn(`Failed to fetch multiple div_ids: ${response.missing_div_ids}`);
            }
            return response.data;
        } catch (error) {
            console.error(`onFetchTreeData error: ${error.message}`);
            return {};
        }
    }

    updateSearch(text) {
        this.txt_search_elem.value = text;
        updateInput(this.txt_search_elem);
        this.applyFilter(this.txt_search_elem.value);
    }

    autoSetTreeWidth() {
        const row = this.container_elem.querySelector(".resize-handle-row");
        if (!isElementLogError(row)) {
            return;
        }

        const left_col = row.firstElementChild;
        if (!isElementLogError(left_col)) {
            return;
        }

        // If the left column is hidden then we don't want to do anything.
        if (left_col.classList.contains("hidden")) {
            return;
        }

        const pad = parseFloat(row.style.gridTemplateColumns.split(" ")[1]);
        const min_left_col_width = parseFloat(left_col.style.flexBasis.slice(0, -2));
        // We know that the tree list is the left column. That is the only one we want to resize.
        let max_width = this.tree_list.getMaxRowWidth();
        if (!isNumber(max_width)) {
            return;
        }
        // Add the resize handle's padding to the result and default to minLeftColWidth if necessary.
        max_width = Math.max(max_width + pad, min_left_col_width);

        // Mimicks resizeHandle.js::setLeftColGridTemplate().
        row.style.gridTemplateColumns = `${max_width}px ${pad}px 1fr`;
    }

    async setDirectoryButtons({source_elem, source_selector, source_class, reset_all} = {}) {
        // At least one argument must be specified.
        if (isNullOrUndefined(source_elem) &&
            isNullOrUndefined(source_selector) &&
            isNullOrUndefined(source_class) &&
            isNullOrUndefined(reset_all)) {
            console.error("At least one argument must be specified.");
            return;
        }

        // Checks if an element exists and is visible on the page.
        const _exists = (elem) => {
            return isElement(elem) && !isNullOrUndefined(elem.offsetParent);
        };

        // source_elem is specified but invalid.
        if (!isNullOrUndefined(source_elem) && !_exists(source_elem)) {
            return;
        }

        // Removes `data-selected` attribute from all tree/dirs buttons.
        const _reset_all_buttons = async({excluded_elems} = {}) => {
            const elems = this.container_elem.querySelectorAll(
                ".extra-network-dirs-view-button, .tree-list-item"
            );
            for (const elem of elems) {
                if (Array.isArray(excluded_elems) && excluded_elems.includes(elem)) {
                    continue;
                }

                const prev = elem.outerHTML;
                delete elem.dataset.selected;
                delete elem.dataset.recurse;
                elem.classList.remove("short-pressed");
                elem.classList.remove("long-pressed");
                if (prev !== elem.outerHTML) {
                    this.tree_list.updateHtml(elem);
                }
            }

            this.tree_list.content_elem.querySelectorAll(
                ".tree-list-item-indent [data-selected]"
            ).forEach(elem => {
                delete elem.dataset.selected;
            });
        };
        _reset_all_buttons.bind(this);

        const _set_recursion_depth = (parent_id, state) => {
            this.tree_list.content_elem.querySelectorAll(
                `.tree-list-item-indent [data-parent-id="${parent_id}"]`
            ).forEach(elem => {
                elem.toggleAttribute("data-selected", state);
            });
        };
        _set_recursion_depth.bind(this);

        if (reset_all === true) {
            _reset_all_buttons();
            await this.tree_list.onRowSelected(); // no args deselects all.
            this.applyDirectoryFilter();
            return;
        }

        if (!_exists(source_elem) && isString(source_selector)) {
            source_elem = this.container_elem.querySelector(source_selector);
        }

        if (!_exists(source_elem) && isString(source_class)) {
            source_elem = this.container_elem.querySelector(`${source_class}[data-selected]`);
        }

        // try to find any selected buttons to use as a source.
        if (!_exists(source_elem)) {
            source_elem = this.container_elem.querySelector("[data-selected]");
        }

        // If we got here with no source elem, then we will take this to mean that
        // we are deselecting all.
        if (!_exists(source_elem)) {
            _reset_all_buttons();
            await this.tree_list.onRowSelected(); // no args deselects all.
            this.applyDirectoryFilter();
            return;
        }

        const source_is_tree = source_elem.classList.contains("tree-list-item");
        const data_path = String.raw`${source_elem.dataset.path.replaceAll("\\", "\\\\")}`;
        const other_selector = source_is_tree ? ".extra-network-dirs-view-button" : ".tree-list-item";
        const other_elem = document.querySelector(`${other_selector}[data-path="${data_path}"]`);
        if (!_exists(other_elem)) {
            // Can't reflect attributes since no matching element exists.
            // This can happen when tree/dirs view is disabled or tree is collapsed.
            _reset_all_buttons({excluded_elems: [source_elem]});
            if (source_is_tree) {
                await this.tree_list.onRowSelected(source_elem);
                _set_recursion_depth(source_elem.dataset.divId, "recurse" in source_elem.dataset);
            } else {
                await this.tree_list.onRowSelected();
            }
            this.applyDirectoryFilter(
                "selected" in source_elem.dataset ? data_path : null,
                "recurse" in source_elem.dataset,
            );
            return;
        }

        const data_selected = "selected" in source_elem.dataset;
        const data_recurse = "recurse" in source_elem.dataset;
        const short_pressed = source_elem.classList.contains("short-pressed");
        const long_pressed = source_elem.classList.contains("long-pressed");

        _reset_all_buttons({excluded_elems: [source_elem, other_elem]});
        other_elem.toggleAttribute("data-selected", data_selected);
        other_elem.toggleAttribute("data-recurse", data_recurse);
        other_elem.classList.toggle("short-pressed", short_pressed);
        other_elem.classList.toggle("long-pressed", long_pressed);

        await this.tree_list.onRowSelected(source_is_tree ? source_elem : other_elem);
        const div_id = source_is_tree ? source_elem.dataset.divId : other_elem.dataset.divId;
        _set_recursion_depth(div_id, data_recurse);
        this.applyDirectoryFilter(
            "selected" in source_elem.dataset ? data_path : null,
            "recurse" in source_elem.dataset,
        );
    }
}

function popup(contents) {
    if (!globalPopup) {
        globalPopup = document.createElement('div');
        globalPopup.classList.add('global-popup');

        var close = document.createElement('div');
        close.classList.add('global-popup-close');
        close.addEventListener("click", closePopup);
        close.title = "Close";
        globalPopup.appendChild(close);

        globalPopupInner = document.createElement('div');
        globalPopupInner.classList.add('global-popup-inner');
        globalPopup.appendChild(globalPopupInner);

        gradioApp().querySelector('.main').appendChild(globalPopup);
    }

    globalPopupInner.innerHTML = '';
    globalPopupInner.appendChild(contents);

    globalPopup.style.display = "flex";
}

function popupId(id) {
    if (!storedPopupIds[id]) {
        storedPopupIds[id] = gradioApp().getElementById(id);
    }

    popup(storedPopupIds[id]);
}

function closePopup() {
    if (!globalPopup) return;
    globalPopup.style.display = "none";
}


// ==== GENERAL EXTRA NETWORKS FUNCTIONS ====

function extraNetworksRemoveFromPrompt(textarea, text, is_neg) {
    let match = text.match(is_neg ? re_extranet_neg : re_extranet);
    let replaced = false;
    let res;
    let sep = opts.extra_networks_add_text_separator;

    if (match) {
        const content = match[1];
        const postfix = match[2];
        let idx = -1;
        res = textarea.value.replaceAll(
            is_neg ? re_extranet_g_neg : re_extranet_g,
            (found, net, pos) => {
                match = found.match(is_neg ? re_extranet_neg : re_extranet);
                if (match[1] === content) {
                    replaced = true;
                    idx = pos;
                    return "";
                }
                return found;
            },
        );
        if (idx >= 0) {
            if (postfix && res.slice(idx, idx + postfix.length) === postfix) {
                res = res.slice(0, idx) + res.slice(idx + postfix.length);
            }
            if (res.slice(idx - sep.length, idx) === sep) {
                res = res.slice(0, idx - sep.length) + res.slice(idx);
            }
            // Remove separator if it is at beginning of string.
            if (res.startsWith(sep)) {
                res = res.slice(sep.length);
            }
        }
    } else {
        res = textarea.value.replaceAll(new RegExp(`((?:${sep})?${text})`, "g"), "");
        replaced = (res !== textarea.value);
    }

    if (replaced) {
        textarea.value = res;
        return true;
    }

    return false;
}

function extraNetworksUpdatePrompt(textarea, text, is_neg) {
    if (!extraNetworksRemoveFromPrompt(textarea, text, is_neg)) {
        if (!textarea.value) {
            // if textarea is empty, dont add the separator.
            textarea.value = text;
        } else {
            textarea.value = textarea.value + opts.extra_networks_add_text_separator + text;
        }
    }

    updateInput(textarea);
}

function extraNetworksSaveCardPreview(event, tabname, filename) {
    const textarea = gradioApp().querySelector(`#${tabname}_preview_filename > label > textarea`);
    const button = gradioApp().getElementById(`${tabname}_save_preview`);

    textarea.value = filename;
    updateInput(textarea);

    button.click();

    event.stopPropagation();
    event.preventDefault();
}

function extraNetworksFlattenMetadata(obj) {
    const result = {};

    // Convert any stringified JSON objects to actual objects
    for (const key of Object.keys(obj)) {
        if (typeof obj[key] === 'string') {
            try {
                const parsed = JSON.parse(obj[key]);
                if (parsed && typeof parsed === 'object') {
                    obj[key] = parsed;
                }
            } catch (error) {
                continue;
            }
        }
    }

    // Flatten the object
    for (const key of Object.keys(obj)) {
        if (typeof obj[key] === 'object' && obj[key] !== null) {
            const nested = extraNetworksFlattenMetadata(obj[key]);
            for (const nestedKey of Object.keys(nested)) {
                result[`${key}/${nestedKey}`] = nested[nestedKey];
            }
        } else {
            result[key] = obj[key];
        }
    }

    // Special case for handling modelspec keys
    for (const key of Object.keys(result)) {
        if (key.startsWith("modelspec.")) {
            result[key.replaceAll(".", "/")] = result[key];
            delete result[key];
        }
    }

    // Add empty keys to designate hierarchy
    for (const key of Object.keys(result)) {
        const parts = key.split("/");
        for (let i = 1; i < parts.length; i++) {
            const parent = parts.slice(0, i).join("/");
            if (!result[parent]) {
                result[parent] = "";
            }
        }
    }

    return result;
}

function extraNetworksShowMetadata(text) {
    try {
        let parsed = JSON.parse(text);
        if (parsed && typeof parsed === 'object') {
            parsed = extraNetworksFlattenMetadata(parsed);
            const table = createVisualizationTable(parsed, 0);
            popup(table);
            return;
        }
    } catch (error) {
        console.error(error);
    }

    var elem = document.createElement('pre');
    elem.classList.add('popup-metadata');
    elem.textContent = text;

    popup(elem);
    return;
}

function extraNetworksRefreshSingleCard(tabname, extra_networks_tabname, name) {
    const tab = extra_networks_tabs[`${tabname}_${extra_networks_tabname}`];
    const elem = tab.cards_list.content_elem.querySelector(`.card[data-name="${name}"]`);
    isElementThrowError(elem);
    tab.refreshSingleCard(elem);
}

function extraNetworksRefreshTab(tabname_full) {
    /** called from python when user clicks the extra networks refresh tab button */
    extra_networks_tabs[tabname_full].refresh();
}

// ==== EVENT HANDLING ====

function extraNetworksFetchMetadata(extra_networks_tabname, card_name) {
    const _showError = () => {
        extraNetworksShowMetadata("there was an error getting metadata");
    };

    requestGet(
        "./sd_extra_networks/metadata",
        {extra_networks_tabname: extra_networks_tabname, item: card_name},
        function(data) {
            if (data && data.metadata) {
                extraNetworksShowMetadata(data.metadata);
            } else {
                _showError();
            }
        },
        _showError,
    );
}

function extraNetworksUnrelatedTabSelected(tabname) {
    /** called from python when user selects an unrelated tab (generate) */
    for (const v of Object.values(extra_networks_tabs)) {
        v.unload();
    }

    // Move all prompts into the selected tab.
    const prompt_container_elem = document.querySelector(`#${tabname}_prompt_container`);
    const prompt_row_elem = document.querySelector(`#${tabname}_prompt_row`);
    const neg_prompt_row_elem = document.querySelector(`#${tabname}_neg_prompt_row`);
    prompt_container_elem.insertBefore(neg_prompt_row_elem, prompt_container_elem.firstChild);
    prompt_container_elem.insertBefore(prompt_row_elem, prompt_container_elem.firstChild);
}

async function extraNetworksTabSelected(tabname_full, show_prompt, show_neg_prompt) {
    /** called from python when user selects an extra networks tab */
    await waitForKeyInObject({obj: extra_networks_tabs, k: tabname_full});
    for (const [k, v] of Object.entries(extra_networks_tabs)) {
        if (k === tabname_full) {
            v.load(show_prompt, show_neg_prompt);
        } else {
            v.unload();
        }
    }
}

function extraNetworksBtnDirsViewItemOnLongPress(event) {
    const btn = event.target.closest(".extra-network-dirs-view-button");
    const pane = btn.closest(".extra-network-pane");
    const tab = extra_networks_tabs[pane.dataset.tabnameFull];

    tab.setDirectoryButtons({source_elem: btn});
}

function extraNetworksBtnDirsViewItemOnClick(event) {
    /** Handles `onclick` events for buttons in the directory view. */
    const btn = event.target.closest(".extra-network-dirs-view-button");
    const pane = btn.closest(".extra-network-pane");
    const tab = extra_networks_tabs[pane.dataset.tabnameFull];

    tab.setDirectoryButtons({source_elem: btn});
}

function extraNetworksControlSearchClearOnClick(event) {
    /** Dispatches custom event when the `clear` button in a search input is clicked. */
    const btn = event.target.closest(".extra-network-control--search-clear");
    const controls = btn.closest(".extra-network-controls");
    extra_networks_tabs[controls.dataset.tabnameFull].updateSearch("");
}

function extraNetworksControlSortModeOnClick(event) {
    /** Handles `onclick` events for Sort Mode buttons. */
    const btn = event.target.closest(".extra-network-control--sort-mode");
    // No operation if button is already selected.
    if ("selected" in btn.dataset) {
        return;
    }

    const controls = btn.closest(".extra-network-controls");
    const tab = extra_networks_tabs[controls.dataset.tabnameFull];
    tab.controls_elem.querySelectorAll(".extra-network-control--sort-mode").forEach(elem => {
        delete elem.dataset.selected;
        delete elem.dataset.recurse;
    });

    btn.dataset.selected = "";

    const sort_mode_str = btn.dataset.sortMode.toLowerCase();

    tab.setSortMode(sort_mode_str);
}

function extraNetworksControlSortDirOnClick(event) {
    /** Handles `onclick` events for the Sort Direction button.
     *
     * Modifies the data attributes of the Sort Direction button to cycle between
     * ascending and descending sort directions.
     */
    const btn = event.target.closest(".extra-network-control--sort-dir");
    const controls = btn.closest(".extra-network-controls");
    const tab = extra_networks_tabs[controls.dataset.tabnameFull];

    const curr_sort_dir_str = btn.dataset.sortDir.toLowerCase();
    if (!["ascending", "descending"].includes(curr_sort_dir_str)) {
        console.error(`Invalid sort_dir_str: ${curr_sort_dir_str}`);
        return;
    }

    let sort_dir_str = curr_sort_dir_str === "ascending" ? "descending" : "ascending";
    btn.dataset.sortDir = sort_dir_str;
    btn.setAttribute("title", `Sort ${sort_dir_str}`);

    tab.setSortDir(sort_dir_str);
}

async function extraNetworksControlTreeViewOnClick(event) {
    /** Handles `onclick` events for the Tree View button.
     *
     * Toggles the tree view in the extra networks pane.
     */
    const btn = event.target.closest(".extra-network-control--tree-view");
    const controls = btn.closest(".extra-network-controls");
    const tab = extra_networks_tabs[controls.dataset.tabnameFull];
    tab.tree_view_en = !("selected" in btn.dataset);
    if ("selected" in btn.dataset) {
        delete btn.dataset.selected;
    } else {
        btn.dataset.selected = "";
    }

    // If hiding, clear the tree list selections before hiding it.
    if (!tab.tree_view_en) {
        await tab.tree_list.onRowSelected();
    }

    tab.tree_list.scroll_elem.parentElement.classList.toggle("hidden", !tab.tree_view_en);
    tab.tree_list.enable(tab.tree_view_en);

    // Apply the resize-handle-hidden class to the resize-handle-row.
    // NOTE: This can be simplified using only css with the ":has" selector however
    // this is only recently supported in firefox. So for now we just add a class
    // to the resize-handle-row instead.
    const resize_handle_row = tab.tree_list.scroll_elem.closest(".resize-handle-row");
    resize_handle_row.classList.toggle("resize-handle-hidden", !tab.tree_view_en);

    if (tab.tree_view_en && tab.dirs_view_en) {
        tab.setDirectoryButtons({source_class: ".extra-network-dirs-view-button"});
    }
}

function extraNetworksControlDirsViewOnClick(event) {
    /** Handles `onclick` events for the Dirs View button.
     *
     * Toggles the directory view in the extra networks pane.
     */
    const btn = event.target.closest(".extra-network-control--dirs-view");
    const controls = btn.closest(".extra-network-controls");
    const tab = extra_networks_tabs[controls.dataset.tabnameFull];
    tab.dirs_view_en = !("selected" in btn.dataset);
    if (tab.dirs_view_en) {
        btn.dataset.selected = "";
    } else {
        delete btn.dataset.selected;
    }

    if (!tab.dirs_view_en) {
        // If hiding, we want to deselect all buttons prior to hiding.
        tab.container_elem.querySelectorAll(
            ".extra-network-dirs-view-button[data-selected]"
        ).forEach(elem => {
            delete elem.dataset.selected;
            delete elem.dataset.recurse;
        });
    }

    tab.container_elem.querySelector(
        ".extra-network-content--dirs-view"
    ).classList.toggle("hidden", !tab.dirs_view_en);

    if (tab.dirs_view_en && tab.tree_view_en) {
        tab.setDirectoryButtons({source_class: ".tree-list-item"});
    }
}

function extraNetworksControlRefreshOnClick(event) {
    /** Handles `onclick` events for the Refresh Page button.
     *
     * In order to actually call the python functions in `ui_extra_networks.py`
     * to refresh the page, we created an empty gradio button in that file with an
     * event handler that refreshes the page. So what this function here does
     * is it manually raises a `click` event on that button.
     */
    clearTimeout(extra_networks_refresh_internal_debounce_timer);
    extra_networks_refresh_internal_debounce_timer = setTimeout(async() => {
        const btn = event.target.closest(".extra-network-control--refresh");
        const controls = btn.closest(".extra-network-controls");
        const tab = extra_networks_tabs[controls.dataset.tabnameFull];
        // We want to reset tab lists on refresh click so that the viewing area
        // shows that it is loading new data.
        tab.tree_list.clear();
        tab.cards_list.clear();
        // Fire an event for this button click.
        gradioApp().getElementById(
            `${controls.dataset.tabnameFull}_extra_refresh_internal`
        ).dispatchEvent(new Event("click"));
    }, EXTRA_NETWORKS_REFRESH_INTERNAL_DEBOUNCE_TIMEOUT_MS);
}

function extraNetworksSelectModel({tab, prompt, neg_prompt, allow_neg, checkpoint_name}) {
    if (checkpoint_name) {
        selectCheckpoint(checkpoint_name);
    } else if (neg_prompt) {
        extraNetworksUpdatePrompt(tab.txt_prompt_elem, prompt);
        extraNetworksUpdatePrompt(tab.txt_neg_prompt_elem, neg_prompt);
    } else if (allow_neg) {
        extraNetworksUpdatePrompt(tab.active_prompt_elem, prompt);
    } else {
        extraNetworksUpdatePrompt(tab.txt_prompt_elem, prompt);
    }
}

function extraNetworksCardOnClick(event) {
    // Do not select the card if its child button-row is the target of the event.
    if (event.target.closest(".button-row")) {
        return;
    }

    const btn = event.target.closest(".card");
    const pane = btn.closest(".extra-network-pane");
    const tab = extra_networks_tabs[pane.dataset.tabnameFull];

    let checkpoint_name;
    if ("isCheckpoint" in btn.dataset) {
        checkpoint_name = btn.dataset.name;
    }
    extraNetworksSelectModel({
        tab: tab,
        prompt: btn.dataset.prompt,
        neg_prompt: btn.dataset.negPrompt,
        allow_neg: btn.dataset.allowNeg,
        checkpoint_name: checkpoint_name,
    });
}

function extraNetworksTreeFileOnClick(event) {
    // Do not select the row if its child button-row is the target of the event.
    if (event.target.closest(".tree-list-item-action")) {
        return;
    }

    const btn = event.target.closest(".tree-list-item");
    const pane = btn.closest(".extra-network-pane");
    const tab = extra_networks_tabs[pane.dataset.tabnameFull];


    let checkpoint_name;
    if ("isCheckpoint" in btn.dataset) {
        checkpoint_name = btn.dataset.name;
    }
    extraNetworksSelectModel({
        tab: tab,
        prompt: btn.dataset.prompt,
        neg_prompt: btn.dataset.negPrompt,
        allow_neg: btn.dataset.allowNeg,
        checkpoint_name: checkpoint_name,
    });
}

async function extraNetworksTreeDirectoryOnLongPress(event) {
    // Do not select the row if its child button-row is the target of the event.
    if (event.target.closest(".tree-list-item-action")) {
        return;
    }

    const btn = event.target.closest(".tree-list-item");
    const pane = btn.closest(".extra-network-pane");
    const tab = extra_networks_tabs[pane.dataset.tabnameFull];

    tab.setDirectoryButtons({source_elem: btn});
}

async function extraNetworksTreeDirectoryOnDblClick(event) {
    // stopPropagation so we don't also trigger event on parent since this btn is nested.
    event.stopPropagation();
    const btn = event.target.closest(".tree-list-item");
    const pane = btn.closest(".extra-network-pane");
    const div_id = btn.dataset.divId;
    const tab = extra_networks_tabs[pane.dataset.tabnameFull];
    await tab.tree_list.toggleRowExpanded(div_id);
    tab.setDirectoryButtons({source_class: ".tree-list-item"});
}

async function extraNetworksTreeDirectoryOnClick(event) {
    // Do not select the row if its child button-row is the target of the event.
    if (event.target.closest(".tree-list-item-action")) {
        return;
    }
    const btn = event.target.closest(".tree-list-item");
    const pane = btn.closest(".extra-network-pane");
    const tab = extra_networks_tabs[pane.dataset.tabnameFull];
    tab.setDirectoryButtons({source_elem: btn});
}

async function extraNetworksTreeDirectoryChevronOnLongPress(event) {
    // stopPropagation so we don't also trigger event on parent since this btn is nested.
    event.stopPropagation();
    const chevron = event.target.closest(".tree-list-item-action--chevron");
    const btn = event.target.closest(".tree-list-item");
    const pane = btn.closest(".extra-network-pane");
    const div_id = btn.dataset.divId;
    const tab = extra_networks_tabs[pane.dataset.tabnameFull];
    if ("expanded" in btn.dataset) {
        await tab.tree_list.collapseAllRows(div_id);
    } else {
        await tab.tree_list.expandAllRows(div_id);
    }
    tab.setDirectoryButtons({source_class: ".tree-list-item"});
}

async function extraNetworksBtnTreeViewChevronOnClick(event) {
    // stopPropagation so we don't also trigger event on parent since this btn is nested.
    event.stopPropagation();
    const btn = event.target.closest(".tree-list-item");
    const pane = btn.closest(".extra-network-pane");
    const div_id = btn.dataset.divId;
    const tab = extra_networks_tabs[pane.dataset.tabnameFull];
    await tab.tree_list.toggleRowExpanded(div_id);
    tab.setDirectoryButtons({source_class: ".tree-list-item"});
}

function extraNetworksBtnShowMetadataOnClick(event) {
    // stopPropagation so we don't also trigger event on parent since this btn is nested.
    event.stopPropagation();
    const btn = event.target.closest(".metadata-button");
    const pane = btn.closest(".extra-network-pane");
    let parent = btn.closest(".card");
    if (!parent) {
        parent = btn.closest(".tree-list-item");
    }
    extraNetworksFetchMetadata(pane.dataset.extraNetworksTabname, parent.dataset.name);
}

function extraNetworksBtnEditMetadataOnClick(event) {
    // stopPropagation so we don't also trigger event on parent since this btn is nested.
    event.stopPropagation();
    const btn = event.target.closest(".edit-button");
    const pane = btn.closest(".extra-network-pane");
    let parent = btn.closest(".card");
    if (!parent) {
        parent = btn.closest(".tree-list-item");
    }
    const id = `${pane.dataset.tabnameFull}_edit_user_metadata`;
    let editor = extraPageUserMetadataEditors[id];
    if (isNullOrUndefined(editor)) {
        editor = {};
        editor.page = gradioApp().getElementById(id);
        editor.nameTextarea = gradioApp().querySelector(`#${id}_name textarea`);
        editor.button = gradioApp().querySelector(`#${id}_button`);
        extraPageUserMetadataEditors[id] = editor;
    }

    editor.nameTextarea.value = parent.dataset.name;
    updateInput(editor.nameTextarea);

    editor.button.click();

    popup(editor.page);
}

function extraNetworksBtnCopyPathOnClick(event) {
    // stopPropagation so we don't also trigger event on parent since this btn is nested.
    event.stopPropagation();
    const btn = event.target.closest(".copy-path-button");
    copyToClipboard(btn.dataset.clipboardText);
}

// ==== MAIN SETUP ====

function extraNetworksSetupEventDelegators() {
    /** Sets up event delegators for all extraNetworks tabs.
     *
     *  These event handlers are not tied to any specific elements on the page.
     *  We do this because elements within each tab may be removed and replaced
     *  which would break references to elements in DOM and thus prevent any event
     *  listeners from firing.
     */

    window.addEventListener("resizeHandleDblClick", event => {
        // See resizeHandle.js::onDoubleClick() for event detail.
        event.stopPropagation();
        const pane = event.target.closest(".extra-network-pane");
        extra_networks_tabs[pane.dataset.tabnameFull].autoSetTreeWidth();
    });

    // Debounce search text input. This way we only search after user is done typing.
    const search_input_debounce = _debounce(tabname_full => {
        extra_networks_tabs[tabname_full].applyFilter();
    }, SEARCH_INPUT_DEBOUNCE_TIME_MS);

    window.addEventListener("keyup", event => {
        const controls = event.target.closest(".extra-network-controls");
        if (isElement(controls)) {
            const tabname_full = controls.dataset.tabnameFull;
            const target = event.target.closest(".extra-network-control--search-text");
            if (isElement(target)) {
                search_input_debounce.call(target, tabname_full);
            }
        }
    });

    window.addEventListener("keydown", event => {
        if (event.key === "Escape") {
            closePopup();
        }
    });

    const click_event_map = {
        ".tree-list-item--file": extraNetworksTreeFileOnClick,
        ".card": extraNetworksCardOnClick,
        ".copy-path-button": extraNetworksBtnCopyPathOnClick,
        ".edit-button": extraNetworksBtnEditMetadataOnClick,
        ".metadata-button": extraNetworksBtnShowMetadataOnClick,
        ".extra-network-control--search-clear": extraNetworksControlSearchClearOnClick,
        ".extra-network-control--sort-mode": extraNetworksControlSortModeOnClick,
        ".extra-network-control--sort-dir": extraNetworksControlSortDirOnClick,
        ".extra-network-control--dirs-view": extraNetworksControlDirsViewOnClick,
        ".extra-network-control--tree-view": extraNetworksControlTreeViewOnClick,
        ".extra-network-control--refresh": extraNetworksControlRefreshOnClick,
    };

    window.addEventListener("click", event => {
        for (const [selector, handler] of Object.entries(click_event_map)) {
            if (event.target.closest(selector)) {
                handler(event);
            }
        }
    });

    // Order in these maps matters since we may have separate events for both a div
    // and for a child within that div however if the child is clicked then we wouldn't
    // want to handle clicks for the parent as well. In this case, order the child's event
    // before the parent and the parent will be ignored.
    // Can add entries with handler=null to forcefully ignore specific event types.

    const short_press_event_map = [
        {
            selector: ".tree-list-item-action--chevron",
            handler: extraNetworksBtnTreeViewChevronOnClick,
        },
        {
            selector: ".tree-list-item--dir",
            negative: ".tree-list-item-action",
            handler: extraNetworksTreeDirectoryOnClick,
        },
        {
            selector: ".extra-network-dirs-view-button",
            handler: extraNetworksBtnDirsViewItemOnClick,
        },
    ];

    const long_press_event_map = [
        {
            selector: ".tree-list-item-action--chevron",
            handler: extraNetworksTreeDirectoryChevronOnLongPress,
        },
        {
            selector: ".tree-list-item--dir",
            negative: ".tree-list-item-action",
            handler: extraNetworksTreeDirectoryOnLongPress,
        },
        {
            selector: ".extra-network-dirs-view-button",
            handler: extraNetworksBtnDirsViewItemOnLongPress,
        },
    ];

    const dbl_press_event_map = [
        {
            selector: ".tree-list-item--dir",
            negative: ".tree-list-item-action",
            handler: extraNetworksTreeDirectoryOnDblClick,
        },
    ];

    const on_short_press = (event, elem, handler) => {
        if (!handler) {
            return;
        }
        // Toggle
        if (elem.classList.contains("long-pressed")) {
            elem.classList.remove("long-pressed");
            delete elem.dataset.selected;
            delete elem.dataset.recurse;
        } else {
            elem.classList.toggle("short-pressed");
            elem.toggleAttribute("data-selected");
        }

        elem.dispatchEvent(new Event("shortpress", event));
        handler(event);
    };

    const on_long_press = (event, elem, handler) => {
        if (!handler) {
            return;
        }
        // If long pressed, we deselect.
        // Else we set as long pressed.
        if (elem.classList.contains("long-pressed")) {
            elem.classList.remove("long-pressed");
            delete elem.dataset.recurse;
            // Don't want to remove selected state if btn was previously short-pressed.
            if (!elem.classList.contains("short-pressed")) {
                delete elem.dataset.selected;
            }
        } else {
            elem.classList.toggle("long-pressed");
            elem.dataset.selected = "";
            elem.dataset.recurse = "";
        }

        elem.dispatchEvent(new Event("longpress", event));
        handler(event);
    };

    const on_dbl_press = (event, elem, handler) => {
        if (!handler) {
            return;
        }
        handler(event);
    };

    let press_timer;
    let press_time_ms = 800;

    window.addEventListener("mousedown", event => {
        for (const obj of short_press_event_map) {
            const elem = event.target.closest(obj.selector);
            const neg = obj.negative ? event.target.closest(obj.negative) : null;
            if (elem && !neg) {
                event.preventDefault();
                event.stopPropagation();
                elem.classList.add("pressed");
            }
        }

        for (const obj of long_press_event_map) {
            const elem = event.target.closest(obj.selector);
            const neg = obj.negative ? event.target.closest(obj.negative) : null;
            if (elem && !neg) {
                event.preventDefault();
                event.stopPropagation();
                elem.classList.add("pressed");
                press_timer = setTimeout(() => {
                    elem.classList.remove("pressed");
                    on_long_press(event, elem, obj.handler);
                }, press_time_ms);
            }
        }

        for (const obj of dbl_press_event_map) {
            const elem = event.target.closest(obj.selector);
            const neg = obj.negative ? event.target.closest(obj.negative) : null;
            if (elem && !neg) {
                event.preventDefault();
                event.stopPropagation();
                elem.classList.add("pressed");
            }
        }
    });

    window.addEventListener("mouseup", event => {
        for (const obj of short_press_event_map) {
            const elem = event.target.closest(obj.selector);
            const neg = obj.negative ? event.target.closest(obj.negative) : null;
            if (elem && !neg) {
                event.preventDefault();
                event.stopPropagation();
                clearTimeout(press_timer);
                if (elem.classList.contains("pressed")) {
                    if (event.detail === 1 ||
                        !dbl_press_event_map.map(x => x.selector).includes(obj.selector)
                    ) {
                        elem.classList.remove("pressed");
                        on_short_press(event, elem, obj.handler);
                    }
                }
            }
        }

        if (event.detail % 2 === 0) {
            for (const obj of dbl_press_event_map) {
                const elem = event.target.closest(obj.selector);
                const neg = obj.negative ? event.target.closest(obj.negative) : null;
                if (elem && !neg) {
                    event.preventDefault();
                    event.stopPropagation();
                    clearTimeout(press_timer);
                    if (elem.classList.contains("pressed")) {
                        elem.classList.remove("pressed");
                        on_dbl_press(event, elem, obj.handler);
                    }
                }
            }
        }

        // long_press_event_map is handled by the timer setup in "mousedown" handlers.
    });
}

async function extraNetworksSetupTab(tabname) {
    const this_tab = await waitForElement(`#${tabname}_extra_tabs`);
    const tab_nav = await waitForElement(`#${tabname}_extra_tabs > div.tab-nav`);
    const controls_div = document.createElement("div");

    controls_div.id = `${tabname}_extra_network_controls_div`;
    controls_div.classList.add("extra-network-controls-div");
    tab_nav.appendChild(controls_div);
    tab_nav.insertBefore(controls_div, null);

    const panes = this_tab.querySelectorAll(`:scope > .tabitem[id^="${tabname}_"]`);
    for (const pane of panes) {
        const tabname_full = pane.id;
        const extra_networks_tabname = tabname_full.replace(`${tabname}_`, "");
        extra_networks_tabs[tabname_full] = new ExtraNetworksTab({
            tabname: tabname,
            extra_networks_tabname: extra_networks_tabname,
        });
        await extra_networks_tabs[tabname_full].setup(pane, controls_div);
    }
}

async function extraNetworksSetup() {
    await waitForBool(initialUiOptionsLoaded);

    await Promise.all([
        extraNetworksSetupTab('txt2img'),
        extraNetworksSetupTab('img2img'),
    ]);

    extraNetworksSetupEventDelegators();
}

onUiLoaded(extraNetworksSetup);
onOptionsChanged(() => initialUiOptionsLoaded.state = true);
