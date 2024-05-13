// Prevent eslint errors on functions defined in other files.
/*global
    selectCheckpoint,
    ExtraNetworksClusterizeTreeList,
    ExtraNetworksClusterizeCardList,
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
let extra_networks_curr_options = {};

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
    card_list;
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
    show_prompt = true;
    show_neg_prompt = true;
    compact_prompt_en = false;
    refresh_in_progress = false;
    dirs_view_en = false;
    tree_view_en = false;
    card_view_en = false;
    card_list_splash_state = null;
    tree_list_splash_state = null;
    directory_filters = {};
    list_button_states = {};
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
            waitForElement(`#${this.tabname_full}_card_list_scroll_area`),
            waitForElement(`#${this.tabname_full}_card_list_content_area`),
        ]);

        this.updateSplashState({card_list_state: "loading", tree_list_state: "loading"});

        this.txt_search_elem = this.controls_elem.querySelector(".extra-network-control--search-text");

        // determine whether compact prompt mode is enabled.
        // cannot await this since it may not exist on page depending on user setting.
        this.compact_prompt_en = isElement(gradioApp().querySelector(".toprow-compact-tools"));
        this.dirs_view_en = opts.extra_networks_dirs_view_default_enabled;
        this.tree_view_en = opts.extra_networks_tree_view_default_enabled;
        this.card_view_en = opts.extra_networks_card_view_default_enabled;

        // setup this tab's controls
        this.controls_elem.id = `${this.tabname_full}_controls`;
        controls_div.insertBefore(this.controls_elem, null);

        const error_btn = this.controls_elem.querySelector(
            ".extra-network-control--refresh"
        ).cloneNode(true);
        error_btn.id = null;
        error_btn.dataset.tabnameFull = this.tabname_full;
        this.clusterize_error_html = "<div>Data Error.<br/>" +
            "Please refresh tab.<br>" +
            `${error_btn.outerHTML}</div>`;

        this.setupTreeList();
        this.setupCardList();

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
        this.card_list.destroy();
        this.tree_list = null;
        this.card_list = null;
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
        this.card_view_en = false;
        this.directory_filters = {};
        this.list_button_states = {};
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

    setupTreeList() {
        if (this.tree_list instanceof ExtraNetworksClusterizeTreeList) {
            this.tree_list.destroy();
        }
        this.tree_list = new ExtraNetworksClusterizeTreeList({
            tabname: this.tabname,
            extra_networks_tabname: this.extra_networks_tabname,
            scrollId: `${this.tabname_full}_tree_list_scroll_area`,
            contentId: `${this.tabname_full}_tree_list_content_area`,
            tag: "button",
            error_html: this.clusterize_error_html,
            callbacks: {
                initData: this.onInitTreeData.bind(this),
                fetchData: this.onFetchTreeData.bind(this),
            },
        });
    }

    setupCardList() {
        if (this.card_list instanceof ExtraNetworksClusterizeCardList) {
            this.card_list.destroy();
        }
        this.card_list = new ExtraNetworksClusterizeCardList({
            tabname: this.tabname,
            extra_networks_tabname: this.extra_networks_tabname,
            scrollId: `${this.tabname_full}_card_list_scroll_area`,
            contentId: `${this.tabname_full}_card_list_content_area`,
            tag: "div",
            no_data_html: "No data matching filter.",
            error_html: this.clusterize_error_html,
            callbacks: {
                initData: this.onInitCardData.bind(this),
                fetchData: this.onFetchCardData.bind(this),
            },
        });
    }

    setSortMode(sort_mode_str) {
        this.sort_mode_str = sort_mode_str;
        this.card_list.setSortMode(this.sort_mode_str);
    }

    setSortDir(sort_dir_str) {
        this.sort_dir_str = sort_dir_str;
        this.card_list.setSortDir(this.sort_dir_str);
    }

    setFilterStr(filter_str) {
        this.filter_str = filter_str;
        this.card_list.setFilterStr(this.filter_str);
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
                    this.card_list.updateHtml(elem, data.html);
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

    updateSplashState({card_list_state, tree_list_state} = {}) {
        const _handle_state = (state_str, loading_elem, no_data_elem) => {
            switch (state_str) {
            case "loading":
                no_data_elem.classList.add("hidden");
                loading_elem.classList.remove("hidden");
                break;
            case "no_data":
                loading_elem.classList.add("hidden");
                no_data_elem.classList.remove("hidden");
                break;
            case "show":
                no_data_elem.classList.add("hidden");
                loading_elem.classList.add("hidden");
                break;
            default:
                break;
            }
        };

        let loading_elem;
        let no_data_elem;

        if (isString(card_list_state)) {
            this.card_list_splash_state = card_list_state;
            loading_elem = document.getElementById(`${this.tabname_full}_card_list_loading_splash`);
            no_data_elem = document.getElementById(`${this.tabname_full}_card_list_no_data_splash`);
            _handle_state(card_list_state, loading_elem, no_data_elem);
        }

        if (isString(tree_list_state)) {
            this.tree_list_splash_state = tree_list_state;
            loading_elem = document.getElementById(`${this.tabname_full}_tree_list_loading_splash`);
            no_data_elem = document.getElementById(`${this.tabname_full}_tree_list_no_data_splash`);
            _handle_state(tree_list_state, loading_elem, no_data_elem);
        }
    }

    async #refresh() {
        this.updateSplashState({card_list_state: "loading", tree_list_state: "loading"});

        try {
            await this.waitForServerPageReady();
        } catch (error) {
            console.error(`refresh error: ${error.message}`);
            this.updateSplashState({card_list_state: "no_data", tree_list_state: "no_data"});
            return;
        }

        const btn_dirs_view = this.controls_elem.querySelector(".extra-network-control--dirs-view");
        const btn_tree_view = this.controls_elem.querySelector(".extra-network-control--tree-view");
        const btn_card_view = this.controls_elem.querySelector(".extra-network-control--card-view");
        this.dirs_view_en = "selected" in btn_dirs_view.dataset;
        this.tree_view_en = "selected" in btn_tree_view.dataset;
        this.card_view_en = "selected" in btn_card_view.dataset;

        const div_dirs = this.container_elem.querySelector(".extra-network-content--dirs-view");
        const div_tree = this.container_elem.querySelector(".extra-network-content--tree-view");
        const div_card = this.container_elem.querySelector(".extra-network-content--card-view");
        // Remove "hidden" class if button is enabled, otherwise add it.
        div_dirs.classList.toggle("hidden", !this.dirs_view_en);
        div_tree.classList.toggle("hidden", !this.tree_view_en);
        div_card.classList.toggle("hidden", !this.card_view_en);

        // Apply the current resize handle classes.
        const resize_handle_row = div_tree.closest(".resize-handle-row");
        resize_handle_row.classList.toggle("resize-handle-hidden", !this.tree_view_en || !this.card_view_en);
        if (this.tree_view_en && !this.card_view_en) {
            div_tree.style.flexGrow = 1;
        } else {
            div_tree.style.flexGrow = null;
        }

        await Promise.all([this.setupTreeList(), this.setupCardList()]);
        this.tree_list.enable(this.tree_view_en);
        this.card_list.enable(this.card_view_en);
        await Promise.all([this.tree_list.load(true), this.card_list.load(true)]);

        // apply the previous sort/filter options
        await this.applyListButtonStates();
        this.setSortMode(this.sort_mode_str);
        this.setSortDir(this.sort_dir_str);
        this.applyDirectoryFilters();
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
        this.updateSplashState({
            card_list_state: this.card_list_splash_state,
            tree_list_state: this.tree_list_splash_state,
        });
        this.showControls();

        const div_dirs = this.container_elem.querySelector(".extra-network-content--dirs-view");
        const div_tree = this.container_elem.querySelector(".extra-network-content--tree-view");
        const div_card = this.container_elem.querySelector(".extra-network-content--card-view");
        div_dirs.classList.toggle("hidden", !this.dirs_view_en);
        div_tree.classList.toggle("hidden", !this.tree_view_en);
        div_card.classList.toggle("hidden", !this.card_view_en);

        // Apply the current resize handle classes.
        const resize_handle_row = div_tree.closest(".resize-handle-row");
        resize_handle_row.classList.toggle("resize-handle-hidden", !this.tree_view_en || !this.card_view_en);
        if (this.tree_view_en && !this.card_view_en) {
            div_tree.style.flexGrow = 1;
        } else {
            div_tree.style.flexGrow = null;
        }

        this.tree_list.enable(this.tree_view_en);
        this.card_list.enable(this.card_view_en);
        await Promise.all([this.tree_list.load(), this.card_list.load()]);
    }

    unload() {
        this.movePrompt(false, false);
        this.hideControls();
        this.tree_list.enable(false);
        this.card_list.enable(false);
    }

    addDirectoryFilter(div_id, filter_str, recurse) {
        filter_str = isString(filter_str) ? filter_str : "";
        recurse = recurse === true || recurse === false ? recurse : false;
        this.card_list.addDirectoryFilter(div_id, filter_str, recurse);
        this.directory_filters[div_id] = {filter_str: filter_str, recurse: recurse};
    }

    removeDirectoryFilter(div_id) {
        this.card_list.removeDirectoryFilter(div_id);
        delete this.directory_filters[div_id];
    }

    clearDirectoryFilters({excluded_div_ids} = {}) {
        if (isNumber(excluded_div_ids)) {
            excluded_div_ids = [excluded_div_ids];
        }

        if (!Array.isArray(excluded_div_ids)) {
            excluded_div_ids = [];
        }

        this.card_list.clearDirectoryFilters({excluded_div_ids: excluded_div_ids});
        for (const div_id of Object.keys(this.directory_filters)) {
            if (excluded_div_ids.includes(div_id)) {
                continue;
            }
            delete this.directory_filters[div_id];
        }
    }

    setDirectoryFilters() {
        this.card_list.setDirectoryFilters(this.directory_filters);
    }

    applyDirectoryFilters() {
        this.card_list.filterData();
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

    async onInitCardData() {
        this.updateSplashState({card_list_state: "loading"});
        try {
            await this.waitForServerPageReady();
        } catch (error) {
            console.error(`onInitCardData error: ${error.message}`);
            this.updateSplashState({card_list_state: "no_data"});
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

        const url = "./sd_extra_networks/init-card-data";
        const payload = {tabname: this.tabname, extra_networks_tabname: this.extra_networks_tabname};
        const timeout_ms = EXTRA_NETWORKS_INIT_DATA_TIMEOUT_MS;
        const opts = {timeout_ms: timeout_ms, response_handler: response_handler};
        try {
            const response = await fetchWithRetryAndBackoff(url, payload, opts);
            if (Object.keys(response.data).length === 0) {
                this.updateSplashState({card_list_state: "no_data"});
            } else {
                this.updateSplashState({card_list_state: "show"});
            }
            return response.data;
        } catch (error) {
            console.error(`onInitCardData error: ${error.message}`);
            this.updateSplashState({card_list_state: "no_data"});
            return {};
        }
    }

    async onInitTreeData() {
        this.updateSplashState({tree_list_state: "loading"});
        try {
            await this.waitForServerPageReady();
        } catch (error) {
            console.error(`onInitTreeData error: ${error.message}`);
            this.updateSplashState({tree_list_state: "no_data"});
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
            if (Object.keys(response.data).length === 0) {
                this.updateSplashState({tree_list_state: "no_data"});
            } else {
                this.updateSplashState({tree_list_state: "show"});
            }
            return response.data;
        } catch (error) {
            console.error(`onInitTreeData error: ${error.message}`);
            this.updateSplashState({tree_list_state: "no_data"});
            return {};
        }
    }

    async onFetchCardData(div_ids) {
        const url = "./sd_extra_networks/fetch-card-data";
        const payload = {extra_networks_tabname: this.extra_networks_tabname, div_ids: div_ids};
        const timeout_ms = EXTRA_NETWORKS_FETCH_DATA_TIMEOUT_MS;
        const opts = {timeout_ms: timeout_ms};
        try {
            const response = await fetchWithRetryAndBackoff(url, payload, opts);
            if (response.missing_div_ids.length) {
                console.warn(`Failed to fetch multiple div_ids: ${response.missing_div_ids}`);
            }
            if (Object.keys(response.data).length === 0) {
                this.updateSplashState({card_list_state: "no_data"});
            } else {
                this.updateSplashState({card_list_state: "show"});
            }
            return response.data;
        } catch (error) {
            console.error(`onFetchCardData error: ${error.message}`);
            this.updateSplashState({card_list_state: "no_data"});
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
            if (Object.keys(response.data).length === 0) {
                this.updateSplashState({tree_list_state: "no_data"});
            } else {
                this.updateSplashState({tree_list_state: "show"});
            }
            return response.data;
        } catch (error) {
            console.error(`onFetchTreeData error: ${error.message}`);
            this.updateSplashState({tree_list_state: "no_data"});
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

    async clearSelectedButtons({excluded_div_ids} = {}) {
        /** excluded_elems can be an element or an array of elements. */
        if (isNumber(excluded_div_ids)) {
            excluded_div_ids = [excluded_div_ids];
        }

        if (!Array.isArray(excluded_div_ids)) {
            excluded_div_ids = [];
        }

        for (const div_id of Object.keys(this.list_button_states)) {
            if (excluded_div_ids.includes(div_id)) {
                continue;
            }

            const tree_btn = this.tree_list.content_elem.querySelector(`.tree-list-item[data-div-id="${div_id}"]`);
            const tree_btn_exists = isElement(tree_btn) && !isNullOrUndefined(tree_btn.offsetParent);
            if (tree_btn_exists) {
                await this.deselectTreeListRow(tree_btn);
            }

            const dirs_btn = this.container_elem.querySelector(`.extra-network-dirs-view-button[data-div-id="${div_id}"]`);
            const dirs_btn_exists = isElement(dirs_btn) && !isNullOrUndefined(dirs_btn.offsetParent);
            if (dirs_btn_exists) {
                this.deselectDirsViewButton(dirs_btn);
            }

            delete this.list_button_states[div_id];
        }

        this.tree_list.clearSelectedRows({excluded_div_ids: excluded_div_ids});
        this.clearDirectoryFilters({excluded_div_ids: excluded_div_ids});
        await this.tree_list.update();
    }

    refreshTreeListRecursionDepth() {
        const keys = Object.keys(this.list_button_states);
        const div_ids = keys.filter(k => this.list_button_states[k].recurse);
        // Fix any invalid recursed rows.
        this.tree_list.content_elem.querySelectorAll(
            ".tree-list-item.recurse"
        ).forEach(elem => {
            if (!keys.includes(elem.dataset.divId) ||
                !this.list_button_states[elem.dataset.divId].recurse
            ) {
                elem.classList.remove("recurse");
                this.tree_list.updateHtml(elem);
            }
        });
        // Fix any invalid recursed indents.
        this.tree_list.content_elem.querySelectorAll(
            ".tree-list-item-indent [data-selected]"
        ).forEach(elem => {
            if (!div_ids.includes(elem.dataset.parentId)) {
                delete elem.dataset.selected;
                this.tree_list.updateHtml(elem.closest(".tree-list-item"));
            }
        });

        // Update indents with current recursed rows.
        div_ids.forEach(div_id => {
            this.tree_list.content_elem.querySelectorAll(
                `.tree-list-item-indent [data-parent-id="${div_id}"]`
            ).forEach(elem => {
                elem.dataset.selected = "";
                this.tree_list.updateHtml(elem.closest(".tree-list-item"));
            });
        });
    }

    setTreeListRecursionDepth(parent_id, state) {
        const elems = this.tree_list.content_elem.querySelectorAll(
            `.tree-list-item-indent [data-parent-id="${parent_id}"]`
        );
        for (const elem of elems) {
            elem.toggleAttribute("data-selected", state);
            this.tree_list.updateHtml(elem.closest(".tree-list-item"));
        }
    }

    async selectTreeListRow(elem) {
        elem.dataset.selected = "";

        const left_click_opt = opts.extra_networks_directory_filter_click_behavior.toLowerCase().trim();
        let recurse;
        if (left_click_opt === "click") {
            recurse = elem.classList.contains("short-pressed") && !elem.classList.contains("long-pressed");
        } else {
            recurse = elem.classList.contains("long-pressed");
        }
        elem.classList.toggle("recurse", recurse);

        this.tree_list.setRowSelected(elem);
        let directory_filter = elem.dataset.path;
        if ("directoryFilterOverride" in elem.dataset) {
            directory_filter = elem.dataset.directoryFilterOverride;
        }

        this.setTreeListRecursionDepth(elem.dataset.divId, recurse);
        this.addDirectoryFilter(elem.dataset.divId, directory_filter, recurse);
        await this.tree_list.update();

        this.list_button_states[elem.dataset.divId] = {
            short_pressed: elem.classList.contains("short-pressed"),
            long_pressed: elem.classList.contains("long-pressed"),
            recurse: elem.classList.contains("recurse"),
        };
    }

    async deselectTreeListRow(elem) {
        delete elem.dataset.selected;
        elem.classList.remove("short-pressed");
        elem.classList.remove("long-pressed");
        elem.classList.remove("recurse");
        this.tree_list.setRowDeselected(elem);
        this.setTreeListRecursionDepth(elem.dataset.divId, false);
        this.removeDirectoryFilter(elem.dataset.divId);
        await this.tree_list.update();
        delete this.list_button_states[elem.dataset.divId];
    }

    selectDirsViewButton(elem) {
        elem.dataset.selected = "";
        let directory_filter = elem.dataset.path;
        if ("directoryFilterOverride" in elem.dataset) {
            directory_filter = elem.dataset.directoryFilterOverride;
        }

        const left_click_opt = opts.extra_networks_directory_filter_click_behavior.toLowerCase().trim();
        let recurse;
        if (left_click_opt === "click") {
            recurse = elem.classList.contains("short-pressed") && !elem.classList.contains("long-pressed");
        } else {
            recurse = elem.classList.contains("long-pressed");
        }
        elem.classList.toggle("recurse", recurse);

        this.addDirectoryFilter(elem.dataset.divId, directory_filter, recurse);

        this.list_button_states[elem.dataset.divId] = {
            short_pressed: elem.classList.contains("short-pressed"),
            long_pressed: elem.classList.contains("long-pressed"),
            recurse: elem.classList.contains("recurse"),
        };
    }

    deselectDirsViewButton(elem) {
        delete elem.dataset.selected;
        elem.classList.remove("short-pressed");
        elem.classList.remove("long-pressed");
        elem.classList.remove("recurse");
        this.removeDirectoryFilter(elem.dataset.divId);
        delete this.list_button_states[elem.dataset.divId];
    }

    async applyListButtonStates() {
        const selected_tree_rows = this.tree_list.content_elem.querySelectorAll(".tree-list-item[data-selected]");
        for (const elem of selected_tree_rows) {
            if (!Object.keys(this.list_button_states).includes(elem.dataset.divId)) {
                await this.deselectTreeListRow(elem);
            }
        }

        const selected_dirs_buttons = this.container_elem.querySelectorAll(".extra-network-dirs-view-button[data-selected]");
        for (const elem of selected_dirs_buttons) {
            if (!Object.keys(this.list_button_states).includes(elem.dataset.divId)) {
                this.deselectDirsViewButton(elem);
            }
        }

        for (const [div_id, state] of Object.entries(this.list_button_states)) {
            const tree_btn = this.tree_list.content_elem.querySelector(`.tree-list-item[data-div-id="${div_id}"]`);
            const tree_btn_exists = isElement(tree_btn) && !isNullOrUndefined(tree_btn.offsetParent);
            if (tree_btn_exists) {
                tree_btn.classList.toggle("short-pressed", state.short_pressed);
                tree_btn.classList.toggle("long-pressed", state.long_pressed);
                tree_btn.classList.toggle("recurse", state.recurse);
                if (state.short_pressed || state.long_pressed || state.recurse) {
                    await this.selectTreeListRow(tree_btn);
                } else {
                    await this.deselectTreeListRow(tree_btn);
                }
            }

            const dirs_btn = this.container_elem.querySelector(`.extra-network-dirs-view-button[data-div-id="${div_id}"]`);
            const dirs_btn_exists = isElement(dirs_btn) && !isNullOrUndefined(dirs_btn.offsetParent);
            if (dirs_btn_exists) {
                dirs_btn.classList.toggle("short-pressed", state.short_pressed);
                dirs_btn.classList.toggle("long-pressed", state.long_pressed);
                dirs_btn.classList.toggle("recurse", state.recurse);
                if (state.short_pressed || state.long_pressed || state.recurse) {
                    this.selectDirsViewButton(dirs_btn);
                } else {
                    this.deselectDirsViewButton(dirs_btn);
                }
            }
        }
        this.applyDirectoryFilters();
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
    const elem = tab.card_list.content_elem.querySelector(`.card[data-name="${name}"]`);
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

function extraNetworksControlSearchClearOnClick(event) {
    /** Dispatches custom event when the `clear` button in a search input is clicked. */
    const btn = event.target.closest(".extra-network-control--search-clear");
    const controls = btn.closest(".extra-network-controls");
    extra_networks_tabs[controls.dataset.tabnameFull].updateSearch("");
}

function extraNetworksControlClearFiltersOnClick(event) {
    /** Clears all directory filters applied to the cards list. */
    const btn = event.target.closest(".extra-network-control--clear-filters");
    const controls = btn.closest(".extra-network-controls");
    const tab = extra_networks_tabs[controls.dataset.tabnameFull];

    tab.clearSelectedButtons();
    tab.clearDirectoryFilters();
    tab.applyDirectoryFilters();
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

    const other_btn = tab.controls_elem.querySelector(
        `.extra-network-control--sort-dir[data-sort-dir="${sort_dir_str}"]`
    );
    delete btn.dataset.selected;
    other_btn.dataset.selected = "";

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
    btn.toggleAttribute("data-selected");
    tab.tree_view_en = "selected" in btn.dataset;

    const div_tree = tab.container_elem.querySelector(".extra-network-content--tree-view");
    div_tree.classList.toggle("hidden", !tab.tree_view_en);
    tab.tree_list.enable(tab.tree_view_en);

    // Apply the resize-handle-hidden class to the resize-handle-row.
    // NOTE: This can be simplified using only css with the ":has" selector however
    // this is only recently supported in firefox. So for now we just add a class
    // to the resize-handle-row instead.
    const resize_handle_row = div_tree.closest(".resize-handle-row");
    resize_handle_row.classList.toggle("resize-handle-hidden", !tab.card_view_en || !tab.tree_view_en);

    if (tab.tree_view_en && !tab.card_view_en) {
        div_tree.style.flexGrow = 1;
    } else {
        div_tree.style.flexGrow = null;
    }

    // If the tree list hasn't loaded yet, we need to force it to load.
    // This can happen if tree view is disabled by default or before refresh.
    // Then after refresh, enabling the tree view will require a load.
    if (tab.tree_view_en && !tab.tree_list.initial_load) {
        await tab.tree_list.load();
    }

    tab.applyListButtonStates();
}

function extraNetworksControlDirsViewOnClick(event) {
    /** Handles `onclick` events for the Dirs View button.
     *
     * Toggles the directory view in the extra networks pane.
     */
    const btn = event.target.closest(".extra-network-control--dirs-view");
    const controls = btn.closest(".extra-network-controls");
    const tab = extra_networks_tabs[controls.dataset.tabnameFull];
    btn.toggleAttribute("data-selected");
    tab.dirs_view_en = "selected" in btn.dataset;

    tab.container_elem.querySelector(
        ".extra-network-content--dirs-view"
    ).classList.toggle("hidden", !tab.dirs_view_en);

    tab.applyListButtonStates();
}

async function extraNetworksControlCardViewOnClick(event) {
    /** Handles `onclick` events for the Card View button.
     *
     * Toggles the card view in the extra networks pane.
     */
    const btn = event.target.closest(".extra-network-control--card-view");
    const controls = btn.closest(".extra-network-controls");
    const tab = extra_networks_tabs[controls.dataset.tabnameFull];
    btn.toggleAttribute("data-selected");
    tab.card_view_en = "selected" in btn.dataset;

    const div_tree = tab.container_elem.querySelector(".extra-network-content--tree-view");
    const div_card = tab.container_elem.querySelector(".extra-network-content--card-view");
    div_card.classList.toggle("hidden", !tab.card_view_en);
    tab.card_list.enable(tab.card_view_en);

    // Apply the resize-handle-hidden class to the resize-handle-row.
    // NOTE: This can be simplified using only css with the ":has" selector however
    // this is only recently supported in firefox. So for now we just add a class
    // to the resize-handle-row instead.
    const resize_handle_row = div_card.closest(".resize-handle-row");
    resize_handle_row.classList.toggle("resize-handle-hidden", !tab.card_view_en || !tab.tree_view_en);

    if (tab.tree_view_en && !tab.card_view_en) {
        div_tree.style.flexGrow = 1;
    } else {
        div_tree.style.flexGrow = null;
    }

    // If the tree list hasn't loaded yet, we need to force it to load.
    // This can happen if tree view is disabled by default or before refresh.
    // Then after refresh, enabling the tree view will require a load.
    if (tab.card_view_en && !tab.card_list.initial_load) {
        await tab.card_list.load();
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
        // Bit of lazy workaround to allow for us to create copies of the refresh
        // button anywhere we want as long as we include tabnameFull in the dataset.
        let tabname_full;
        if (isElement(controls)) {
            tabname_full = controls.dataset.tabnameFull;
        } else {
            tabname_full = btn.dataset.tabnameFull;
        }
        const tab = extra_networks_tabs[tabname_full];
        if (isNullOrUndefined(tab)) {
            return;
        }

        tab.updateSplashState({card_list_state: "loading", tree_list_state: "loading"});
        // We want to reset tab lists on refresh click so that the viewing area
        // shows that it is loading new data.
        tab.tree_list.clear();
        tab.card_list.clear();
        // Fire an event for this button click.
        gradioApp().getElementById(
            `${tabname_full}_extra_refresh_internal`
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

// ==== TREE VIEW DIRECTORY ROW CLICK EVENTS ====
async function extraNetworksTreeDirectoryRowClickHandler({event, deselect_others} = {}) {
    const btn = event.target.closest(".tree-list-item");
    const pane = btn.closest(".extra-network-pane");
    const tab = extra_networks_tabs[pane.dataset.tabnameFull];
    if (deselect_others === true) {
        await tab.clearSelectedButtons({excluded_div_ids: [btn.dataset.divId]});
    }
    if ("selected" in btn.dataset) {
        await tab.selectTreeListRow(btn);
    } else {
        await tab.deselectTreeListRow(btn);
    }
    tab.applyListButtonStates();
}

function extraNetworksTreeDirectoryOnClick(event) {
    /** Selects the clicked row and applies a recursive filter for the directory.
     *  Deselects all other rows.
    */
    extraNetworksTreeDirectoryRowClickHandler({event: event, deselect_others: true});
}

function extraNetworksTreeDirectoryOnCtrlClick(event) {
    /** Selects the clicked row. Does not deselect other rows. */
    extraNetworksTreeDirectoryRowClickHandler({event: event, deselect_others: false});
}

function extraNetworksTreeDirectoryOnLongPress(event) {
    /** Selects the clicked row. Deselects all other rows. */
    extraNetworksTreeDirectoryRowClickHandler({event: event, deselect_others: true});
}

function extraNetworksTreeDirectoryOnLongCtrlPress(event) {
    /** Selects the clicked row and applies a recursive filter for the directory.
     *  Does not deselect other rows.
    */
    extraNetworksTreeDirectoryRowClickHandler({event: event, deselect_others: false});
}

async function extraNetworksTreeDirectoryOnDblClick(event) {
    /** Expands the selected directory. */
    const btn = event.target.closest(".tree-list-item");
    const pane = btn.closest(".extra-network-pane");
    const tab = extra_networks_tabs[pane.dataset.tabnameFull];

    await tab.tree_list.toggleRowExpanded(btn.dataset.divId);
    tab.refreshTreeListRecursionDepth();

    //await tab.clearSelectedButtons();
    tab.applyListButtonStates();
}

// ==== TREE VIEW DIRECTORY ROW CHEVRON CLICK EVENTS ====
async function extraNetworksBtnTreeViewChevronOnClick(event) {
    const btn = event.target.closest(".tree-list-item");
    const pane = btn.closest(".extra-network-pane");
    const tab = extra_networks_tabs[pane.dataset.tabnameFull];

    await tab.tree_list.toggleRowExpanded(btn.dataset.divId);
    tab.refreshTreeListRecursionDepth();

    tab.applyListButtonStates();
}

async function extraNetworksBtnTreeViewChevronOnCtrlClick(event) {
    const btn = event.target.closest(".tree-list-item");
    const pane = btn.closest(".extra-network-pane");
    const tab = extra_networks_tabs[pane.dataset.tabnameFull];
    await tab.tree_list.expandAllRows(btn.dataset.divId);

    tab.applyListButtonStates();
}

async function extraNetworksBtnTreeViewChevronOnShiftClick(event) {
    const btn = event.target.closest(".tree-list-item");
    const pane = btn.closest(".extra-network-pane");
    const tab = extra_networks_tabs[pane.dataset.tabnameFull];
    await tab.tree_list.collapseAllRows(btn.dataset.divId);

    tab.applyListButtonStates();
}

// ==== DIRS VIEW BUTTON CLICK EVENTS ====
async function extraNetworksBtnDirsViewItemClickHandler({event, deselect_others} = {}) {
    const btn = event.target.closest(".extra-network-dirs-view-button");
    const pane = btn.closest(".extra-network-pane");
    const tab = extra_networks_tabs[pane.dataset.tabnameFull];
    if (deselect_others === true) {
        await tab.clearSelectedButtons({excluded_div_ids: [btn.dataset.divId]});
    }
    if ("selected" in btn.dataset) {
        tab.selectDirsViewButton(btn);
    } else {
        tab.deselectDirsViewButton(btn);
    }
    tab.applyListButtonStates();
}

function extraNetworksBtnDirsViewItemOnClick(event) {
    /** Selects the clicked button. Deselects all other buttons. */
    extraNetworksBtnDirsViewItemClickHandler({event: event, deselect_others: true});
}

function extraNetworksBtnDirsViewItemOnCtrlClick(event) {
    /** Selects the clicked button. Does not deselect other buttons. */
    extraNetworksBtnDirsViewItemClickHandler({event: event, deselect_others: false});
}

function extraNetworksBtnDirsViewItemOnLongPress(event) {
    /** Selects the clicked button and applies a recursive filter for the directory.
     *  Deselects all other buttons.
    */
    extraNetworksBtnDirsViewItemClickHandler({event: event, deselect_others: true});
}

function extraNetworksBtnDirsViewItemOnLongCtrlPress(event) {
    /** Selects the clicked button and applies a recursive filter for the directory.
     *  Does not deselect other buttons.
    */
    extraNetworksBtnDirsViewItemClickHandler({event: event, deselect_others: false});
}


// ==== OTHER BUTTON HANDLERS ====
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

function extraNetworksOnOptionsChanged() {
    initialUiOptionsLoaded.state = true;

    try {
        const keys = Object.keys(opts).filter(k => k.startsWith("extra_networks_"));
        const changes = {};
        for (const k of keys) {
            if (!(k in opts) || isNullOrUndefined(opts[k])) {
                continue;
            }
            if (!(k in extra_networks_curr_options)) {
                extra_networks_curr_options[k] = opts[k];
            } else {
                if (extra_networks_curr_options[k] !== opts[k]) {
                    changes[k] = extra_networks_curr_options[k];
                }
                extra_networks_curr_options[k] = opts[k];
            }
        }
    } catch (error) {
        console.warn("Error parsing options:", error);
    }
}

function extraNetworksSetupEventDelegators() {
    /** Sets up event delegators for all extraNetworks tabs.
     *
     *  These event handlers are not tied to any specific elements on the page.
     *  We do this because elements within each tab may be removed and replaced
     *  which would break references to elements in DOM and thus prevent any event
     *  listeners from firing.
     */
    let long_press_timer;
    let long_press_time_ms = opts.extra_networks_long_press_time_ms;
    if (long_press_time_ms < 0) {
        long_press_time_ms = 0;
    }
    let dbl_press_timer;
    let dbl_press_time_ms = opts.extra_networks_dbl_press_time_ms;
    if (dbl_press_time_ms < 0) {
        dbl_press_time_ms = 0;
    }

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
        ".extra-network-control--clear-filters": extraNetworksControlClearFiltersOnClick,
        ".extra-network-control--sort-mode": extraNetworksControlSortModeOnClick,
        ".extra-network-control--sort-dir": extraNetworksControlSortDirOnClick,
        ".extra-network-control--dirs-view": extraNetworksControlDirsViewOnClick,
        ".extra-network-control--tree-view": extraNetworksControlTreeViewOnClick,
        ".extra-network-control--card-view": extraNetworksControlCardViewOnClick,
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

    const short_ctrl_press_event_map = [
        {
            selector: ".tree-list-item-action--chevron",
            handler: extraNetworksBtnTreeViewChevronOnCtrlClick,
        },
        {
            selector: ".tree-list-item--dir",
            negative: ".tree-list-item-action",
            handler: extraNetworksTreeDirectoryOnCtrlClick,
        },
        {
            selector: ".extra-network-dirs-view-button",
            handler: extraNetworksBtnDirsViewItemOnCtrlClick,
        },
    ];

    const short_shift_press_event_map = [
        {
            selector: ".tree-list-item-action--chevron",
            handler: extraNetworksBtnTreeViewChevronOnShiftClick,
        },
    ];

    const long_press_event_map = [
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

    const long_ctrl_press_event_map = [
        {
            selector: ".tree-list-item--dir",
            negative: ".tree-list-item-action",
            handler: extraNetworksTreeDirectoryOnLongCtrlPress,
        },
        {
            selector: ".extra-network-dirs-view-button",
            handler: extraNetworksBtnDirsViewItemOnLongCtrlPress,
        },
    ];

    const dbl_press_event_map = [
        {
            selector: ".tree-list-item--dir",
            negative: ".tree-list-item-action",
            handler: extraNetworksTreeDirectoryOnDblClick,
        },
    ];

    const _on_short_press = (event, elem, handler) => {
        if (!handler) {
            return;
        }
        // Toggle
        if (elem.classList.contains("long-pressed")) {
            elem.classList.remove("long-pressed");
            elem.classList.remove("short-pressed");
            delete elem.dataset.selected;
        } else {
            elem.classList.toggle("short-pressed");
            elem.toggleAttribute("data-selected");
        }

        elem.dispatchEvent(new Event("shortpress", event));
        handler(event);
    };

    const _on_long_press = (event, elem, handler) => {
        if (!handler) {
            return;
        }
        // If long pressed, we deselect.
        // Else we set as long pressed.
        if (elem.classList.contains("long-pressed")) {
            elem.classList.remove("long-pressed");
            // Don't want to remove selected state if btn was previously short-pressed.
            if (!elem.classList.contains("short-pressed")) {
                delete elem.dataset.selected;
            }
        } else {
            elem.classList.toggle("long-pressed");
            elem.dataset.selected = "";
        }

        elem.dispatchEvent(new Event("longpress", event));
        handler(event);
    };

    const _on_dbl_press = (event, elem, handler) => {
        if (!handler) {
            return;
        }
        handler(event);
    };

    const _get_elem_from_event_map = (event, event_map) => {
        for (const obj of event_map) {
            const elem = event.target.closest(obj.selector);
            const neg = obj.negative ? event.target.closest(obj.negative) : null;
            if (isElement(elem) && !neg) {
                return {target: elem, handler: obj.handler};
            }
        }
        return null;
    };

    const _get_event_map = ({event, default_map, ctrl_map, shift_map, ctrl_shift_map} = {}) => {
        if (!isNullOrUndefined(ctrl_shift_map) && event.ctrlKey && event.shiftKey) {
            return ctrl_shift_map;
        } else if (!isNullOrUndefined(ctrl_map) && event.ctrlKey) {
            return ctrl_map;
        } else if (!isNullOrUndefined(shift_map) && event.shiftKey) {
            return shift_map;
        } else {
            return default_map;
        }
    };

    const _get_short_press_event_elem = (event) => {
        const event_map = _get_event_map({
            event: event,
            default_map: short_press_event_map,
            ctrl_map: short_ctrl_press_event_map,
            shift_map: short_shift_press_event_map,
        });
        return _get_elem_from_event_map(event, event_map);
    };

    const _get_long_press_event_elem = (event) => {
        const event_map = _get_event_map({
            event: event,
            default_map: long_press_event_map,
            ctrl_map: long_ctrl_press_event_map,
        });
        return _get_elem_from_event_map(event, event_map);
    };

    const _get_dbl_press_event_elem = (event) => {
        const event_map = _get_event_map({
            event: event,
            default_map: dbl_press_event_map,
        });
        return _get_elem_from_event_map(event, event_map);
    };

    window.addEventListener("pointerdown", event => {
        clearTimeout(long_press_timer);
        long_press_timer = null;

        if (event.target.hasPointerCapture(event.pointerId)) {
            event.target.releasePointerCapture(event.pointerId);
        }

        if (event.pointerType === "mouse" && event.button !== 0) {
            return;
        }

        const short_press_res = _get_short_press_event_elem(event);
        if (!isNullOrUndefined(short_press_res)) {
            event.stopPropagation();
            short_press_res.target.classList.add("pressed");
        }

        const long_press_res = _get_long_press_event_elem(event);
        if (!isNullOrUndefined(long_press_res)) {
            event.stopPropagation();
            long_press_res.target.classList.add("pressed");
            long_press_timer = setTimeout(() => {
                long_press_res.target.classList.remove("pressed");
                long_press_timer = null;
                _on_long_press(event, long_press_res.target, long_press_res.handler);
            }, long_press_time_ms);
        }
    });

    window.addEventListener("pointerup", event => {
        clearTimeout(long_press_timer);
        long_press_timer = null;

        // Shortcut to avoid lookup of element if no parent is pressed.
        if (!isElement(event.target.closest(".pressed"))) {
            return;
        }

        const short_press_res = _get_short_press_event_elem(event);
        if (!isNullOrUndefined(short_press_res)) {
            event.stopPropagation();
            if (short_press_res.target.classList.contains("pressed")) {
                short_press_res.target.classList.remove("pressed");
                _on_short_press(event, short_press_res.target, short_press_res.handler);
            }
        }

        const long_press_res = _get_short_press_event_elem(event);
        if (!isNullOrUndefined(long_press_res)) {
            event.stopPropagation();
            long_press_res.target.classList.remove("pressed");
        }

        const dbl_press_res = _get_dbl_press_event_elem(event);
        if (!isNullOrUndefined(dbl_press_res)) {
            if (isNullOrUndefined(dbl_press_timer)) {
                dbl_press_res.target.dataset.awaitDblClick = "";
                dbl_press_timer = setTimeout(() => {
                    dbl_press_timer = null;
                    delete dbl_press_res.target.dataset.awaitDblClick;
                }, dbl_press_time_ms);
            } else if ("awaitDblClick" in dbl_press_res.target.dataset) {
                // Only count as dbl click if it is the same element.
                clearTimeout(dbl_press_timer);
                dbl_press_timer = null;
                delete dbl_press_res.target.dataset.awaitDblClick;
                _on_dbl_press(event, dbl_press_res.target, dbl_press_res.handler);
            }
        }
    });

    window.addEventListener("pointerout", event => {
        clearTimeout(long_press_timer);
        long_press_timer = null;
        clearTimeout(dbl_press_timer);
        dbl_press_timer = null;

        // Shortcut to avoid lookup of element if no parent is pressed.
        if (!isElement(event.target.closest(".pressed"))) {
            return;
        }

        const short_press_res = _get_short_press_event_elem(event);
        if (!isNullOrUndefined(short_press_res)) {
            event.stopPropagation();
            short_press_res.target.classList.remove("pressed");
        }

        const long_press_res = _get_long_press_event_elem(event);
        if (!isNullOrUndefined(long_press_res)) {
            event.stopPropagation();
            long_press_res.target.classList.remove("pressed");
        }

        const dbl_press_res = _get_dbl_press_event_elem(event);
        if (!isNullOrUndefined(dbl_press_res)) {
            event.stopPropagation();
            delete dbl_press_res.target.dataset.awaitDblClick;
        }
    });

    // Disable the context menu for long press items. On mobile, long press causes
    // context menu to appear which we don't want.
    window.oncontextmenu = event => {
        const long_press_res = _get_long_press_event_elem(event);
        if (!isNullOrUndefined(long_press_res)) {
            return false;
        }
    };
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
        const tab = new ExtraNetworksTab({
            tabname: tabname,
            extra_networks_tabname: extra_networks_tabname,
        });
        await tab.setup(pane, controls_div);
        extra_networks_tabs[tabname_full] = tab;
    }
}

async function extraNetworksSetup() {
    await waitForBool(initialUiOptionsLoaded);
    // FIXME 1s delay added to allow for the rest of the UI to load before we setup
    // our tabs. This is a workaround to prevent the user from switching tabs too quick
    // after reloading the UI. If user switches tabs immediately after reloading the webpage
    // then the tab.load() will run before this setup runs. No clue how this is even possible
    // since we are waiting for setup to complete before loading so something weird is
    // happening in the background.
    setTimeout(async() => {
        await Promise.all([
            extraNetworksSetupTab('txt2img'),
            extraNetworksSetupTab('img2img'),
        ]);
        extraNetworksSetupEventDelegators();
    }, 1000);
}

window.addEventListener("keydown", (event) => {
    if (event.repeat) {
        return;
    }

    if (event.ctrlKey) {
        document.body.classList.add("ctrl-pressed");
    }
    if (event.shiftKey) {
        document.body.classList.add("shift-pressed");
    }
    if (event.altKey) {
        document.body.classList.add("alt-pressed");
    }
});

window.addEventListener("keyup", (event) => {
    if (event.repeat) {
        return;
    }

    if (!event.ctrlKey) {
        document.body.classList.remove("ctrl-pressed");
    }
    if (!event.shiftKey) {
        document.body.classList.remove("shift-pressed");
    }
    if (!event.altKey) {
        document.body.classList.remove("alt-pressed");
    }
});

onUiLoaded(extraNetworksSetup);
onOptionsChanged(extraNetworksOnOptionsChanged);
