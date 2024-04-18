// Prevent eslint errors on functions defined in other files.
/*global
    ExtraNetworksClusterizeTreeList,
    ExtraNetworksClusterizeCardsList,
    waitForElement,
    isElement,
    isElementThrowError,
    requestGetPromise,
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
const EXTRA_NETWORKS_GET_PAGE_READY_MAX_ATTEMPTS = 10;
const EXTRA_NETWORKS_WAIT_FOR_PAGE_READY_TIMEOUT_MS = 1000;
const EXTRA_NETWORKS_REQUEST_GET_TIMEOUT_MS = 1000;
const EXTRA_NETWORKS_REFRESH_INTERNAL_DEBOUNCE_TIMEOUT_MS = 200;

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
    show_prompt = true;
    show_neg_prompt = true;
    compact_prompt_en = false;
    refresh_in_progress = false;
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

        // determine whether compact prompt mode is enabled.
        // cannot await this since it may not exist on page depending on user setting.
        this.compact_prompt_en = isElement(gradioApp().querySelector(".toprow-compact-tools"));

        // setup this tab's controls
        this.controls_elem.id = `${this.tabname_full}_controls`;
        controls_div.insertBefore(this.controls_elem, null);

        await Promise.all([this.setupTreeList(), this.setupCardsList()]);

        const sort_mode_elem = this.controls_elem.querySelector(".extra-network-control--sort-mode[data-selected='']");
        isElementThrowError(sort_mode_elem);
        const sort_dir_elem = this.controls_elem.querySelector(".extra-network-control--sort-dir");
        isElementThrowError(sort_dir_elem);

        this.setSortMode(sort_mode_elem.dataset.sortMode);
        this.setSortDir(sort_dir_elem.dataset.sortDir);
        this.setFilterStr(this.txt_search_elem.value.toLowerCase());

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

    refreshSingleCard(name) {
        requestGet(
            "./sd_extra_networks/get-single-card",
            {
                tabname: this.tabname,
                extra_networks_tabname: this.extra_networks_tabname,
                name: name,
            },
            (data) => {
                if (data && data.html) {
                    this.cards_list.updateCard(name, data.html);
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
        const btn_dirs_view = this.controls_elem.querySelector(".extra-network-control--dirs-view");
        const btn_tree_view = this.controls_elem.querySelector(".extra-network-control--tree-view");
        const div_dirs = this.container_elem.querySelector(".extra-network-content--dirs-view");
        // We actually want to select the tree view's column in the resize-handle-row.
        // This is what we actually show/hide, not the inner elements.
        const div_tree = this.tree_list.scroll_elem.closest(".resize-handle-col");

        // Remove "hidden" class if button is enabled, otherwise add it.
        div_dirs.classList.toggle("hidden", !("selected" in btn_dirs_view.dataset));
        div_tree.classList.toggle("hidden", !("selected" in btn_tree_view.dataset));

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
        this.setFilterStr(this.filter_str);
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

    applyFilter() {
        // We only want to filter/sort the cards list.
        this.setFilterStr(this.txt_search_elem.value.toLowerCase());

        // If the search input has changed since selecting a button to populate it
        // then we want to disable the button that previously populated the search input.

        // tree view buttons
        let btn = this.container_elem.querySelector(".tree-list-item[data-selected='']");
        if (isElement(btn) && btn.dataset.path !== this.txt_search_elem.value && "selected" in btn.dataset) {
            this.tree_list.onRowSelected(btn.dataset.divId, btn, false);
        }
        // dirs view buttons
        btn = this.container_elem.querySelector(".extra-network-dirs-view-button[data-selected='']");
        if (isElement(btn) && btn.textContent.trim() !== this.txt_search_elem.value) {
            delete btn.dataset.selected;
        }
    }

    async waitForServerPageReady(max_attempts = EXTRA_NETWORKS_GET_PAGE_READY_MAX_ATTEMPTS) {
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
        const err_prefix = `error waiting for server page (${this.tabname_full})`;
        return new Promise((resolve, reject) => {
            let attempt = 0;
            const loop = () => {
                setTimeout(async() => {
                    try {
                        await requestGetPromise(
                            "./sd_extra_networks/page-is-ready",
                            {extra_networks_tabname: this.extra_networks_tabname},
                            EXTRA_NETWORKS_WAIT_FOR_PAGE_READY_TIMEOUT_MS,
                        );
                        return resolve();
                    } catch (error) {
                        // If we get anything other than a timeout error, reject.
                        // Otherwise, fall through to retry request.
                        if (error.status !== 408) {
                            return reject(`${err_prefix}: uncaught exception: ${JSON.stringify(error)}`);
                        }
                    }

                    if (max_attempts !== 0 && attempt++ >= max_attempts) {
                        return reject(`${err_prefix}: max attempts exceeded`);
                    } else {
                        // small delay since our request has a timeout.
                        setTimeout(() => loop(), 100);
                    }
                }, 0);
            };
            return loop();
        });
    }

    async onInitCardsData() {
        try {
            await this.waitForServerPageReady();
        } catch (error) {
            console.error(JSON.stringify(error));
            return {};
        }

        const url = "./sd_extra_networks/init-cards-data";
        const payload = {tabname: this.tabname, extra_networks_tabname: this.extra_networks_tabname};
        const timeout = EXTRA_NETWORKS_REQUEST_GET_TIMEOUT_MS;
        try {
            const response = await requestGetPromise(url, payload, timeout);
            return response.response;
        } catch (error) {
            console.error(JSON.stringify(error));
            return {};
        }
    }

    async onInitTreeData() {
        try {
            await this.waitForServerPageReady();
        } catch (error) {
            console.error(JSON.stringify(error));
            return {};
        }

        const url = "./sd_extra_networks/init-tree-data";
        const payload = {tabname: this.tabname, extra_networks_tabname: this.extra_networks_tabname};
        const timeout = EXTRA_NETWORKS_REQUEST_GET_TIMEOUT_MS;
        try {
            const response = await requestGetPromise(url, payload, timeout);
            return response.response;
        } catch (error) {
            console.error(JSON.stringify(error));
            return {};
        }
    }

    async onFetchCardsData(div_ids) {
        try {
            await this.waitForServerPageReady();
        } catch (error) {
            console.error(JSON.stringify(error));
            return {};
        }

        const url = "./sd_extra_networks/fetch-cards-data";
        const payload = {extra_networks_tabname: this.extra_networks_tabname, div_ids: div_ids};
        const timeout = EXTRA_NETWORKS_REQUEST_GET_TIMEOUT_MS;
        try {
            const response = await requestGetPromise(url, payload, timeout);
            return response.response;
        } catch (error) {
            console.error(JSON.stringify(error));
            return {};
        }
    }

    async onFetchTreeData(div_ids) {
        try {
            await this.waitForServerPageReady();
        } catch (error) {
            console.error(JSON.stringify(error));
            return {};
        }

        const url = "./sd_extra_networks/fetch-tree-data";
        const payload = {extra_networks_tabname: this.extra_networks_tabname, div_ids: div_ids};
        const timeout = EXTRA_NETWORKS_REQUEST_GET_TIMEOUT_MS;
        try {
            const response = await requestGetPromise(url, payload, timeout);
            return response.response;
        } catch (error) {
            console.error(JSON.stringify(error));
            return {};
        }
    }

    updateSearch(text) {
        this.txt_search_elem.value = text;
        updateInput(this.txt_search_elem);
        this.applyFilter();
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
}

//

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
    tab.refreshSingleCard(name);
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

function extraNetworksBtnDirsViewItemOnClick(event, tabname_full) {
    /** Handles `onclick` events for buttons in the directory view. */
    const tab = extra_networks_tabs[tabname_full];
    const container_elem = tab.container_elem;
    const txt_search_elem = tab.txt_search_elem;

    const _deselect_all_buttons = () => {
        container_elem.querySelectorAll(
            ".extra-network-dirs-view-button[data-selected='']"
        ).forEach(elem => {
            delete elem.dataset.selected;
        });
        // deselect tree view rows
        tab.tree_list.onRowSelected(); // empty params deselects all rows.
    };

    const _select_button = (elem) => {
        _deselect_all_buttons();
        // update search input with selected button's path.
        elem.dataset.selected = "";
        txt_search_elem.value = elem.textContent.trim();

        // Select the corresponding tree view button.
        if ("selected" in elem.dataset) {
            const tree_row = tab.container_elem.querySelector(`.tree-list-item[data-path="${elem.textContent.trim()}"]`);
            if (isElement(tree_row)) {
                tab.tree_list.onRowSelected(tree_row.dataset.divId, tree_row);
            }
        }
    };

    const _deselect_button = (elem) => {
        delete elem.dataset.selected;
        txt_search_elem.value = "";
        // deselect tree view rows
        tab.tree_list.onRowSelected(); // empty params deselects all rows.
    };

    if ("selected" in event.target.dataset) {
        _deselect_button(event.target);
    } else {
        _select_button(event.target);
    }

    updateInput(txt_search_elem);
    tab.applyFilter();

    event.stopPropagation();
}

function extraNetworksControlSearchClearOnClick(event, tabname_full) {
    /** Dispatches custom event when the `clear` button in a search input is clicked. */
    const txt_search_elem = extra_networks_tabs[tabname_full].txt_search_elem;
    txt_search_elem.value = "";
    txt_search_elem.dispatchEvent(
        new CustomEvent(
            "extra-network-control--search-clear",
            {bubbles: true, detail: {tabname_full: tabname_full}},
        )
    );

    event.stopPropagation();
}

function extraNetworksControlSortModeOnClick(event, tabname_full) {
    /** Handles `onclick` events for Sort Mode buttons. */
    const tab = extra_networks_tabs[tabname_full];
    tab.controls_elem.querySelectorAll(".extra-network-control--sort-mode").forEach(elem => {
        delete elem.dataset.selected;
    });

    event.currentTarget.dataset.selected = "";

    const sort_mode_str = event.currentTarget.dataset.sortMode.toLowerCase();

    tab.setSortMode(sort_mode_str);

    event.stopPropagation();
}

function extraNetworksControlSortDirOnClick(event, tabname_full) {
    /** Handles `onclick` events for the Sort Direction button.
     *
     * Modifies the data attributes of the Sort Direction button to cycle between
     * ascending and descending sort directions.
     */
    const tab = extra_networks_tabs[tabname_full];

    const curr_sort_dir_str = event.currentTarget.dataset.sortDir.toLowerCase();
    if (!["ascending", "descending"].includes(curr_sort_dir_str)) {
        console.error(`Invalid sort_dir_str: ${curr_sort_dir_str}`);
        return;
    }

    let sort_dir_str = curr_sort_dir_str === "ascending" ? "descending" : "ascending";
    event.currentTarget.dataset.sortDir = sort_dir_str;
    event.currentTarget.setAttribute("title", `Sort ${sort_dir_str}`);

    tab.setSortDir(sort_dir_str);

    event.stopPropagation();
}

function extraNetworksControlTreeViewOnClick(event, tabname_full) {
    /** Handles `onclick` events for the Tree View button.
     *
     * Toggles the tree view in the extra networks pane.
     */
    let show;
    if ("selected" in event.currentTarget.dataset) {
        delete event.currentTarget.dataset.selected;
        show = false;
    } else {
        event.currentTarget.dataset.selected = "";
        show = true;
    }

    const tab = extra_networks_tabs[tabname_full];
    tab.tree_list.scroll_elem.parentElement.classList.toggle("hidden", !show);
    tab.tree_list.enable(show);

    // Apply the resize-handle-hidden class to the resize-handle-row.
    // NOTE: This can be simplified using only css with the ":has" selector however
    // this is only recently supported in firefox. So for now we just add a class
    // to the resize-handle-row instead.
    const resize_handle_row = tab.tree_list.scroll_elem.closest(".resize-handle-row");
    resize_handle_row.classList.toggle("resize-handle-hidden", !show);

    event.stopPropagation();
}

function extraNetworksControlDirsViewOnClick(event, tabname_full) {
    /** Handles `onclick` events for the Dirs View button.
     *
     * Toggles the directory view in the extra networks pane.
     */
    const show = !("selected" in event.currentTarget.dataset);
    if (show) {
        event.currentTarget.dataset.selected = "";
    } else {
        delete event.currentTarget.dataset.selected;
    }

    const tab = extra_networks_tabs[tabname_full];
    tab.container_elem.querySelector(
        ".extra-network-content--dirs-view"
    ).classList.toggle("hidden", !show);

    event.stopPropagation();
}

function extraNetworksControlRefreshOnClick(event, tabname_full) {
    /** Handles `onclick` events for the Refresh Page button.
     *
     * In order to actually call the python functions in `ui_extra_networks.py`
     * to refresh the page, we created an empty gradio button in that file with an
     * event handler that refreshes the page. So what this function here does
     * is it manually raises a `click` event on that button.
     */
    clearTimeout(extra_networks_refresh_internal_debounce_timer);
    extra_networks_refresh_internal_debounce_timer = setTimeout(() => {
        // We want to reset tab lists on refresh click so that the viewing area
        // shows that it is loading new data.
        extra_networks_tabs[tabname_full].tree_list.clear();
        extra_networks_tabs[tabname_full].cards_list.clear();
        // Fire an event for this button click.
        gradioApp().getElementById(`${tabname_full}_extra_refresh_internal`).dispatchEvent(new Event("click"));
    }, EXTRA_NETWORKS_REFRESH_INTERNAL_DEBOUNCE_TIMEOUT_MS);

    event.stopPropagation();
}

function extraNetworksCardOnClick(event, tabname_full) {
    const elem = event.currentTarget;
    const tab = extra_networks_tabs[tabname_full];
    if ("negPrompt" in elem.dataset) {
        extraNetworksUpdatePrompt(tab.txt_prompt_elem, elem.dataset.prompt);
        extraNetworksUpdatePrompt(tab.txt_neg_prompt_elem, elem.dataset.negPrompt);
    } else if ("allowNeg" in elem.dataset) {
        extraNetworksUpdatePrompt(tab.active_prompt_elem, elem.dataset.prompt);
    } else {
        extraNetworksUpdatePrompt(tab.txt_prompt_elem, elem.dataset.prompt);
    }

    event.stopPropagation();
}

function extraNetworksTreeFileOnClick(event, btn, tabname_full) {
    return;
}

function extraNetworksTreeDirectoryOnClick(event, btn, tabname_full) {
    const true_targ = event.target;
    const div_id = btn.dataset.divId;

    const tab = extra_networks_tabs[tabname_full];

    if (true_targ.matches(".tree-list-item-action--leading, .tree-list-item-action-chevron")) {
        // If user clicks on the chevron, then we do not select the folder.
        const prev_selected_elem = gradioApp().querySelector(".tree-list-item[data-selected='']");
        tab.tree_list.onRowExpandClick(div_id, btn);
        const selected_elem = gradioApp().querySelector(".tree-list-item[data-selected='']");
        if (isElement(prev_selected_elem) && !isElement(selected_elem)) {
            // is a selected element was removed, clear filter.
            tab.updateSearch("");
        }
    } else {
        // user clicked anywhere else on the row
        tab.tree_list.onRowSelected(div_id, btn);
        // Select the corresponding dirs view button.
        if ("selected" in btn.dataset) {
            tab.container_elem.querySelectorAll(".extra-network-dirs-view-button").forEach(elem => {
                if (elem.textContent.trim() === btn.dataset.path) {
                    elem.dataset.selected = "";
                } else {
                    delete elem.dataset.selected;
                }
            });

        }
        tab.updateSearch("selected" in btn.dataset ? btn.dataset.path : "");
    }
}

function extraNetworksTreeOnClick(event, tabname_full) {
    const btn = event.target.closest(".tree-list-item");
    if (!isElementLogError(btn)) {
        return;
    }

    if (btn.dataset.treeEntryType === "file") {
        extraNetworksTreeFileOnClick(event, btn, tabname_full);
    } else {
        extraNetworksTreeDirectoryOnClick(event, btn, tabname_full);
    }

    event.stopPropagation();
}

function extraNetworksBtnShowMetadataOnClick(event, extra_networks_tabname, card_name) {
    extraNetworksFetchMetadata(extra_networks_tabname, card_name);
    event.stopPropagation();
}

function extraNetworksBtnEditMetadataOnClick(event, tabname_full, card_name) {
    const id = `${tabname_full}_edit_user_metadata`;
    let editor = extraPageUserMetadataEditors[id];
    if (isNullOrUndefined(editor)) {
        editor = {};
        editor.page = gradioApp().getElementById(id);
        editor.nameTextarea = gradioApp().querySelector(`#${id}_name textarea`);
        editor.button = gradioApp().querySelector(`#${id}_button`);
        extraPageUserMetadataEditors[id] = editor;
    }

    editor.nameTextarea.value = card_name;
    updateInput(editor.nameTextarea);

    editor.button.click();

    popup(editor.page);

    event.stopPropagation();
}

function extraNetworksBtnCopyPathOnClick(event, path) {
    copyToClipboard(path);
    event.stopPropagation();
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

    // Update search filter whenever the search input's clear button is pressed.
    window.addEventListener("extra-network-control--search-clear", event => {
        event.stopPropagation();
        extra_networks_tabs[event.detail.tabname_full].applyFilter();
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
