// Prevent eslint errors on functions defined in other files.
/*global
    ExtraNetworksClusterizeTreeList,
    ExtraNetworksClusterizeCardsList,
*/
/*eslint no-undef: "error"*/

const SEARCH_INPUT_DEBOUNCE_TIME_MS = 250;

const re_extranet = /<([^:^>]+:[^:]+):[\d.]+>(.*)/;
const re_extranet_g = /<([^:^>]+:[^:]+):[\d.]+>/g;
const re_extranet_neg = /\(([^:^>]+:[\d.]+)\)/;
const re_extranet_g_neg = /\(([^:^>]+:[\d.]+)\)/g;
const activePromptTextarea = {};
const clusterizers = {};
var globalPopup = null;
var globalPopupInner = null;
const storedPopupIds = {};
const extraPageUserMetadataEditors = {};
// A flag used by the `waitForBool` promise to determine when we first load Ui Options.
const initialUiOptionsLoaded = {state: false};

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

function extraNetworksClusterizersEnable(tabname_full) {
    for (const [_tabname_full, tab_clusterizers] of Object.entries(clusterizers)) {
        for (const v of Object.values(tab_clusterizers)) {
            v.enable(_tabname_full === tabname_full);
        }
    }
}

async function extraNetworksClusterizersLoadTab({
    tabname_full,
    selected = false,
    fetch_data = false,
}) {
    if (!keyExistsLogError(clusterizers, tabname_full)) {
        return;
    }

    if (selected) {
        extraNetworksClusterizersEnable(tabname_full);
    }

    for (const v of Object.values(clusterizers[tabname_full])) {
        await v.load(fetch_data);
        await v.refresh(true);
    }
}

function extraNetworksRegisterPromptForTab(tabname, id) {
    var textarea = gradioApp().querySelector(`#${id} > label > textarea`);

    if (!activePromptTextarea[tabname]) {
        activePromptTextarea[tabname] = textarea;
    }

    textarea.addEventListener("focus", function() {
        activePromptTextarea[tabname] = textarea;
    });
}

function extraNetworksMovePromptToTab(tabname, id, showPrompt, showNegativePrompt) {
    if (!gradioApp().querySelector('.toprow-compact-tools')) return; // only applicable for compact prompt layout

    var promptContainer = gradioApp().getElementById(`${tabname}_prompt_container`);
    var prompt = gradioApp().getElementById(`${tabname}_prompt_row`);
    var negPrompt = gradioApp().getElementById(`${tabname}_neg_prompt_row`);
    var elem = id ? gradioApp().getElementById(id) : null;

    if (showNegativePrompt && elem) {
        elem.insertBefore(negPrompt, elem.firstChild);
    } else {
        promptContainer.insertBefore(negPrompt, promptContainer.firstChild);
    }

    if (showPrompt && elem) {
        elem.insertBefore(prompt, elem.firstChild);
    } else {
        promptContainer.insertBefore(prompt, promptContainer.firstChild);
    }

    if (elem) {
        elem.classList.toggle('extra-page-prompts-active', showNegativePrompt || showPrompt);
    }
}

function extraNetworksShowControlsForPage(tabname, tabname_full) {
    gradioApp().querySelectorAll(`#${tabname}_extra_tabs .extra-network-controls-div > div`).forEach((elem) => {
        let show = `${tabname_full}_controls` === elem.id;
        elem.classList.toggle("hidden", !show);
    });
}

function extraNetworksRemoveFromPrompt(textarea, text, is_neg) {
    let match = text.match(is_neg ? re_extranet_neg : re_extranet);
    let replaced = false;
    let res;
    let prefix = opts.extra_networks_add_text_separator;

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
            if (postfix && res.slice(idx, postfix.length) === postfix) {
                res = res.slice(0, idx) + res.slice(idx + postfix.length);
            }
            if (res.slice(idx - prefix.length, prefix.length) === prefix) {
                res = res.slice(0, idx - prefix.length) + res.slice(idx);
            }
        }
    } else {
        res = textarea.value.replaceAll(new RegExp(`((?:${extraTextBeforeNet})?${text})`, "g"), "");
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
        textarea.value = textarea.value + opts.extra_networks_add_text_separator + text;
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
    requestGet(
        "./sd_extra_networks/get-single-card",
        {tabname: tabname, extra_networks_tabname: extra_networks_tabname, name: name},
        (data) => {
            if (data && data.html) {
                const card = gradioApp().querySelector(`${tabname}_${extra_networks_tabname}_cards > .card[data-name="${name}"]`);
                const new_div = document.createElement("div");
                new_div.innerHTML = data.html;
                const new_card = new_div.firstElementChild;

                new_card.style.display = "";
                card.parentElement.insertBefore(new_card, card);
                card.parentElement.removeChild(card);
            }
        },
    );
}

async function extraNetworksRefreshTab(tabname_full) {
    /** called from python when user clicks the extra networks refresh tab button */
    // Reapply controls since they don't change on refresh.
    const controls = gradioApp().getElementById(`${tabname_full}_controls`);
    let btn_dirs_view = controls.querySelector(".extra-network-control--dirs-view");
    let btn_tree_view = controls.querySelector(".extra-network-control--tree-view");

    const pane = gradioApp().getElementById(`${tabname_full}_pane`);
    let div_dirs = pane.querySelector(".extra-network-content--dirs-view");
    let div_tree = pane.querySelector(`.extra-network-content.resize-handle-col:has(> #${tabname_full}_tree_list_scroll_area)`);

    // Remove "hidden" class if button is enabled, otherwise add it.
    div_dirs.classList.toggle("hidden", !("selected" in btn_dirs_view.dataset));
    div_tree.classList.toggle("hidden", !("selected" in btn_tree_view.dataset));

    await waitForKeyInObject({k: tabname_full, obj: clusterizers});
    for (const _tabname_full of Object.keys(clusterizers)) {
        let selected = _tabname_full == tabname_full;
        await extraNetworksClusterizersLoadTab({
            tabname_full: _tabname_full,
            selected: selected,
            fetch_data: true,
        });
    }
}

function extraNetworksAutoSetTreeWidth(pane) {
    if (!isElementLogError(pane)) {
        return;
    }

    const tabname_full = pane.dataset.tabnameFull;

    // This event is only applied to the currently selected tab if has clusterize lists.
    if (!keyExists(clusterizers, tabname_full)) {
        return;
    }

    const row = pane.querySelector(".resize-handle-row");
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
    let max_width = clusterizers[tabname_full].tree_list.getMaxRowWidth();
    // Add the resize handle's padding to the result and default to minLeftColWidth if necessary.
    max_width = Math.max(max_width + pad, min_left_col_width);

    // Mimicks resizeHandle.js::setLeftColGridTemplate().
    row.style.gridTemplateColumns = `${max_width}px ${pad}px 1fr`;
}

function extraNetworksApplyFilter(tabname_full) {
    if (!keyExistsLogError(clusterizers, tabname_full)) {
        return;
    }

    const pane = gradioApp().getElementById(`${tabname_full}_pane`);
    if (!isElementLogError(pane)) {
        return;
    }

    const txt_search = gradioApp().querySelector(`#${tabname_full}_controls .extra-network-control--search-text`);
    if (!isElementLogError(txt_search)) {
        return;
    }

    // We only want to filter/sort the cards list.
    clusterizers[tabname_full].cards_list.setFilterStr(txt_search.value.toLowerCase());

    // If the search input has changed since selecting a button to populate it
    // then we want to disable the button that previously populated the search input.
    // tree view buttons
    let btn = pane.querySelector(".tree-list-item[data-selected='']");
    if (isElement(btn) && btn.dataset.path !== txt_search.value && "selected" in btn.dataset) {
        clusterizers[tabname_full].tree_list.onRowSelected(btn.dataset.divId, btn, false);
    }
    // dirs view buttons
    btn = pane.querySelector(".extra-network-dirs-view-button[data-selected='']");
    if (isElement(btn) && btn.textContent.trim() !== txt_search.value) {
        delete btn.dataset.selected;
    }
}

// ==== EVENT HANDLING ====

async function extraNetworksInitCardsData(tabname, extra_networks_tabname) {
    const res = await requestGetPromise(
        "./sd_extra_networks/init-cards-data",
        {
            tabname: tabname,
            extra_networks_tabname: extra_networks_tabname,
        },
    );
    return JSON.parse(res);
}

async function extraNetworksInitTreeData(tabname, extra_networks_tabname) {
    const res = await requestGetPromise(
        "./sd_extra_networks/init-tree-data",
        {
            tabname: tabname,
            extra_networks_tabname: extra_networks_tabname,
        },
    );
    return JSON.parse(res);
}

async function extraNetworksOnInitData(tabname, extra_networks_tabname, class_name) {
    if (class_name === "ExtraNetworksClusterizeTreeList") {
        return await extraNetworksInitTreeData(tabname, extra_networks_tabname);
    } else if (class_name === "ExtraNetworksClusterizeCardsList") {
        return await extraNetworksInitCardsData(tabname, extra_networks_tabname);
    }
}

async function extraNetworksFetchCardsData(extra_networks_tabname, div_ids) {
    const res = await requestGetPromise(
        "./sd_extra_networks/fetch-cards-data",
        {
            extra_networks_tabname: extra_networks_tabname,
            div_ids: div_ids,
        },
    );
    return JSON.parse(res);
}

async function extraNetworksFetchTreeData(extra_networks_tabname, div_ids) {
    const res = await requestGetPromise(
        "./sd_extra_networks/fetch-tree-data",
        {
            extra_networks_tabname: extra_networks_tabname,
            div_ids: div_ids,
        },
    );
    return JSON.parse(res);
}

async function extraNetworksOnFetchData(class_name, extra_networks_tabname, div_ids) {
    if (class_name === "ExtraNetworksClusterizeTreeList") {
        return await extraNetworksFetchTreeData(extra_networks_tabname, div_ids);
    } else if (class_name === "ExtraNetworksClusterizeCardsList") {
        return await extraNetworksFetchCardsData(extra_networks_tabname, div_ids);
    }
}

function extraNetworksFetchMetadata(extra_networks_tabname, card_name) {
    const _showError = () => { extraNetworksShowMetadata("there was an error getting metadata"); };

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
    extraNetworksMovePromptToTab(tabname, '', false, false);
    extraNetworksShowControlsForPage(tabname, null);
}

async function extraNetworksTabSelected(
    tabname,
    id,
    showPrompt,
    showNegativePrompt,
    tabname_full,
) {
    /** called from python when user selects an extra networks tab */
    extraNetworksMovePromptToTab(tabname, id, showPrompt, showNegativePrompt);
    extraNetworksShowControlsForPage(tabname, tabname_full);

    await waitForKeyInObject({k: tabname_full, obj: clusterizers});
    await extraNetworksClusterizersLoadTab({
        tabname_full: tabname_full,
        selected: true,
        fetch_data: false,
    });
}

function extraNetworksBtnDirsViewItemOnClick(event, tabname_full) {
    /** Handles `onclick` events for buttons in the directory view. */
    const txt_search = gradioApp().querySelector(`#${tabname_full}_controls .extra-network-control--search-text`);
    
    const _deselect_all_buttons = () => {
        gradioApp().querySelectorAll(".extra-network-dirs-view-button").forEach((elem) => {
            delete elem.dataset.selected;
        });
    };

    const _select_button = (elem) => {
        _deselect_all_buttons();
        // Update search input with select button's path.
        elem.dataset.selected = "";
        txt_search.value = elem.textContent.trim();
    };

    const _deselect_button = (elem) => {
        delete elem.dataset.selected;
        txt_search.value = "";
    };

    if ("selected" in event.target.dataset) {
        _deselect_button(event.target);
    } else {
        _select_button(event.target);
    }

    updateInput(txt_search);
    extraNetworksApplyFilter(tabname_full);
}

function extraNetworksControlSearchClearOnClick(event, tabname_full) {
    /** Dispatches custom event when the `clear` button in a search input is clicked. */
    let clear_btn = event.target.closest(".extra-network-control--search-clear");
    let txt_search = clear_btn.previousElementSibling;
    txt_search.value = "";
    txt_search.dispatchEvent(
        new CustomEvent(
            "extra-network-control--search-clear",
            {bubbles: true, detail: {tabname_full: tabname_full}},
        )
    );
}

function extraNetworksControlSortModeOnClick(event, tabname_full) {
    /** Handles `onclick` events for Sort Mode buttons. */
    event.currentTarget.parentElement.querySelectorAll('.extra-network-control--sort-mode').forEach(elem => {
        delete elem.dataset.selected;
    });

    event.currentTarget.dataset.selected = "";

    if (!keyExists(clusterizers, tabname_full)) {
        return;
    }

    const sort_mode_str = event.currentTarget.dataset.sortMode.toLowerCase();
    clusterizers[tabname_full].cards_list.setSortMode(sort_mode_str);
}

function extraNetworksControlSortDirOnClick(event, tabname_full) {
    /** Handles `onclick` events for the Sort Direction button.
     *
     * Modifies the data attributes of the Sort Direction button to cycle between
     * ascending and descending sort directions.
     */
    const curr_sort_dir_str = event.currentTarget.dataset.sortDir.toLowerCase();
    if (!["ascending", "descending"].includes(curr_sort_dir_str)) {
        console.error(`Invalid sort_dir_str: ${curr_sort_dir_str}`);
        return;
    }

    let sort_dir_str = curr_sort_dir_str === "ascending" ? "descending" : "ascending";
    event.currentTarget.dataset.sortDir = sort_dir_str;
    event.currentTarget.setAttribute("title", `Sort ${sort_dir_str}`);

    if (!keyExists(clusterizers, tabname_full)) {
        return;
    }

    clusterizers[tabname_full].cards_list.setSortDir(sort_dir_str);
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

    if (!keyExists(clusterizers, tabname_full)) {
        return;
    }
    clusterizers[tabname_full].tree_list.scroll_elem.parentElement.classList.toggle("hidden", !show);
    clusterizers[tabname_full].tree_list.enable(show);
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

    const pane = gradioApp().getElementById(`${tabname_full}_pane`);
    pane.querySelector(".extra-network-content--dirs-view").classList.toggle("hidden", !show);
}

function extraNetworksControlRefreshOnClick(event, tabname_full) {
    /** Handles `onclick` events for the Refresh Page button.
     *
     * In order to actually call the python functions in `ui_extra_networks.py`
     * to refresh the page, we created an empty gradio button in that file with an
     * event handler that refreshes the page. So what this function here does
     * is it manually raises a `click` event on that button.
     */
    // reset states
    initialUiOptionsLoaded.state = false;

    // We want to reset all clusterizers on refresh click so that the viewing area
    // shows that it is loading new data.
    for (const _tabname_full of Object.keys(clusterizers)) {
        for (const v of Object.values(clusterizers[_tabname_full])) {
            v.clear();
        }
    }

    // Fire an event for this button click.
    gradioApp().getElementById(`${tabname_full}_extra_refresh_internal`).dispatchEvent(new Event("click"));
}

function extraNetworksCardOnClick(event, tabname) {
    const elem = event.currentTarget;
    const prompt_elem = gradioApp().querySelector(`#${tabname}_prompt > label > textarea`);
    const neg_prompt_elem = gradioApp().querySelector(`#${tabname}_neg_prompt > label > textarea`);
    if ("negPrompt" in elem.dataset) {
        extraNetworksUpdatePrompt(prompt_elem, elem.dataset.prompt);
        extraNetworksUpdatePrompt(neg_prompt_elem, elem.dataset.negPrompt);
    } else if ("allowNeg" in elem.dataset) {
        extraNetworksUpdatePrompt(activePromptTextarea[tabname], elem.dataset.prompt);
    } else {
        extraNetworksUpdatePrompt(prompt_elem, elem.dataset.prompt);
    }
}

function extraNetworksTreeFileOnClick(event, btn, tabname_full) {
    return;
}

function extraNetworksTreeDirectoryOnClick(event, btn, tabname_full) {
    return;
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
        extraNetworksAutoSetTreeWidth(event.target.closest(".extra-network-pane"));
    });

    // Update search filter whenever the search input's clear button is pressed.
    window.addEventListener("extra-network-control--search-clear", event => {
        event.stopPropagation();
        extraNetworksApplyFilter(event.detail.tabname_full);
    });

    // Debounce search text input. This way we only search after user is done typing.
    const search_input_debounce = debounce((tabname_full) => {
        extraNetworksApplyFilter(tabname_full);
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

async function extraNetworksSetupTabContent(tabname, pane, controls_div) {
    const tabname_full = pane.id;
    const extra_networks_tabname = tabname_full.replace(`${tabname}_`, "");

    const controls = await waitForElement(`#${tabname_full}_pane .extra-network-controls`);
    await waitForElement(`#${tabname_full}_pane .extra-network-content--dirs-view`);

    controls.id = `${tabname_full}_controls`;
    controls_div.insertBefore(controls, null);

    clusterizers[tabname_full] = {
        tree_list: new ExtraNetworksClusterizeTreeList({
            tabname: tabname,
            extra_networks_tabname: extra_networks_tabname,
            scrollId: `${tabname_full}_tree_list_scroll_area`,
            contentId: `${tabname_full}_tree_list_content_area`,
            tag: "div",
            callbacks: {
                initData: extraNetworksOnInitData,
                fetchData: extraNetworksOnFetchData,
            },
        }),
        cards_list: new ExtraNetworksClusterizeCardsList({
            tabname: tabname,
            extra_networks_tabname: extra_networks_tabname,
            scrollId: `${tabname_full}_cards_list_scroll_area`,
            contentId: `${tabname_full}_cards_list_content_area`,
            tag: "div",
            callbacks: {
                initData: extraNetworksOnInitData,
                fetchData: extraNetworksOnFetchData,
            },
        }),
    };

    await clusterizers[tabname_full].tree_list.setup();
    await clusterizers[tabname_full].cards_list.setup();

    if (pane.style.display !== "none") {
        extraNetworksShowControlsForPage(tabname, tabname_full);
    }
}

async function extraNetworksSetupTab(tabname) {
    let controls_div;

    const this_tab = await waitForElement(`#${tabname}_extra_tabs`);
    const tab_nav = await waitForElement(`#${tabname}_extra_tabs > div.tab-nav`);
    
    controls_div = document.createElement("div");
    controls_div.classList.add("extra-network-controls-div");
    tab_nav.appendChild(controls_div);
    tab_nav.insertBefore(controls_div, null);
    const panes = this_tab.querySelectorAll(`:scope > .tabitem[id^="${tabname}_"]`);
    for (const pane of panes) {
        await extraNetworksSetupTabContent(tabname, pane, controls_div);
    }
    extraNetworksRegisterPromptForTab(tabname, `${tabname}_prompt`);
    extraNetworksRegisterPromptForTab(tabname, `${tabname}_neg_prompt`);
}

async function extraNetworksSetup() {
    await waitForBool(initialUiOptionsLoaded);

    extraNetworksSetupTab('txt2img');
    extraNetworksSetupTab('img2img');
    extraNetworksSetupEventDelegators();
}

onUiLoaded(extraNetworksSetup);
onOptionsChanged(() => initialUiOptionsLoaded.state = true);
