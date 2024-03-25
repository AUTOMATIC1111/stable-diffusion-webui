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
const initialUiOptionsLoaded = { state: false };

/** Helper functions for checking types and simplifying logging. */

const isString = x => typeof x === "string" || x instanceof String;
const isStringLogError = x => {
    if (isString(x)) {
        return true;
    }
    console.error("expected string, got:", typeof x);
    return false;
};
const isNull = x => typeof x === "null" || x === null;
const isUndefined = x => typeof x === "undefined" || x === undefined;
// checks both null and undefined for simplicity sake.
const isNullOrUndefined = x => isNull(x) || isUndefined(x);
const isNullOrUndefinedLogError = x => {
    if (isNullOrUndefined(x)) {
        console.error("Variable is null/undefined.");
        return true;
    }
    return false;
};

const isElement = x => x instanceof Element;
const isElementLogError = x => {
    if (isElement(x)) {
        return true;
    }
    console.error("expected element type, got:", typeof x);
    return false;
}

const getElementByIdLogError = selector => {
    let elem = gradioApp().getElementById(selector);
    isElementLogError(elem);
    return elem;
};

const querySelectorLogError = selector => {
    let elem = gradioApp().querySelector(selector);
    isElementLogError(elem);
    return elem;
}

const debounce = (handler, timeout_ms) => {
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
}

const waitForElement = selector => {
    /** Promise that waits for an element to exist in DOM. */
    return new Promise(resolve => {
        if (document.querySelector(selector)) {
            return resolve(document.querySelector(selector));
        }

        const observer = new MutationObserver(mutations => {
            if (document.querySelector(selector)) {
                observer.disconnect();
                resolve(document.querySelector(selector));
            }
        });

        observer.observe(document.documentElement, {
            childList: true,
            subtree: true
        });
    });
}

const waitForBool = o => {
    /** Promise that waits for a boolean to be true.
     * 
     *  `o` must be an Object of the form:
     *  { state: <bool value> }
     * 
     *  Resolves when (state === true)
     */
    return new Promise(resolve => {
        (function _waitForBool() {
            if (o.state) {
                return resolve();
            }
            setTimeout(_waitForBool, 100);
        })();
    });
}

const waitForKeyInObject = o =>  {
    /** Promise that waits for a key to exist in an object.
     *  
     *  `o` must be an Object of the form:
     *  {
     *      obj: <object to watch for key>,
     *      k: <key to watch for>, 
     *  }
     * 
     *  Resolves when (k in obj)
     */
    return new Promise(resolve => {
        (function _waitForKeyInObject() {
            if (o.k in o.obj) {
                return resolve();
            }
            setTimeout(_waitForKeyInObject, 100);
        })();
    });
}

const waitForValueInObject = o => {
    /** Promise that waits for a key value pair in an Object.
     *  
     *  `o` must be an Object of the form:
     *  {
     *      obj: <object containing value>,
     *      k: <key in object>,
     *      v: <value at key for comparison>
     *  }
     * 
     *  Resolves when obj[k] == v
     */
    return new Promise(resolve => {
        waitForKeyInObject({ k: o.k, obj: o.obj }).then(() => {
            (function _waitForValueInObject() {

                if (o.k in o.obj && o.obj[o.k] == o.v) {
                    return resolve();
                }
                setTimeout(_waitForValueInObject, 100);
            })();
        });
    });
}

function toggleCss(key, css, enable) {
    var style = document.getElementById(key);
    if (enable && !style) {
        style = document.createElement('style');
        style.id = key;
        style.type = 'text/css';
        document.head.appendChild(style);
    }
    if (style && !enable) {
        document.head.removeChild(style);
    }
    if (style) {
        style.innerHTML == '';
        style.appendChild(document.createTextNode(css));
    }
}

function extraNetworksRefreshTab(tabname_full) {
    // Reapply controls since they don't change on refresh.
    let btn_dirs_view = gradioApp().querySelector(`#${tabname_full}_extra_dirs_view`);
    let btn_tree_view = gradioApp().querySelector(`#${tabname_full}_extra_tree_view`);

    let div_dirs = gradioApp().querySelector(`#${tabname_full}_dirs`);
    let div_tree = gradioApp().querySelector(`#${tabname_full}_tree_list_scroll_area`);

    // Remove "hidden" class if button is enabled, otherwise add it.
    div_dirs.classList.toggle("hidden", !btn_dirs_view.classList.contains("extra-network-control--enabled"));
    div_tree.classList.toggle("hidden", !btn_tree_view.classList.contains("extra-network-control--enabled"));

    waitForKeyInObject({ k: tabname_full, obj: clusterizers })
        .then(() => extraNetworkClusterizersOnTabLoad(tabname_full));
}

function extraNetworksRegisterPromptForTab(tabname, id) {
    var textarea = gradioApp().querySelector(`#${id} > label > textarea`);

    if (!activePromptTextarea[tabname]) {
        activePromptTextarea[tabname] = textarea;
    }

    textarea.addEventListener("focus", function () {
        activePromptTextarea[tabname] = textarea;
    });
}

function extraNetworksSetupTabContent(tabname, pane, controls_div) {
    const tabname_full = pane.id;
    let txt_search;

    Promise.all([
        waitForElement(`#${tabname_full}_extra_search`).then((elem) => txt_search = elem),
        waitForElement(`#${tabname_full}_dirs`),
        waitForElement(`#${tabname_full}_tree_list_scroll_area > #${tabname_full}_tree_list_content_area`),
        waitForElement(`#${tabname_full}_cards_list_scroll_area > #${tabname_full}_cards_list_content_area`),
    ]).then(() => {
        // Now that we have our elements in DOM, we create the clusterize lists.
        clusterizers[tabname_full] = {
            tree_list: new ExtraNetworksClusterizeTreeList({
                data_id: `${tabname_full}_tree_list_data`,
                scroll_id: `${tabname_full}_tree_list_scroll_area`,
                content_id: `${tabname_full}_tree_list_content_area`,
                txt_search_elem: txt_search,
            }),
            cards_list: new ExtraNetworksClusterizeCardsList({
                data_id: `${tabname_full}_cards_list_data`,
                scroll_id: `${tabname_full}_cards_list_scroll_area`,
                content_id: `${tabname_full}_cards_list_content_area`,
                txt_search_elem: txt_search,
            }),
        };

        // Debounce search text input. This way we only search after user is done typing.
        const search_input_debounce = debounce(() => {
            extraNetworksApplyFilter(tabname_full);
        }, SEARCH_INPUT_DEBOUNCE_TIME_MS);
        txt_search.addEventListener("keyup", search_input_debounce);

        // Update search filter whenever the search input's clear button is pressed.
        txt_search.addEventListener("extra-network-control--search-clear", () => {
            extraNetworksApplyFilter(tabname_full);
        });

        // Insert the controls into the page.
        const controls = gradioApp().querySelector(`#${tabname_full}_controls`);
        controls_div.insertBefore(controls, null);
        if (pane.style.display != "none") {
            extraNetworksShowControlsForPage(tabname, tabname_full);
        }
    });
}

function extraNetworksSetupTab(tabname) {
    let this_tab;
    let tab_nav;
    let controls_div;
    Promise.all([
        waitForElement(`#${tabname}_extra_tabs`).then((elem) => this_tab = elem),
        waitForElement(`#${tabname}_extra_tabs > div.tab-nav`).then((elem) => tab_nav = elem),
    ]).then(() => {
        controls_div = document.createElement("div");
        controls_div.classList.add("extra-network-controls-div");
        tab_nav.appendChild(controls_div);
        tab_nav.insertBefore(controls_div, null);
        this_tab.querySelectorAll(`:scope > .tabitem[id^="${tabname}_"]`).forEach((elem) => {
            extraNetworksSetupTabContent(tabname, elem, controls_div);
        });
        extraNetworksRegisterPromptForTab(tabname, `${tabname}_prompt`);
        extraNetworksRegisterPromptForTab(tabname, `${tabname}_neg_prompt`);
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
        elem.style.display = `${tabname_full}_controls` === elem.id ? "" : "none";
    });
}


function extraNetworksUnrelatedTabSelected(tabname) { // called from python when user selects an unrelated tab (generate)
    extraNetworksMovePromptToTab(tabname, '', false, false);
    extraNetworksShowControlsForPage(tabname, null);
}

function extraNetworksTabSelected(tabname, id, showPrompt, showNegativePrompt, tabname_full) { // called from python when user selects an extra networks tab
    extraNetworksMovePromptToTab(tabname, id, showPrompt, showNegativePrompt);
    extraNetworksShowControlsForPage(tabname, tabname_full);

    waitForKeyInObject({ k: tabname_full, obj: clusterizers })
        .then(() => extraNetworkClusterizersOnTabLoad(tabname_full));
}

function extraNetworksApplyFilter(tabname_full) {
    if (!(tabname_full in clusterizers)) {
        console.error(`${tabname_full} not in clusterizers.`);
        return;
    }

    // We only want to filter/sort the cards list.
    clusterizers[tabname_full].cards_list.applyFilter();

    // If the search input has changed since selecting a button to populate it
    // then we want to disable the button that previously populated the search input.
    const txt_search = gradioApp().querySelector(`#${tabname_full}_extra_search`);
    if (!isElementLogError(txt_search)) {
        return;
    }

    // tree view buttons
    let btn = gradioApp().querySelector(`#${tabname_full}_tree_list_content_area .tree-list-item[data-selected=""]`);
    if (isElement(btn) && btn.dataset.path !== txt_search.value && "selected" in btn.dataset) {
        clusterizers[tabname_full].tree_list.onRowSelected(btn.dataset.divId, btn, false);
    }
    // dirs view buttons
    btn = gradioApp().querySelector(`#${tabname_full}_dirs .extra-network-dirs-view-button[data-selected=""]`);
    if (isElement(btn) && btn.textContent.trim() !== txt_search.value) {
        delete btn.dataset.selected;
    }
}

function extraNetworksClusterizersEnable(tabname_full) {
    /** Enables the selected tab's clusterize lists and disables all others. */
    // iterate over tabnames
    for (const [_tabname_full, tab_clusterizers] of Object.entries(clusterizers)) {
        // iterate over clusterizers in tab
        for (const v of Object.values(tab_clusterizers)) {
            v.enable(_tabname_full === tabname_full);
        }
    }
}

function extraNetworkClusterizersOnTabLoad(tabname_full) { /** promise */
    return new Promise(resolve => {
        // Enables a tab's clusterizer, updates its data, and rebuilds it.
        if (!(tabname_full in clusterizers)) {
            return resolve();
        }
        
        (async () => {
            // Enable then load the selected tab's clusterize lists.
            extraNetworksClusterizersEnable(tabname_full);
            for (const v of Object.values(clusterizers[tabname_full])) {
                await v.load();
            }
        })().then(() => {
            return resolve();
        });
    });
}

function extraNetworksSetupEventHandlers() {
    // Handle doubleclick events from the resize handle.
    // This will automatically fit the left pane to the size of its largest loaded row.
    window.addEventListener("resizeHandleDblClick", (e) => {
        // See resizeHandle.js::onDoubleClick() for event detail.
        e.stopPropagation();
        const tabname_full = e.target.closest(".extra-network-pane").dataset.tabnameFull;
        // This event is only applied to the currently selected tab if has clusterize list.
        if (!(tabname_full in clusterizers)) {
            return;
        }

        // Shouldn't be possible, but if tree is hidden then we want to ignore this event
        // since there wouldn't be any content to calculate row width. When hidden,
        // the resize handle shouldn't exist on page but just in case...
        if (gradioApp().getElementById(`${tabname_full}_tree_list_scroll_area`).parentElement.classList.contains("hidden")) {
            return;
        }

        // We know that the tree list is the left column. That is the only one we want to resize.
        let max_width = clusterizers[tabname_full].tree_list.getMaxRowWidth();
        // Add the resize handle's padding to the result and default to minLeftColWidth if necessary.
        max_width = Math.max(max_width + e.detail.pad, e.detail.minLeftColWidth);

        e.detail.setLeftColGridTemplate(e.target, max_width);
    });
}

function setupExtraNetworks() {
    waitForBool(initialUiOptionsLoaded).then(() => {
        extraNetworksSetupTab('txt2img');
        extraNetworksSetupTab('img2img');
        extraNetworksSetupEventHandlers();
    });
}

function tryToRemoveExtraNetworkFromPrompt(textarea, text, isNeg) {
    var m = text.match(isNeg ? re_extranet_neg : re_extranet);
    var replaced = false;
    var newTextareaText;
    var extraTextBeforeNet = opts.extra_networks_add_text_separator;
    if (m) {
        var extraTextAfterNet = m[2];
        var partToSearch = m[1];
        var foundAtPosition = -1;
        newTextareaText = textarea.value.replaceAll(isNeg ? re_extranet_g_neg : re_extranet_g, function (found, net, pos) {
            m = found.match(isNeg ? re_extranet_neg : re_extranet);
            if (m[1] == partToSearch) {
                replaced = true;
                foundAtPosition = pos;
                return "";
            }
            return found;
        });
        if (foundAtPosition >= 0) {
            if (extraTextAfterNet && newTextareaText.substr(foundAtPosition, extraTextAfterNet.length) == extraTextAfterNet) {
                newTextareaText = newTextareaText.substr(0, foundAtPosition) + newTextareaText.substr(foundAtPosition + extraTextAfterNet.length);
            }
            if (newTextareaText.substr(foundAtPosition - extraTextBeforeNet.length, extraTextBeforeNet.length) == extraTextBeforeNet) {
                newTextareaText = newTextareaText.substr(0, foundAtPosition - extraTextBeforeNet.length) + newTextareaText.substr(foundAtPosition);
            }
        }
    } else {
        newTextareaText = textarea.value.replaceAll(new RegExp(`((?:${extraTextBeforeNet})?${text})`, "g"), "");
        replaced = (newTextareaText != textarea.value);
    }

    if (replaced) {
        textarea.value = newTextareaText;
        return true;
    }

    return false;
}

function updatePromptArea(text, textArea, isNeg) {
    if (!tryToRemoveExtraNetworkFromPrompt(textArea, text, isNeg)) {
        textArea.value = textArea.value + opts.extra_networks_add_text_separator + text;
    }

    updateInput(textArea);
}

function cardClicked(tabname, textToAdd, textToAddNegative, allowNegativePrompt) {
    if (textToAddNegative.length > 0) {
        updatePromptArea(textToAdd, gradioApp().querySelector(`#${tabname}_prompt > label > textarea`));
        updatePromptArea(textToAddNegative, gradioApp().querySelector(`#${tabname}_neg_prompt > label > textarea`), true);
    } else {
        var textarea = allowNegativePrompt ? activePromptTextarea[tabname] : gradioApp().querySelector(`#${tabname}_prompt > label > textarea`);
        updatePromptArea(textToAdd, textarea);
    }
}

function saveCardPreview(event, tabname, filename) {
    var textarea = gradioApp().querySelector(`#${tabname}_preview_filename > label > textarea`);
    var button = gradioApp().getElementById(`${tabname}_save_preview`);

    textarea.value = filename;
    updateInput(textarea);

    button.click();

    event.stopPropagation();
    event.preventDefault();
}

function extraNetworksTreeProcessFileClick(event, btn, tabname_full) {
    /**
     * Processes `onclick` events when user clicks on files in tree.
     *
     * @param event         The generated event.
     * @param btn           The clicked `tree-list-item` button.
     * @param tabname_full  The full active tabname.
     *                      i.e. txt2img_lora, img2img_checkpoints, etc.
     */
    // NOTE: Currently unused.
    return;
}

function extraNetworksTreeProcessDirectoryClick(event, btn, tabname_full) {
    /**
     * Processes `onclick` events when user clicks on directories in tree.
     *
     * Here is how the tree reacts to clicks for various states:
     * unselected unopened directory: Directory is selected and expanded.
     * unselected opened directory: Directory is selected.
     * selected opened directory: Directory is collapsed and deselected.
     * chevron is clicked: Directory is expanded or collapsed. Selected state unchanged.
     *
     * @param event         The generated event.
     * @param btn           The clicked `tree-list-item` button.
     * @param tabname_full  The full active tabname.
     *                      i.e. txt2img_lora, img2img_checkpoints, etc.
     */
    // This is the actual target that the user clicked on within the target button.
    // We use this to detect if the chevron was clicked.
    var true_targ = event.target;
    const div_id = btn.dataset.divId;

    function _updateSearch(_search_text) {
        // Update search input with select button's path.
        var search_input_elem = gradioApp().querySelector(`#${tabname_full}_extra_search`);
        search_input_elem.value = _search_text;
        updateInput(search_input_elem);
        extraNetworksApplyFilter(tabname_full);
    }

    if (true_targ.matches(".tree-list-item-action--leading, .tree-list-item-action-chevron")) {
        // If user clicks on the chevron, then we do not select the folder.
        let prev_selected_elem = gradioApp().querySelector(".tree-list-item[data-selected='']");
        clusterizers[tabname_full].tree_list.onRowExpandClick(div_id, btn);
        let selected_elem = gradioApp().querySelector(".tree-list-item[data-selected='']");
        if (isElement(prev_selected_elem) && !isElement(selected_elem)) {
            // if a selected element was removed, clear filter.
            _updateSearch("");
        }
    } else {
        // user clicked anywhere else on row.
        clusterizers[tabname_full].tree_list.onRowSelected(div_id, btn);
        _updateSearch("selected" in btn.dataset ? btn.dataset.path : "");
    }
}

function extraNetworksTreeOnClick(event, tabname_full) {
    /**
     * Handles `onclick` events for buttons within an `extra-network-tree .tree-list--tree`.
     *
     * Determines whether the clicked button in the tree is for a file entry or a directory
     * then calls the appropriate function.
     *
     * @param event         The generated event.
     * @param tabname_full  The full active tabname.
     *                      i.e. txt2img_lora, img2img_checkpoints, etc.
     */
    let btn = event.target.closest(".tree-list-item");
    if (btn.dataset.treeEntryType === "file") {
        extraNetworksTreeProcessFileClick(event, btn, tabname_full);
    } else {
        extraNetworksTreeProcessDirectoryClick(event, btn, tabname_full);
    }
    event.stopPropagation();
}

function extraNetworkDirsOnClick(event, tabname_full) {
    /** Handles `onclick` events for buttons in the directory view. */
    let txt_search = gradioApp().querySelector(`#${tabname_full}_extra_search`);

    function _deselect_all() {
        // deselect all buttons
        gradioApp().querySelectorAll(".extra-network-dirs-view-button").forEach((elem) => {
            delete elem.dataset.selected;
        });
    }

    function _select_button(elem) {
        _deselect_all();
        // Update search input with select button's path.
        elem.dataset.selected = "";
        txt_search.value = elem.textContent.trim();
    }

    function _deselect_button(elem) {
        delete elem.dataset.selected;
        txt_search.value = "";
    }


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
            { detail: { tabname_full: tabname_full } },
        )
    );
}

function extraNetworksControlSortModeOnClick(event, tabname_full) {
    /** Handles `onclick` events for Sort Mode buttons. */
    event.currentTarget.parentElement.querySelectorAll('.extra-network-control--sort-mode').forEach(function (x) {
        x.classList.remove('extra-network-control--enabled');
    });

    event.currentTarget.classList.add('extra-network-control--enabled');

    if (tabname_full in clusterizers) {
        clusterizers[tabname_full].cards_list.setSortMode(
            event.currentTarget.dataset.sortMode.toLowerCase()
        );
        extraNetworksApplyFilter(tabname_full);
    }
}

function extraNetworksControlSortDirOnClick(event, tabname_full) {
    /** Handles `onclick` events for the Sort Direction button.
     *
     * Modifies the data attributes of the Sort Direction button to cycle between
     * ascending and descending sort directions.
     */
    if (event.currentTarget.dataset.sortDir.toLowerCase() == "ascending") {
        event.currentTarget.dataset.sortDir = "descending";
        event.currentTarget.setAttribute("title", "Sort descending");
    } else {
        event.currentTarget.dataset.sortDir = "ascending";
        event.currentTarget.setAttribute("title", "Sort ascending");
    }

    if (tabname_full in clusterizers) {
        clusterizers[tabname_full].cards_list.setSortDir(event.currentTarget.dataset.sortDir.toLowerCase());
        extraNetworksApplyFilter(tabname_full);
    }
}

function extraNetworksControlTreeViewOnClick(event, tabname_full) {
    /** Handles `onclick` events for the Tree View button.
     *
     * Toggles the tree view in the extra networks pane.
     */
    var button = event.currentTarget;
    button.classList.toggle("extra-network-control--enabled");
    var show = button.classList.contains("extra-network-control--enabled");

    gradioApp().getElementById(`${tabname_full}_tree_list_scroll_area`).parentElement.classList.toggle("hidden", !show);
    clusterizers[tabname_full].tree_list.enable(show);
}

function extraNetworksControlDirsViewOnClick(event, tabname_full) {
    /** Handles `onclick` events for the Dirs View button.
     *
     * Toggles the directory view in the extra networks pane.
     */
    var button = event.currentTarget;
    button.classList.toggle("extra-network-control--enabled");
    var show = button.classList.contains("extra-network-control--enabled");

    gradioApp().getElementById(`${tabname_full}_dirs`).classList.toggle("hidden", !show);
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
    // Fire an event for this button click.
    gradioApp().getElementById(`${tabname_full}_extra_refresh_internal`).dispatchEvent(new Event("click"));
}

function closePopup() {
    if (!globalPopup) return;
    globalPopup.style.display = "none";
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

function requestGet(url, data, handler, errorHandler) {
    var xhr = new XMLHttpRequest();
    var args = Object.keys(data).map(function (k) {
        return encodeURIComponent(k) + '=' + encodeURIComponent(data[k]);
    }).join('&');
    xhr.open("GET", url + "?" + args, true);

    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                try {
                    var js = JSON.parse(xhr.responseText);
                    handler(js);
                } catch (error) {
                    console.error(error);
                    errorHandler();
                }
            } else {
                errorHandler();
            }
        }
    };
    var js = JSON.stringify(data);
    xhr.send(js);
}

function extraNetworksCopyPathToClipboard(event, path) {
    navigator.clipboard.writeText(path);
    event.stopPropagation();
}

function extraNetworksRequestMetadata(event, extraPage, cardName) {
    var showError = function () {
        extraNetworksShowMetadata("there was an error getting metadata");
    };

    requestGet("./sd_extra_networks/metadata", { page: extraPage, item: cardName }, function (data) {
        if (data && data.metadata) {
            extraNetworksShowMetadata(data.metadata);
        } else {
            showError();
        }
    }, showError);

    event.stopPropagation();
}

function extraNetworksEditUserMetadata(event, tabname, extraPage, cardName) {
    var id = tabname + '_' + extraPage + '_edit_user_metadata';

    var editor = extraPageUserMetadataEditors[id];
    if (!editor) {
        editor = {};
        editor.page = gradioApp().getElementById(id);
        editor.nameTextarea = gradioApp().querySelector("#" + id + "_name" + ' textarea');
        editor.button = gradioApp().querySelector("#" + id + "_button");
        extraPageUserMetadataEditors[id] = editor;
    }

    editor.nameTextarea.value = cardName;
    updateInput(editor.nameTextarea);

    editor.button.click();

    popup(editor.page);

    event.stopPropagation();
}

function extraNetworksRefreshSingleCard(page, tabname, name) {
    requestGet("./sd_extra_networks/get-single-card", { page: page, tabname: tabname, name: name }, function (data) {
        if (data && data.html) {
            var card = gradioApp().querySelector(`#${tabname}_${page.replace(" ", "_")}_cards > .card[data-name="${name}"]`);

            var newDiv = document.createElement('DIV');
            newDiv.innerHTML = data.html;
            var newCard = newDiv.firstElementChild;

            newCard.style.display = '';
            card.parentElement.insertBefore(newCard, card);
            card.parentElement.removeChild(card);
        }
    });
}

window.addEventListener("keydown", function (event) {
    if (event.key == "Escape") {
        closePopup();
    }
});

onUiLoaded(setupExtraNetworks);
onOptionsChanged(() => initialUiOptionsLoaded.state = true);