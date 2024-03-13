const re_extranet = /<([^:^>]+:[^:]+):[\d.]+>(.*)/;
const re_extranet_g = /<([^:^>]+:[^:]+):[\d.]+>/g;
const re_extranet_neg = /\(([^:^>]+:[\d.]+)\)/;
const re_extranet_g_neg = /\(([^:^>]+:[\d.]+)\)/g;
const extraNetworksApplyFilter = {};
const extraNetworksApplySort = {};
const activePromptTextarea = {};
const clusterizers = {};
const extra_networks_json_proxy = {};
const extra_networks_proxy_listener = setupProxyListener(
    extra_networks_json_proxy,
    function() {},
    proxyJsonUpdated,
);
var globalPopup = null;
var globalPopupInner = null;
const storedPopupIds = {};
const extraPageUserMetadataEditors = {};
const uiAfterScriptsCallbacks = [];
var uiAfterScriptsTimeout = null;
var executedAfterScripts = false;

function waitForElement(selector) {
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

        // If you get "parameter 1 is not of type 'Node'" error, see https://stackoverflow.com/a/77855838/492336
        observer.observe(document.documentElement, {
            childList: true,
            subtree: true
        });
    });
}

function setupProxyListener(target, pre_handler, post_handler) {
    /** Sets up a listener for variable changes. */
    var proxy = new Proxy(target, {
        set: function (t, k, v) {
            pre_handler.call(t, k, v);
            t[k] = v;
            post_handler.call(t, k, v);
            return true;
        }
    });
    return proxy
}

function proxyJsonUpdated(k, v) {
    /** Callback triggered when JSON data is updated by the proxy listener. */
    // use `this` for current object
    // We don't do error handling here because we want to fail gracefully if data is
    // not yet present.
    if (!(v.dataset.tabnameFull in clusterizers)) {
        // Clusterizers not yet initialized.
        return;
    }
    clusterizers[v.dataset.tabnameFull][v.dataset.proxyName].parseJson(v.dataset.json);
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
    if (!(tabname_full in clusterizers)) {
        return;
    }

    for (_tabname_full of Object.keys(clusterizers)) {

        if (_tabname_full === tabname_full) {
            // Set the selected tab as active since it is now visible on page.
            clusterizers[_tabname_full].tree_list.enable();
            clusterizers[_tabname_full].cards_list.enable();
        } else {
            // Deactivate all other tabs since they are no longer visible.
            clusterizers[_tabname_full].tree_list.disable();
            clusterizers[_tabname_full].cards_list.disable();
        }
    }

    clusterizers[tabname_full].tree_list.rebuild();
    clusterizers[tabname_full].cards_list.rebuild();

    for (var elem of gradioApp().querySelectorAll('.extra-networks-script-data')) {
        extra_networks_proxy_listener[`${elem.dataset.tabnameFull}_${elem.dataset.proxyName}`] = elem;
    }
}

function setupExtraNetworksForTab(tabname) {
    function registerPrompt(tabname, id) {
        var textarea = gradioApp().querySelector(`#${id} > label > textarea`);

        if (!activePromptTextarea[tabname]) {
            activePromptTextarea[tabname] = textarea;
        }

        textarea.addEventListener("focus", function() {
            activePromptTextarea[tabname] = textarea;
        });
    }

    var tabnav = gradioApp().querySelector(`#${tabname}_extra_tabs > div.tab-nav`);
    var controlsDiv = document.createElement("div");
    controlsDiv.classList.add("extra-networks-controls-div");
    tabnav.appendChild(controlsDiv);
    tabnav.insertBefore(controlsDiv, null);

    var this_tab = gradioApp().querySelector(`#${tabname}_extra_tabs`);
    this_tab.querySelectorAll(`:scope > [id^="${tabname}_"]`).forEach(function(elem) {
        let tabname_full = elem.id;
        let txt_search;
        let btn_sort_mode;
        let btn_sort_dir;
        let btn_refresh;

        var applyFilter = function() {
            if (!(tabname_full in clusterizers)) {
                console.error(`applyFilter: ${tabname_full} not in clusterizers:`);
                return;
            }
            // Only touch cards_list. tree_list remains static.
            clusterizers[tabname_full].cards_list.setSortMode(btn_sort_mode);
            clusterizers[tabname_full].cards_list.setSortDir(btn_sort_dir);
            clusterizers[tabname_full].cards_list.applyFilter(txt_search.value);
        };
        extraNetworksApplyFilter[tabname_full] = applyFilter;

        var applySort = function() {
            if (!(tabname_full in clusterizers)) {
                console.error(`applySort: ${tabname_full} not in clusterizers:`);
                return;
            }
            // Only touch cards_list. tree_list remains static.
            clusterizers[tabname_full].cards_list.setSortMode(btn_sort_mode);
            clusterizers[tabname_full].cards_list.setSortDir(btn_sort_dir);
            clusterizers[tabname_full].cards_list.applyFilter(txt_search.value); // filter also sorts
        };
        extraNetworksApplySort[tabname_full] = applySort;
        /** #TODO
         * Figure out if we can use the following in the clusterize setup:
         * var frag = document.createDocumentFragment();
         * sortedCards.forEach(function(card) {
         *     frag.appendChild(card);
         * });
         * parent.appendChild(frag);
         */

        // Wait for all required elements before setting up the tab.
        waitForElement(`#${tabname_full}_extra_search`)
            .then((el) => {
                txt_search = el;
            })
            .then(() => {
                waitForElement(`#${tabname_full}_extra_sort_mode`)
                    .then((el) => { btn_sort_mode = el; });
            })
            .then(() => {
                waitForElement(`#${tabname_full}_extra_sort_dir`)
                    .then((el) => { btn_sort_dir = el; });
            })
            .then(() => {
                waitForElement(`#${tabname_full}_extra_refresh`)
                    .then((el) => { btn_refresh = el; });
            })
            .then(() => {
                waitForElement(`#${tabname_full}_tree_list_scroll_area > #${tabname_full}_tree_list_content_area`)
                    .then(() => { return; });
            })
            .then(() => {
                waitForElement(`#${tabname_full}_cards_list_scroll_area > #${tabname_full}_cards_list_content_area`)
                    .then(() => { return; });
            })
            .then(() => {
                console.log("LOADING TAB:", tabname_full, clusterizers[tabname_full]);
                // Now that we have our elements in DOM, we create the clusterize lists.
                clusterizers[tabname_full] = {
                    tree_list: new ExtraNetworksClusterizeTreeList({
                            scroll_id: `${tabname_full}_tree_list_scroll_area`,
                            content_id: `${tabname_full}_tree_list_content_area`,
                    }),
                    cards_list: new ExtraNetworksClusterizeCardsList({
                            scroll_id: `${tabname_full}_cards_list_scroll_area`,
                            content_id: `${tabname_full}_cards_list_content_area`,
                    }),
                };

                applyFilter();

                // Debounce search text input. This way we only search after user is done typing.
                let typing_timer;
                let done_typing_interval_ms = 250;
                txt_search.addEventListener("keyup", () => {
                    clearTimeout(typing_timer);
                    if (txt_search.value) {
                        typing_timer = setTimeout(applyFilter, done_typing_interval_ms);
                    }
                });
                // Triggered on "enter" key or when "x" is clicked to clear search.
                txt_search.addEventListener("extra-network-control--search-clear", applyFilter);

                // Insert the controls into the page.
                var controls = gradioApp().querySelector(`#${tabname_full}_controls`);
                controlsDiv.insertBefore(controls, null);
                if (elem.style.display != "none") {
                    extraNetworksShowControlsForPage(tabname, tabname_full);
                }
            });
    });

    registerPrompt(tabname, `${tabname}_prompt`);
    registerPrompt(tabname, `${tabname}_neg_prompt`);
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
    gradioApp().querySelectorAll(`#${tabname}_extra_tabs .extra-networks-controls-div > div`).forEach(function(elem) {
        var targetId = `${tabname_full}_controls`;
        elem.style.display = elem.id == targetId ? "" : "none";
    });
}


function extraNetworksUnrelatedTabSelected(tabname) { // called from python when user selects an unrelated tab (generate)
    extraNetworksMovePromptToTab(tabname, '', false, false);

    extraNetworksShowControlsForPage(tabname, null);
}

function extraNetworksTabSelected(tabname, id, showPrompt, showNegativePrompt, tabname_full) { // called from python when user selects an extra networks tab
    extraNetworksMovePromptToTab(tabname, id, showPrompt, showNegativePrompt);

    extraNetworksShowControlsForPage(tabname, tabname_full);

    for (_tabname_full of Object.keys(clusterizers)) {
        if (_tabname_full === tabname_full) {
            // Set the selected tab as active since it is now visible on page.
            clusterizers[_tabname_full].tree_list.enable();
            clusterizers[_tabname_full].cards_list.enable();
        } else {
            // Deactivate all other tabs since they are no longer visible.
            clusterizers[_tabname_full].tree_list.disable();
            clusterizers[_tabname_full].cards_list.disable();
        }
    }

    if (!document.body.contains(clusterizers[tabname_full].tree_list.scroll_elem)) {
        clusterizers[tabname_full].tree_list.rebuild();
    }
    if (!document.body.contains(clusterizers[tabname_full].cards_list.scroll_elem)) {
        clusterizers[tabname_full].cards_list.rebuild();
    }
}

function applyExtraNetworkFilter(tabname_full) {
    setTimeout(extraNetworksApplyFilter[tabname_full], 1);
}

function applyExtraNetworkSort(tabname_full) {
    setTimeout(extraNetworksApplySort[tabname_full], 1);
}

function setupExtraNetworksData() {
    // Manually force read the json data.
    for (var elem of gradioApp().querySelectorAll('.extra-networks-script-data')) {
        extra_networks_proxy_listener[`${elem.dataset.tabnameFull}_${elem.dataset.proxyName}`] = elem;
    }
}

function setupExtraNetworks() {
    setupExtraNetworksForTab('txt2img');
    setupExtraNetworksForTab('img2img');

    // Handle window resizes. Delay of 500ms after resize before firing an event
    // as a way of "debouncing" resizes.
    var resize_timer;
    window.addEventListener("resize", () => {
        clearTimeout(resize_timer);
        resize_timer = setTimeout(function() {
            // Update rows for each list.
            for (_tabname_full of Object.keys(clusterizers)) {
                clusterizers[_tabname_full].tree_list.updateRows();
                clusterizers[_tabname_full].cards_list.updateRows();
            }
        }, 500); // ms
    });

    // Handle resizeHandle resizes. Only fires on mouseup after resizing.
    window.addEventListener("resizeHandleResized", (e) => {
        for (_tabname_full of Object.keys(clusterizers)) {
            // Update rows for each list.
            clusterizers[_tabname_full].tree_list.updateRows();
            clusterizers[_tabname_full].cards_list.updateRows();
        }
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
        newTextareaText = textarea.value.replaceAll(isNeg ? re_extranet_g_neg : re_extranet_g, function(found, net, pos) {
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

function extraNetworksSearchButton(event, tabname_full) {
    var searchTextarea = gradioApp().querySelector("#" + tabname_full + "_extra_search");
    var button = event.target;
    var text = button.classList.contains("search-all") ? "" : button.textContent.trim();

    searchTextarea.value = text;
    updateInput(searchTextarea);
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

    function _expandOrCollapse(_btn) {
        // Expands/Collapses all children of the button.
        if ("expanded" in _btn.dataset) {
            delete _btn.dataset.expanded;
            clusterizers[tabname_full].tree_list.removeChildRows(div_id);
        } else {
            _btn.dataset.expanded = "";
            clusterizers[tabname_full].tree_list.addChildRows(div_id);
        }
        // update html after changing attr.
        clusterizers[tabname_full].tree_list.updateDivContent(div_id, _btn.outerHTML);
        clusterizers[tabname_full].tree_list.updateRows();
    }

    function _removeSelectedFromAll() {
        // Removes the `selected` attribute from all buttons.
        var sels = document.querySelectorAll(".tree-list-item");
        [...sels].forEach(el => {
            delete el.dataset.selected;
        });
    }

    function _selectButton(_btn) {
        // Removes `data-selected` attribute from all buttons then adds to passed button.
        _removeSelectedFromAll();
        _btn.dataset.selected = "";
    }

    function _updateSearch(_search_text) {
        // Update search input with select button's path.
        var search_input_elem = gradioApp().querySelector("#" + tabname_full + "_extra_search");
        search_input_elem.value = _search_text;
        updateInput(search_input_elem);
        applyExtraNetworksFilter(tabname_full);
    }


    // If user clicks on the chevron, then we do not select the folder.
    if (true_targ.matches(".tree-list-item-action--leading, .tree-list-item-action-chevron")) {
        _expandOrCollapse(btn);
    } else {
        // User clicked anywhere else on the button.
        if ("selected" in btn.dataset) {
            // If folder is selected, deselect button.
            delete btn.dataset.selected;
            _expandOrCollapse(ul, btn);
            _updateSearch("");
        } else {
            // If folder is not selected, select it.
            _selectButton(btn);
            _updateSearch(btn.dataset.path);
        }
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

function extraNetworksControlSearchClearOnClick(event, tabname_full) {
    /** Clears the search <input> text. */
    let clear_btn = event.target.closest(".extra-network-control--search-clear");
    let txt_search = clear_btn.previousElementSibling;
    txt_search.value = "";
    txt_search.dispatchEvent(new CustomEvent("extra-network-control--search-clear", {}));
}

function extraNetworksControlSortModeOnClick(event, tabname_full) {
    /** Handles `onclick` events for Sort Mode buttons. */

    var self = event.currentTarget;
    var parent = event.currentTarget.parentElement;

    parent.querySelectorAll('.extra-network-control--sort').forEach(function(x) {
        x.classList.remove('extra-network-control--enabled');
    });

    self.classList.add('extra-network-control--enabled');

    applyExtraNetworkSort(tabname_full);
}

function extraNetworksControlSortDirOnClick(event, tabname_full) {
    /**
     * Handles `onclick` events for the Sort Direction button.
     *
     * Modifies the data attributes of the Sort Direction button to cycle between
     * ascending and descending sort directions.
     *
     * @param event         The generated event.
     * @param tabname_full  The full active tabname.
     *                      i.e. txt2img_lora, img2img_checkpoints, etc.
     */
    if (event.currentTarget.dataset.sortDir.toLowerCase() == "ascending") {
        event.currentTarget.dataset.sortDir = "descending";
        event.currentTarget.setAttribute("title", "Sort descending");
    } else {
        event.currentTarget.dataset.sortDir = "ascending";
        event.currentTarget.setAttribute("title", "Sort ascending");
    }
    applyExtraNetworkSort(tabname_full);
}

function extraNetworksControlTreeViewOnClick(event, tabname_full) {
    /**
     * Handles `onclick` events for the Tree View button.
     *
     * Toggles the tree view in the extra networks pane.
     *
     * @param event         The generated event.
     * @param tabname_full  The full active tabname.
     *                      i.e. txt2img_lora, img2img_checkpoints, etc.
     */
    var button = event.currentTarget;
    button.classList.toggle("extra-network-control--enabled");
    var show = !button.classList.contains("extra-network-control--enabled");

    var pane = gradioApp().getElementById(`${tabname_full}_pane`);
    pane.classList.toggle("extra-network-dirs-hidden", show);
}

function extraNetworksControlRefreshOnClick(event, tabname_full) {
    /**
     * Handles `onclick` events for the Refresh Page button.
     *
     * In order to actually call the python functions in `ui_extra_networks.py`
     * to refresh the page, we created an empty gradio button in that file with an
     * event handler that refreshes the page. So what this function here does
     * is it manually raises a `click` event on that button.
     *
     * @param event         The generated event.
     * @param tabname_full  The full active tabname.
     *                      i.e. txt2img_lora, img2img_checkpoints, etc.
     */
    var btn_refresh_internal = gradioApp().getElementById(`${tabname_full}_extra_refresh_internal`);
    btn_refresh_internal.dispatchEvent(new Event("click"));
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
        console.eror(error);
    }

    var elem = document.createElement('pre');
    elem.classList.add('popup-metadata');
    elem.textContent = text;

    popup(elem);
    return;
}

function requestGet(url, data, handler, errorHandler) {
    var xhr = new XMLHttpRequest();
    var args = Object.keys(data).map(function(k) {
        return encodeURIComponent(k) + '=' + encodeURIComponent(data[k]);
    }).join('&');
    xhr.open("GET", url + "?" + args, true);

    xhr.onreadystatechange = function() {
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

function extraNetworksCopyCardPathToClipboard(event, path) {
    navigator.clipboard.writeText(path);
    event.stopPropagation();
}

function extraNetworksRequestMetadata(event, extraPage, cardName) {
    var showError = function() {
        extraNetworksShowMetadata("there was an error getting metadata");
    };

    requestGet("./sd_extra_networks/metadata", {page: extraPage, item: cardName}, function(data) {
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
    requestGet("./sd_extra_networks/get-single-card", {page: page, tabname: tabname, name: name}, function(data) {
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

window.addEventListener("keydown", function(event) {
    if (event.key == "Escape") {
        closePopup();
    }
});

onUiLoaded(setupExtraNetworks);
