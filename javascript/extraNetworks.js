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

function setupExtraNetworksForTab(tabname) {
    function registerPrompt(tabname, id) {
        var textarea = gradioApp().querySelector("#" + id + " > label > textarea");

        if (!activePromptTextarea[tabname]) {
            activePromptTextarea[tabname] = textarea;
        }

        textarea.addEventListener("focus", function() {
            activePromptTextarea[tabname] = textarea;
        });
    }

    var this_tab = gradioApp().querySelector('#' + tabname + '_extra_tabs');
    this_tab.classList.add('extra-networks');
    this_tab.querySelectorAll(":scope > [id^='" + tabname + "_']").forEach(function(elem) {
        var tab_id = elem.getAttribute("id");
        var search = gradioApp().querySelector("#" + tab_id + "_extra_search");
        if (!search) {
            return; // `continue` doesn't work in `forEach` loops. This is equivalent.
        }

        var tabs = gradioApp().querySelector('#' + tabname + '_extra_tabs > div');
        var sort = gradioApp().getElementById(tabname + '_extra_sort');
        var sortOrder = gradioApp().getElementById(tabname + '_extra_sortorder');
        var refresh = gradioApp().getElementById(tabname + '_extra_refresh');
        var promptContainer = gradioApp().querySelector('.prompt-container-compact#' + tabname + '_prompt_container');
        var negativePrompt = gradioApp().querySelector('#' + tabname + '_neg_prompt');
        tabs.appendChild(sort);
        tabs.appendChild(sortOrder);
        tabs.appendChild(refresh);

        var applyFilter = function() {
            var searchTerm = search.value.toLowerCase();

            gradioApp().querySelectorAll('#' + tabname + '_extra_tabs div.card').forEach(function(elem) {
                var searchOnly = elem.querySelector('.search_only');

                var text = Array.prototype.map.call(elem.querySelectorAll('.search_terms'), function(t) {
                    return t.textContent.toLowerCase();
                }).join(" ");

                var visible = text.indexOf(searchTerm) != -1;

                if (searchOnly && searchTerm.length < 4) {
                    visible = false;
                }

                elem.style.display = visible ? "" : "none";
            });

            applySort();
        };

        var applySort = function() {
            var cards = gradioApp().querySelectorAll('#' + tabname + '_extra_tabs div.card');

            var reverse = sortOrder.classList.contains("sortReverse");
            var sortKey = sort.querySelector("input").value.toLowerCase().replace("sort", "").replaceAll(" ", "_").replace(/_+$/, "").trim() || "name";
            sortKey = "sort" + sortKey.charAt(0).toUpperCase() + sortKey.slice(1);
            var sortKeyStore = sortKey + "-" + (reverse ? "Descending" : "Ascending") + "-" + cards.length;

            if (sortKeyStore == sort.dataset.sortkey) {
                return;
            }
            sort.dataset.sortkey = sortKeyStore;

            cards.forEach(function(card) {
                card.originalParentElement = card.parentElement;
            });
            var sortedCards = Array.from(cards);
            sortedCards.sort(function(cardA, cardB) {
                var a = cardA.dataset[sortKey];
                var b = cardB.dataset[sortKey];
                if (!isNaN(a) && !isNaN(b)) {
                    return parseInt(a) - parseInt(b);
                }

                return (a < b ? -1 : (a > b ? 1 : 0));
            });
            if (reverse) {
                sortedCards.reverse();
            }
            cards.forEach(function(card) {
                card.remove();
            });
            sortedCards.forEach(function(card) {
                card.originalParentElement.appendChild(card);
            });
        };

        search.addEventListener("input", applyFilter);
        sortOrder.addEventListener("click", function() {
            sortOrder.classList.toggle("sortReverse");
            applySort();
        });
        applyFilter();

        extraNetworksApplySort[tab_id] = applySort;
        extraNetworksApplyFilter[tab_id] = applyFilter;
    });

    registerPrompt(tabname, tabname + "_prompt");
    registerPrompt(tabname, tabname + "_neg_prompt");
}

function extraNetworksMovePromptToTab(tabname, id, showPrompt, showNegativePrompt) {
    if (!gradioApp().querySelector('.toprow-compact-tools')) return; // only applicable for compact prompt layout

    var promptContainer = gradioApp().getElementById(tabname + '_prompt_container');
    var prompt = gradioApp().getElementById(tabname + '_prompt_row');
    var negPrompt = gradioApp().getElementById(tabname + '_neg_prompt_row');
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

function clearSearch(tabname) {
    // Clear search box.
    var tab_id = tabname + "_extra_search";
    var searchTextarea = gradioApp().querySelector("#" + tab_id + ' > label > textarea');
    searchTextarea.value = "";
    updateInput(searchTextarea);
}


function extraNetworksUnrelatedTabSelected(tabname) { // called from python when user selects an unrelated tab (generate)
    extraNetworksMovePromptToTab(tabname, '', false, false);
}

function extraNetworksTabSelected(tabname, id, showPrompt, showNegativePrompt) { // called from python when user selects an extra networks tab
    extraNetworksMovePromptToTab(tabname, id, showPrompt, showNegativePrompt);
}

function applyExtraNetworkFilter(tabname) {
    setTimeout(extraNetworksApplyFilter[tabname], 1);
}

function applyExtraNetworkSort(tabname) {
    setTimeout(extraNetworksApplySort[tabname], 1);
}

var extraNetworksApplyFilter = {};
var extraNetworksApplySort = {};
var activePromptTextarea = {};

function setupExtraNetworks() {
    setupExtraNetworksForTab('txt2img');
    setupExtraNetworksForTab('img2img');
}

var re_extranet = /<([^:^>]+:[^:]+):[\d.]+>(.*)/;
var re_extranet_g = /<([^:^>]+:[^:]+):[\d.]+>/g;

var re_extranet_neg = /\(([^:^>]+:[\d.]+)\)/;
var re_extranet_g_neg = /\(([^:^>]+:[\d.]+)\)/g;
function tryToRemoveExtraNetworkFromPrompt(textarea, text, isNeg) {
    var m = text.match(isNeg ? re_extranet_neg : re_extranet);
    var replaced = false;
    var newTextareaText;
    if (m) {
        var extraTextBeforeNet = opts.extra_networks_add_text_separator;
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
        newTextareaText = textarea.value.replaceAll(new RegExp(text, "g"), function(found) {
            if (found == text) {
                replaced = true;
                return "";
            }
            return found;
        });
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
        updatePromptArea(textToAdd, gradioApp().querySelector("#" + tabname + "_prompt > label > textarea"));
        updatePromptArea(textToAddNegative, gradioApp().querySelector("#" + tabname + "_neg_prompt > label > textarea"), true);
    } else {
        var textarea = allowNegativePrompt ? activePromptTextarea[tabname] : gradioApp().querySelector("#" + tabname + "_prompt > label > textarea");
        updatePromptArea(textToAdd, textarea);
    }
}

function saveCardPreview(event, tabname, filename) {
    var textarea = gradioApp().querySelector("#" + tabname + '_preview_filename  > label > textarea');
    var button = gradioApp().getElementById(tabname + '_save_preview');

    textarea.value = filename;
    updateInput(textarea);

    button.click();

    event.stopPropagation();
    event.preventDefault();
}

function extraNetworksTreeProcessFileClick(event, btn, tabname, tab_id) {
    /**
     * Processes `onclick` events when user clicks on files in tree.
     *
     * @param event     The generated event.
     * @param btn       The clicked `action-list-item` button.
     * @param tabname   The name of the active tab in the sd webui. Ex: txt2img, img2img, etc.
     * @param tab_id    The id of the active extraNetworks tab. Ex: lora, checkpoints, etc.
     */
    var par = btn.parentElement;
    var search_id = tabname + "_" + tab_id + "_extra_search";
    var type = par.getAttribute("data-tree-entry-type");
    var path = par.getAttribute("data-path");
}

function extraNetworksTreeProcessDirectoryClick(event, btn, tabname, tab_id) {
    /**
     * Processes `onclick` events when user clicks on directories in tree.
     *
     * Here is how the tree reacts to clicks for various states:
     * unselected unopened directory: Diretory is selected and expanded.
     * unselected opened directory: Directory is selected.
     * selected opened directory: Directory is collapsed and deselected.
     * chevron is clicked: Directory is expanded or collapsed. Selected state unchanged.
     *
     * @param event     The generated event.
     * @param btn       The clicked `action-list-item` button.
     * @param tabname   The name of the active tab in the sd webui. Ex: txt2img, img2img, etc.
     * @param tab_id    The id of the active extraNetworks tab. Ex: lora, checkpoints, etc.
     */
    var ul = btn.nextElementSibling;
    // This is the actual target that the user clicked on within the target button.
    // We use this to detect if the chevron was clicked.
    var true_targ = event.target;

    function _expand_or_collapse(_ul, _btn) {
        // Expands <ul> if it is collapsed, collapses otherwise. Updates button attributes.
        if (_ul.hasAttribute("data-hidden")) {
            _ul.removeAttribute("data-hidden");
            _btn.setAttribute("expanded", "true");
        } else {
            _ul.setAttribute("data-hidden", "");
            _btn.setAttribute("expanded", "false");
        }
    }

    function _remove_selected_from_all() {
        // Removes the `selected` attribute from all buttons.
        var sels = document.querySelectorAll("button.action-list-content");
        [...sels].forEach(el => {
            el.removeAttribute("selected");
        });
    }

    function _select_button(_btn) {
        // Removes `selected` attribute from all buttons then adds to passed button.
        _remove_selected_from_all();
        _btn.setAttribute("selected", "");
    }

    function _update_search(_tabname, _tab_id, _search_text) {
        // Update search input with select button's path.
        var search_input_elem = gradioApp().querySelector("#" + tabname + "_" + tab_id + "_extra_search");
        search_input_elem.value = _search_text;
        updateInput(search_input_elem);
    }


    // If user clicks on the chevron, then we do not select the folder.
    if (true_targ.matches(".action-list-item-action--leading, .action-list-item-action-chevron")) {
        _expand_or_collapse(ul, btn);
    } else {
        // User clicked anywhere else on the button.
        if (btn.hasAttribute("selected") && !ul.hasAttribute("data-hidden")) {
            // If folder is select and open, collapse and deselect button.
            _expand_or_collapse(ul, btn);
            btn.removeAttribute("selected");
            _update_search(tabname, tab_id, "");
        } else if (!(!btn.hasAttribute("selected") && !ul.hasAttribute("data-hidden"))) {
            // If folder is open and not selected, then we don't collapse; just select.
            // NOTE: Double inversion sucks but it is the clearest way to show the branching here.
            _expand_or_collapse(ul, btn);
            _select_button(btn, tabname, tab_id);
            _update_search(tabname, tab_id, btn.parentElement.getAttribute("data-path"));
        } else {
            // All other cases, just select the button.
            _select_button(btn, tabname, tab_id);
            _update_search(tabname, tab_id, btn.parentElement.getAttribute("data-path"));
        }
    }
}

function extraNetworksTreeOnClick(event, tabname, tab_id) {
    /**
     * Handles `onclick` events for buttons within an `extra-network-tree .action-list--tree`.
     *
     * Determines whether the clicked button in the tree is for a file entry or a directory
     * then calls the appropriate function.
     *
     * @param event     The generated event.
     * @param tabname   The name of the active tab in the sd webui. Ex: txt2img, img2img, etc.
     * @param tab_id    The id of the active extraNetworks tab. Ex: lora, checkpoints, etc.
     */
    var btn = event.currentTarget;
    var par = btn.parentElement;
    if (par.getAttribute("data-tree-entry-type") === "file") {
        extraNetworksTreeProcessFileClick(event, btn, tabname, tab_id);
    } else {
        extraNetworksTreeProcessDirectoryClick(event, btn, tabname, tab_id);
    }
}

var globalPopup = null;
var globalPopupInner = null;

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

var storedPopupIds = {};
function popupId(id) {
    if (!storedPopupIds[id]) {
        storedPopupIds[id] = gradioApp().getElementById(id);
    }

    popup(storedPopupIds[id]);
}

function extraNetworksShowMetadata(text) {
    var elem = document.createElement('pre');
    elem.classList.add('popup-metadata');
    elem.textContent = text;

    popup(elem);
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

function extraNetworksCopyCardPath(event, path) {
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

var extraPageUserMetadataEditors = {};

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

/**
 * Setup custom loading for this script.
 * We need to wait for all of our HTML to be generated in the extra networks tabs
 * before we can actually run the `setupExtraNetworks` function.
 * The `onUiLoaded` function actually runs before all of our extra network tabs are
 * finished generating. Thus we needed this new method.
 *
 */

var uiAfterScriptsCallbacks = [];
var uiAfterScriptsTimeout = null;
var executedAfterScripts = false;

function scheduleAfterScriptsCallbacks() {
    clearTimeout(uiAfterScriptsTimeout);
    uiAfterScriptsTimeout = setTimeout(function() {
        executeCallbacks(uiAfterScriptsCallbacks);
    }, 200);
}

document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m) {
        if (!executedAfterScripts &&
            gradioApp().querySelectorAll("[id$='_extra_search']").length == 8) {
            executedAfterScripts = true;
            scheduleAfterScriptsCallbacks();
        }
    });
    mutationObserver.observe(gradioApp(), {childList: true, subtree: true});
});

uiAfterScriptsCallbacks.push(setupExtraNetworks);
