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
    gradioApp().querySelector('#' + tabname + '_extra_tabs').classList.add('extra-networks');

    var tabs = gradioApp().querySelector('#' + tabname + '_extra_tabs > div');
    var searchDiv = gradioApp().getElementById(tabname + '_extra_search');
    var search = searchDiv.querySelector('textarea');
    var sort = gradioApp().getElementById(tabname + '_extra_sort');
    var sortOrder = gradioApp().getElementById(tabname + '_extra_sortorder');
    var refresh = gradioApp().getElementById(tabname + '_extra_refresh');
    var showDirsDiv = gradioApp().getElementById(tabname + '_extra_show_dirs');
    var showDirs = gradioApp().querySelector('#' + tabname + '_extra_show_dirs input');
    var promptContainer = gradioApp().querySelector('.prompt-container-compact#' + tabname + '_prompt_container');
    var negativePrompt = gradioApp().querySelector('#' + tabname + '_neg_prompt');

    tabs.appendChild(searchDiv);
    tabs.appendChild(sort);
    tabs.appendChild(sortOrder);
    tabs.appendChild(refresh);
    tabs.appendChild(showDirsDiv);

    var applyFilter = function() {
        var searchTerm = search.value.toLowerCase();

        gradioApp().querySelectorAll('#' + tabname + '_extra_tabs div.card').forEach(function(elem) {
            var searchOnly = elem.querySelector('.search_only');
            var text = elem.querySelector('.name').textContent.toLowerCase() + " " + elem.querySelector('.search_term').textContent.toLowerCase();

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

    extraNetworksApplySort[tabname] = applySort;
    extraNetworksApplyFilter[tabname] = applyFilter;

    var showDirsUpdate = function() {
        var css = '#' + tabname + '_extra_tabs .extra-network-subdirs { display: none; }';
        toggleCss(tabname + '_extra_show_dirs_style', css, !showDirs.checked);
        localSet('extra-networks-show-dirs', showDirs.checked ? 1 : 0);
    };
    showDirs.checked = localGet('extra-networks-show-dirs', 1) == 1;
    showDirs.addEventListener("change", showDirsUpdate);
    showDirsUpdate();
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


function extraNetworksUrelatedTabSelected(tabname) { // called from python when user selects an unrelated tab (generate)
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

    function registerPrompt(tabname, id) {
        var textarea = gradioApp().querySelector("#" + id + " > label > textarea");

        if (!activePromptTextarea[tabname]) {
            activePromptTextarea[tabname] = textarea;
        }

        textarea.addEventListener("focus", function() {
            activePromptTextarea[tabname] = textarea;
        });
    }

    registerPrompt('txt2img', 'txt2img_prompt');
    registerPrompt('txt2img', 'txt2img_neg_prompt');
    registerPrompt('img2img', 'img2img_prompt');
    registerPrompt('img2img', 'img2img_neg_prompt');
}

onUiLoaded(setupExtraNetworks);

var re_extranet = /<([^:^>]+:[^:]+):[\d.]+>(.*)/;
var re_extranet_g = /<([^:^>]+:[^:]+):[\d.]+>/g;

function tryToRemoveExtraNetworkFromPrompt(textarea, text) {
    var m = text.match(re_extranet);
    var replaced = false;
    var newTextareaText;
    if (m) {
        var extraTextBeforeNet = opts.extra_networks_add_text_separator;
        var extraTextAfterNet = m[2];
        var partToSearch = m[1];
        var foundAtPosition = -1;
        newTextareaText = textarea.value.replaceAll(re_extranet_g, function(found, net, pos) {
            m = found.match(re_extranet);
            if (m[1] == partToSearch) {
                replaced = true;
                foundAtPosition = pos;
                return "";
            }
            return found;
        });

        if (foundAtPosition >= 0) {
            if (newTextareaText.substr(foundAtPosition, extraTextAfterNet.length) == extraTextAfterNet) {
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

function cardClicked(tabname, textToAdd, allowNegativePrompt) {
    var textarea = allowNegativePrompt ? activePromptTextarea[tabname] : gradioApp().querySelector("#" + tabname + "_prompt > label > textarea");

    if (!tryToRemoveExtraNetworkFromPrompt(textarea, textToAdd)) {
        textarea.value = textarea.value + opts.extra_networks_add_text_separator + textToAdd;
    }

    updateInput(textarea);
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

function extraNetworksSearchButton(tabs_id, event) {
    var searchTextarea = gradioApp().querySelector("#" + tabs_id + ' > label > textarea');
    var button = event.target;
    var text = button.classList.contains("search-all") ? "" : button.textContent.trim();

    searchTextarea.value = text;
    updateInput(searchTextarea);
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
