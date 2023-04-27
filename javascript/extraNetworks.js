
function setupExtraNetworksForTab(tabname){
    gradioApp().querySelector('#'+tabname+'_extra_tabs').classList.add('extra-networks')

    var tabs = gradioApp().querySelector('#'+tabname+'_extra_tabs > div')
    var search = gradioApp().querySelector('#'+tabname+'_extra_search textarea')
    var refresh = gradioApp().getElementById(tabname+'_extra_refresh')
    var descriptInput = gradioApp().getElementById(tabname+ '_description_input')

    search.classList.add('search')
    tabs.appendChild(search)
    tabs.appendChild(refresh)
    tabs.appendChild(descriptInput)
    
    search.addEventListener("input", function(evt){
        searchTerm = search.value.toLowerCase()

        gradioApp().querySelectorAll('#'+tabname+'_extra_tabs div.card').forEach(function(elem){
            text = elem.querySelector('.name').textContent.toLowerCase() + " " + elem.querySelector('.search_term').textContent.toLowerCase()
            elem.style.display = text.indexOf(searchTerm) == -1 ? "none" : ""
        })
    });
}

var activePromptTextarea = {};

function setupExtraNetworks(){
    setupExtraNetworksForTab('txt2img')
    setupExtraNetworksForTab('img2img')

    function registerPrompt(tabname, id){
        var textarea = gradioApp().querySelector("#" + id + " > label > textarea");

        if (! activePromptTextarea[tabname]){
            activePromptTextarea[tabname] = textarea
        }

		textarea.addEventListener("focus", function(){
            activePromptTextarea[tabname] = textarea;
		});
    }

    registerPrompt('txt2img', 'txt2img_prompt')
    registerPrompt('txt2img', 'txt2img_neg_prompt')
    registerPrompt('img2img', 'img2img_prompt')
    registerPrompt('img2img', 'img2img_neg_prompt')
}

onUiLoaded(setupExtraNetworks)

var re_extranet   =    /<([^:]+:[^:]+):[\d\.]+>/;
var re_extranet_g = /\s+<([^:]+:[^:]+):[\d\.]+>/g;

function tryToRemoveExtraNetworkFromPrompt(textarea, text){
    var m = text.match(re_extranet)
    if(! m) return false

    var partToSearch = m[1]
    var replaced = false
    var newTextareaText = textarea.value.replaceAll(re_extranet_g, function(found, index){
        m = found.match(re_extranet);
        if(m[1] == partToSearch){
            replaced = true;
            return ""
        }
        return found;
    })

    if(replaced){
        textarea.value = newTextareaText
        return true;
    }

    return false
}

function cardClicked(tabname, textToAdd, allowNegativePrompt){
    var textarea = allowNegativePrompt ? activePromptTextarea[tabname] : gradioApp().querySelector("#" + tabname + "_prompt > label > textarea")

    if(! tryToRemoveExtraNetworkFromPrompt(textarea, textToAdd)){
        textarea.value = textarea.value + opts.extra_networks_add_text_separator + textToAdd
    }

    updateInput(textarea)
    stopPropagation()
}

function saveCardPreview(event, tabname, filename){
    var textarea = gradioApp().querySelector("#" + tabname + '_preview_filename  > label > textarea')
    var button = gradioApp().getElementById(tabname + '_save_preview')

    textarea.value = filename
    updateInput(textarea)

    button.click()

    event.stopPropagation()
    event.preventDefault()
}

function saveCardDescription(event, tabname, filename, currentView, itemName, itemDescript){
    var filename_textarea = gradioApp().querySelector("#" + tabname + '_description_filename  > label > textarea')
    var description_textarea = gradioApp().querySelector("#" + tabname+ '_description_input > label > textarea')
    var button = gradioApp().getElementById(tabname + '_save_description')
    var descrip_current_textarea_in_advanced = gradioApp().querySelector("#tab_" + tabname + " textarea.desc-text[data-desc-card-name='" + itemName + "']")

    filename_textarea.value = filename

    if (currentView=="advanced" && descrip_current_textarea_in_advanced.value){
        updateInput(descrip_current_textarea_in_advanced)
        description_textarea.value = descrip_current_textarea_in_advanced.value 
    }else if(currentView=="advanced"){
        description_textarea.value = itemDescript
        // in advanced view, if indivisual textarea has issue when try to save, reverse to previous description 
    }
    toggleEditSwitch(event, tabname, currentView, itemName)
    updateInput(filename_textarea )
    updateInput(description_textarea)
    
    button.click()

    event.stopPropagation()
    event.preventDefault()
}

function readCardDescription(event, tabname, filename, currentView, itemName, itemDescript){
    var filename_textarea = gradioApp().querySelector("#" + tabname + '_description_filename  > label > textarea')
    var description_textarea = gradioApp().querySelector("#" + tabname+ '_description_input > label > textarea')
    var button = gradioApp().getElementById(tabname + '_read_description')
    var descrip_current_textarea_in_advanced = gradioApp().querySelector("#tab_" + tabname + " textarea.desc-text[data-desc-card-name='" + itemName + "']")

    filename_textarea.value = filename
    description_textarea.value = itemDescript

    updateInput(filename_textarea)
    updateInput(description_textarea)

    if (currentView=="advanced" && descrip_current_textarea_in_advanced){
        descrip_current_textarea_in_advanced.value = description_textarea.value 
        updateInput(descrip_current_textarea_in_advanced)
    }
    
    button.click()

    descrip_current_textarea_in_advanced.focus();

    event.stopPropagation()
    event.preventDefault()
}

function toggleEditSwitch(event, tabname, currentView, itemName){
    var descrip_current_textarea_in_advanced = gradioApp().querySelector("#tab_" + tabname + " textarea.desc-text[data-desc-card-name='" + itemName + "']")
    var toggle_action = gradioApp().querySelector("#tab_" + tabname + " .card[data-card-name='" + itemName + "'] .card-info .description-actions")
    var toggle_list = gradioApp().querySelectorAll("#tab_" + tabname + " .card[data-card-name='" + itemName + "'] .card-info .description-actions ul.actions-btns-list")

    if (currentView=="advanced" && descrip_current_textarea_in_advanced){
        descrip_current_textarea_in_advanced.readOnly = !descrip_current_textarea_in_advanced.readOnly
        toggle_action.classList.toggle("toggle-off")
        toggle_list.forEach(function(toggle_list) {
            toggle_list.classList.toggle("toggle-off")
        })
        if (!descrip_current_textarea_in_advanced.readOnly) {
            descrip_current_textarea_in_advanced.focus();
        }
    }
    event.preventDefault()
}

function extraNetworksSearchButton(tabs_id, event){
    searchTextarea = gradioApp().querySelector("#" + tabs_id + ' > div > textarea')
    button = event.target
    text = button.classList.contains("search-all") ? "" : button.textContent.trim()

    searchTextarea.value = text
    updateInput(searchTextarea)
}

var globalPopup = null;
var globalPopupInner = null;
function popup(contents){
    if(! globalPopup){
        globalPopup = document.createElement('div')
        globalPopup.onclick = function(){ globalPopup.style.display = "none"; };
        globalPopup.classList.add('global-popup');

        var close = document.createElement('div')
        close.classList.add('global-popup-close');
        close.onclick = function(){ globalPopup.style.display = "none"; };
        close.title = "Close";
        globalPopup.appendChild(close)

        globalPopupInner = document.createElement('div')
        globalPopupInner.onclick = function(event){ event.stopPropagation(); return false; };
        globalPopupInner.classList.add('global-popup-inner');
        globalPopup.appendChild(globalPopupInner)

        gradioApp().appendChild(globalPopup);
    }

    globalPopupInner.innerHTML = '';
    globalPopupInner.appendChild(contents);

    globalPopup.style.display = "flex";
}

function extraNetworksShowMetadata(text){
    elem = document.createElement('pre')
    elem.classList.add('popup-metadata');
    elem.textContent = text;

    popup(elem);
}

function requestGet(url, data, handler, errorHandler){
    var xhr = new XMLHttpRequest();
    var args = Object.keys(data).map(function(k){ return encodeURIComponent(k) + '=' + encodeURIComponent(data[k]) }).join('&')
    xhr.open("GET", url + "?" + args, true);

    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                try {
                    var js = JSON.parse(xhr.responseText);
                    handler(js)
                } catch (error) {
                    console.error(error);
                    errorHandler()
                }
            } else{
                errorHandler()
            }
        }
    };
    var js = JSON.stringify(data);
    xhr.send(js);
}

function extraNetworksRequestMetadata(event, extraPage, cardName){
    showError = function(){ extraNetworksShowMetadata("there was an error getting metadata"); }

    requestGet("./sd_extra_networks/metadata", {"page": extraPage, "item": cardName}, function(data){
        if(data && data.metadata){
            extraNetworksShowMetadata(data.metadata)
        } else{
            showError()
        }
    }, showError)

    event.stopPropagation()
}
