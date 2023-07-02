let globalPopup = null;
let globalPopupInner = null;
const activePromptTextarea = {};

function setupExtraNetworksForTab(tabname) {
  gradioApp().querySelector(`#${tabname}_extra_tabs`).classList.add('extra-networks');
  const tabs = gradioApp().querySelector(`#${tabname}_extra_tabs > div`);
  const search = gradioApp().querySelector(`#${tabname}_extra_search textarea`);
  const refresh = gradioApp().getElementById(`${tabname}_extra_refresh`);
  const description = gradioApp().getElementById(`${tabname}_description`);
  const close = gradioApp().getElementById(`${tabname}_extra_close`);
  search.classList.add('search');
  description.classList.add('description');
  tabs.appendChild(refresh);
  tabs.appendChild(close);
  tabs.appendChild(search);
  tabs.appendChild(description);
  search.addEventListener('input', (evt) => {
    searchTerm = search.value.toLowerCase();
    gradioApp().querySelectorAll(`#${tabname}_extra_tabs div.card`).forEach((elem) => {
      text = `${elem.querySelector('.name').textContent.toLowerCase()} ${elem.querySelector('.search_term').textContent.toLowerCase()}`;
      elem.style.display = text.indexOf(searchTerm) == -1 ? 'none' : '';
    });
  });
  intersectionObserver = new IntersectionObserver((entries) => {
    // if (entries[0].intersectionRatio <= 0) onHidden();
    const en = gradioApp().getElementById(`${tabname}_extra_networks`);
    if (entries[0].intersectionRatio > 0) {
      for (el of Array.from(gradioApp().querySelectorAll('.extra-network-cards'))) {
        const rect = el.getBoundingClientRect();
        en.style.transition = 'width 0.2s ease';
        if (rect.top > 0) {
          if (!en) return
          if (window.opts.extra_networks_card_cover == 'cover') {
            en.style.zIndex = 9999;
            en.style.position = 'absolute';
            en.style.right = 'unset';
            en.style.width = 'unset';
            el.style.height = document.body.offsetHeight - el.getBoundingClientRect().top + 'px'; 
            gradioApp().getElementById(`${tabname}_settings`).parentNode.style.width = 'unset'
          } if (window.opts.extra_networks_card_cover == 'sidebar') {
            en.style.zIndex = 0;
            en.style.position = 'absolute';
            en.style.right = '0';
            en.style.width = window.opts.extra_networks_sidebar_width + 'vw';
            el.style.height = gradioApp().getElementById(`${tabname}_settings`).offsetHeight - 90 + 'px'; 
            gradioApp().getElementById(`${tabname}_settings`).parentNode.style.width = 100 - 2 - window.opts.extra_networks_sidebar_width + 'vw';
          } else {
            en.style.zIndex = 0;
            en.style.position = 'relative';
            en.style.right = 'unset';
            en.style.width = 'unset';
            el.style.height = window.innerHeight - el.getBoundingClientRect().top + 'px'; 
            gradioApp().getElementById(`${tabname}_settings`).parentNode.style.width = 'unset'
          }
        }
      }
    } else {
      en.style.width = 0;
      gradioApp().getElementById(`${tabname}_settings`).parentNode.style.width = 'unset'
    }
  });
  intersectionObserver.observe(search); // monitor visibility of 
}

function setupExtraNetworks() {
  setupExtraNetworksForTab('txt2img');
  setupExtraNetworksForTab('img2img');
  
  function registerPrompt(tabname, id) {
    const textarea = gradioApp().querySelector(`#${id} > label > textarea`);
    if (!activePromptTextarea[tabname]) activePromptTextarea[tabname] = textarea;
    textarea.addEventListener('focus', () => {
      activePromptTextarea[tabname] = textarea;
    });
  }

  registerPrompt('txt2img', 'txt2img_prompt');
  registerPrompt('txt2img', 'txt2img_neg_prompt');
  registerPrompt('img2img', 'img2img_prompt');
  registerPrompt('img2img', 'img2img_neg_prompt');
}

onUiLoaded(setupExtraNetworks);
const re_extranet = /<([^:]+:[^:]+):[\d\.]+>/;
const re_extranet_g = /\s+<([^:]+:[^:]+):[\d\.]+>/g;

function tryToRemoveExtraNetworkFromPrompt(textarea, text) {
  var m = text.match(re_extranet);
  var replaced = false;
  var newTextareaText;
  if (m) {
      var partToSearch = m[1];
      newTextareaText = textarea.value.replaceAll(re_extranet_g, function(found) {
          m = found.match(re_extranet);
          if (m[1] == partToSearch) {
              replaced = true;
              return "";
          }
          return found;
      });
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

function refreshExtraNetworks(tabname) {
  gradioApp().querySelector(`#${tabname}_extra_networks textarea`)?.dispatchEvent(new Event('input'));
}

function cardClicked(tabname, textToAdd, allowNegativePrompt) {
  const textarea = allowNegativePrompt ? activePromptTextarea[tabname] : gradioApp().querySelector(`#${tabname}_prompt > label > textarea`);
  if (!tryToRemoveExtraNetworkFromPrompt(textarea, textToAdd)) textarea.value = textarea.value + opts.extra_networks_add_text_separator + textToAdd;
  updateInput(textarea);
}

function saveCardPreview(event, tabname, filename) {
  console.log('saveCardPreview', event, tabname, filename)
  const textarea = gradioApp().querySelector(`#${tabname}_preview_filename  > label > textarea`);
  const button = gradioApp().getElementById(`${tabname}_save_preview`);
  textarea.value = filename;
  updateInput(textarea);
  button.click();
  event.stopPropagation();
  event.preventDefault();
}

function saveCardDescription(event, tabname, filename, descript) {
  console.log('saveCardDescription', event, tabname, filename, descript);
  const textarea = gradioApp().querySelector(`#${tabname}_description_filename  > label > textarea`);
  const button = gradioApp().getElementById(`${tabname}_save_description`);
  const description = gradioApp().getElementById(`${tabname}_description`);
  textarea.value = filename;
  description.value = descript;
  updateInput(textarea);
  button.click();
  event.stopPropagation();
  event.preventDefault();
}

function readCardDescription(event, tabname, filename, descript, extraPage, cardName) {
  console.log('readCardDescription', event, tabname, filename, descript, extraPage, cardName);
  const textarea = gradioApp().querySelector(`#${tabname}_description_filename  > label > textarea`);
  const description = gradioApp().querySelector(`#${tabname}_description > label > textarea`);
  const button = gradioApp().getElementById(`${tabname}_read_description`);
  textarea.value = filename;
  description.value = descript;
  updateInput(textarea);
  updateInput(description);
  button.click();
  event.stopPropagation();
  event.preventDefault();
}

function extraNetworksSearchButton(tabs_id, event) {
  searchTextarea = gradioApp().querySelector(`#${tabs_id} > div > textarea`);
  button = event.target;
  text = button.classList.contains('search-all') ? '' : `/${button.textContent.trim()}/`;
  searchTextarea.value = text;
  updateInput(searchTextarea);
}

function popup(contents) {
  if (!globalPopup) {
    globalPopup = document.createElement('div');
    globalPopup.onclick = function () { globalPopup.style.display = 'none'; };
    globalPopup.classList.add('global-popup');
    const close = document.createElement('div');
    close.classList.add('global-popup-close');
    close.onclick = function () { globalPopup.style.display = 'none'; };
    close.title = 'Close';
    globalPopup.appendChild(close);
    globalPopupInner = document.createElement('div');
    globalPopupInner.onclick = function (event) { event.stopPropagation(); return false; };
    globalPopupInner.classList.add('global-popup-inner');
    globalPopup.appendChild(globalPopupInner);
    gradioApp().appendChild(globalPopup);
  }
  globalPopupInner.innerHTML = '';
  globalPopupInner.appendChild(contents);
  globalPopup.style.display = 'flex';
}

function readCardMetadata(event, extraPage, cardName) {
  requestGet('./sd_extra_networks/metadata', { page: extraPage, item: cardName }, (data) => {
    if (data?.metadata && (typeof(data?.metadata) === 'string')) {
      elem = document.createElement('pre');
      elem.classList.add('popup-metadata');
      elem.textContent = data.metadata;
      popup(elem);
    }
  }, () => {});
  event.stopPropagation();
  event.preventDefault();
}

function readCardInformation(event, extraPage, cardName) {
  requestGet('./sd_extra_networks/info', { page: extraPage, item: cardName }, (data) => {
    if (data?.info && (typeof(data?.info) === 'string')) {
      elem = document.createElement('pre');
      elem.classList.add('popup-metadata');
      elem.textContent = data.info;
      popup(elem);
    }
  }, () => {});
  event.stopPropagation();
  event.preventDefault();
}

function requestGet(url, data, handler, errorHandler) {
  const xhr = new XMLHttpRequest();
  const args = Object.keys(data).map((k) => `${encodeURIComponent(k)}=${encodeURIComponent(data[k])}`).join('&');
  xhr.open('GET', `${url}?${args}`, true);
  xhr.onreadystatechange = function () {
    if (xhr.readyState === 4) {
      if (xhr.status === 200) {
        try {
          const js = JSON.parse(xhr.responseText);
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
  const js = JSON.stringify(data);
  xhr.send(js);
}
