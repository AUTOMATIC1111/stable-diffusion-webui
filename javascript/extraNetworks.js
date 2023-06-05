function setupExtraNetworksForTab(tabname) {
  gradioApp().querySelector(`#${tabname}_extra_tabs`).classList.add('extra-networks');
  const tabs = gradioApp().querySelector(`#${tabname}_extra_tabs > div`);
  const search = gradioApp().querySelector(`#${tabname}_extra_search textarea`);
  const refresh = gradioApp().getElementById(`${tabname}_extra_refresh`);
  const descriptInput = gradioApp().getElementById(`${tabname}_description_input`);
  const close = gradioApp().getElementById(`${tabname}_extra_close`);
  search.classList.add('search');
  tabs.appendChild(search);
  tabs.appendChild(refresh);
  tabs.appendChild(close);
  tabs.appendChild(descriptInput);
  search.addEventListener('input', (evt) => {
    searchTerm = search.value.toLowerCase();
    gradioApp().querySelectorAll(`#${tabname}_extra_tabs div.card`).forEach((elem) => {
      text = `${elem.querySelector('.name').textContent.toLowerCase()} ${elem.querySelector('.search_term').textContent.toLowerCase()}`;
      elem.style.display = text.indexOf(searchTerm) == -1 ? 'none' : '';
    });
  });
}

const activePromptTextarea = {};

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
  let m = text.match(re_extranet);
  if (!m) return false;
  const partToSearch = m[1];
  let replaced = false;
  const newTextareaText = textarea.value.replaceAll(re_extranet_g, (found, index) => {
    m = found.match(re_extranet);
    if (m[1] === partToSearch) {
      replaced = true;
      return '';
    }
    return found;
  });
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
  const textarea = gradioApp().querySelector(`#${tabname}_preview_filename  > label > textarea`);
  const button = gradioApp().getElementById(`${tabname}_save_preview`);
  textarea.value = filename;
  updateInput(textarea);
  button.click();
  event.stopPropagation();
  event.preventDefault();
}

function saveCardDescription(event, tabname, filename, descript) {
  const textarea = gradioApp().querySelector(`#${tabname}_description_filename  > label > textarea`);
  const button = gradioApp().getElementById(`${tabname}_save_description`);
  const description = gradioApp().getElementById(`${tabname}_description_input`);
  textarea.value = filename;
  description.value = descript;
  updateInput(textarea);
  button.click();
  event.stopPropagation();
  event.preventDefault();
}

function readCardDescription(event, tabname, filename, descript, extraPage, cardName) {
  const textarea = gradioApp().querySelector(`#${tabname}_description_filename  > label > textarea`);
  const description_textarea = gradioApp().querySelector(`#${tabname}_description_input > label > textarea`);
  const button = gradioApp().getElementById(`${tabname}_read_description`);
  textarea.value = filename;
  description_textarea.value = descript;
  updateInput(textarea);
  updateInput(description_textarea);
  button.click();
  event.stopPropagation();
  event.preventDefault();
}

function extraNetworksSearchButton(tabs_id, event) {
  searchTextarea = gradioApp().querySelector(`#${tabs_id} > div > textarea`);
  button = event.target;
  text = button.classList.contains('search-all') ? '' : button.textContent.trim();
  searchTextarea.value = text;
  updateInput(searchTextarea);
}

let globalPopup = null;
let globalPopupInner = null;
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
    if (data && data.metadata) {
      elem = document.createElement('pre');
      elem.classList.add('popup-metadata');
      elem.textContent = data.metadata;
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
