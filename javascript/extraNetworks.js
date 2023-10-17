const activePromptTextarea = {};

// helpers

const requestGet = (url, data, handler) => {
  const xhr = new XMLHttpRequest();
  const args = Object.keys(data).map((k) => `${encodeURIComponent(k)}=${encodeURIComponent(data[k])}`).join('&');
  xhr.open('GET', `${url}?${args}`, true);
  xhr.onreadystatechange = () => {
    if (xhr.readyState === 4) {
      if (xhr.status === 200) handler(JSON.parse(xhr.responseText));
      else console.error(`Request: url=${url} status=${xhr.status} err`);
    }
  };
  xhr.send(JSON.stringify(data));
};

const getENActiveTab = () => gradioApp().getElementById('tab_txt2img').style.display === 'block' ? 'txt2img' : 'img2img';

const getENActivePage = () => {
  const tabname = getENActiveTab();
  const page = gradioApp().querySelector(`#${tabname}_extra_networks > .tabs > .tab-nav > .selected`);
  const pageName = page ? page.innerText : '';
  const btnApply = gradioApp().getElementById(`${tabname}_extra_apply`);
  if (btnApply) btnApply.style.display = pageName === 'Style' ? 'inline-flex' : 'none';
  return pageName;
};

const setENState = (state) => {
  if (!state) return;
  state.tab = getENActiveTab();
  state.page = getENActivePage();
  // log('setENState', state);
  const el = gradioApp().querySelector(`#${state.tab}_extra_state  > label > textarea`);
  el.value = JSON.stringify(state);
  updateInput(el);
};

// methods

function showCardDetails(event) {
  const tabname = getENActiveTab();
  const btn = gradioApp().getElementById(`${tabname}_extra_details_btn`);
  btn.click();
  event.stopPropagation();
  event.preventDefault();
}

function getCardDetails(...args) {
  const el = event?.target?.parentElement?.parentElement;
  if (el?.classList?.contains('card')) setENState({ op: 'getCardDetails', item: el.dataset.name });
  else setENState({ op: 'getCardDetails', item: null });
  return [...args];
}

function readCardTags(el, tags) {
  const clickTag = (e, tag) => {
    e.preventDefault();
    e.stopPropagation();
    const textarea = activePromptTextarea[getENActiveTab()];
    if (textarea.value.indexOf(tag) !== -1) textarea.value = textarea.value.replace(tag, '');
    else textarea.value += ` ${tag}`;
    updateInput(textarea);
  };
  if (tags.length === 0) return;
  const cardTags = tags.split('|');
  if (!cardTags || cardTags.length === 0) return;
  const tagsEl = el.getElementsByClassName('tags')[0];
  if (!tagsEl?.children || tagsEl.children.length > 0) return;
  for (const tag of cardTags) {
    const span = document.createElement('span');
    span.classList.add('tag');
    span.textContent = tag;
    span.onclick = (e) => clickTag(e, tag);
    tagsEl.appendChild(span);
  }
}

function readCardDescription(page, item) {
  requestGet('/sd_extra_networks/description', { page, item }, (data) => {
    const tabname = getENActiveTab();
    const description = gradioApp().querySelector(`#${tabname}_description > label > textarea`);
    description.value = data?.description?.trim() || '';
    // description.focus();
    updateInput(description);
    setENState({ op: 'readCardDescription', page, item });
  });
}

async function filterExtraNetworksForTab(tabname, searchTerm) {
  let found = 0;
  let items = 0;
  const t0 = performance.now();
  const cards = Array.from(gradioApp().querySelectorAll(`#${tabname}_extra_tabs div.card`));
  cards.forEach((elem) => {
    items += 1;
    if (searchTerm === '') {
      elem.style.display = '';
    } else {
      let text = `${elem.querySelector('.name').textContent.toLowerCase()} ${elem.querySelector('.search_term').textContent}`;
      text = text.toLowerCase().replace('models--', 'Diffusers').replace('\\', '/');
      if (text.indexOf(searchTerm) === -1) {
        elem.style.display = 'none';
      } else {
        elem.style.display = '';
        found += 1;
      }
    }
  });
  const t1 = performance.now();
  if (found > 0) log(`filterExtraNetworks: text=${searchTerm} items=${items} match=${found} time=${Math.round(1000 * (t1 - t0)) / 1000000}`);
  else log(`filterExtraNetworks: text=all items=${items} time=${Math.round(1000 * (t1 - t0)) / 1000000}`);
}

function tryToRemoveExtraNetworkFromPrompt(textarea, text) {
  const re_extranet = /<([^:]+:[^:]+):[\d\.]+>/;
  const re_extranet_g = /\s+<([^:]+:[^:]+):[\d\.]+>/g;
  let m = text.match(re_extranet);
  let replaced = false;
  let newTextareaText;
  if (m) {
    const partToSearch = m[1];
    newTextareaText = textarea.value.replaceAll(re_extranet_g, (found) => {
      m = found.match(re_extranet);
      if (m[1] === partToSearch) {
        replaced = true;
        return '';
      }
      return found;
    });
  } else {
    newTextareaText = textarea.value.replaceAll(new RegExp(text, 'g'), (found) => {
      if (found === text) {
        replaced = true;
        return '';
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
  log('refreshExtraNetworks', tabname, gradioApp().querySelector(`#${tabname}_extra_networks textarea`)?.value);
  gradioApp().querySelector(`#${tabname}_extra_networks textarea`)?.dispatchEvent(new Event('input'));
}

function cardClicked(textToAdd, allowNegativePrompt) {
  const tabname = getENActiveTab();
  const textarea = allowNegativePrompt ? activePromptTextarea[tabname] : gradioApp().querySelector(`#${tabname}_prompt > label > textarea`);
  if (textarea.value.indexOf(textToAdd) !== -1) textarea.value = textarea.value.replace(textToAdd, '');
  else textarea.value += textToAdd;
  updateInput(textarea);
}

function extraNetworksSearchButton(event) {
  const tabname = getENActiveTab();
  const searchTextarea = gradioApp().querySelector(`#${tabname}_extra_search textarea`);
  const button = event.target;
  const text = button.classList.contains('search-all') ? '' : `${button.textContent.trim()}/`;
  searchTextarea.value = text;
  updateInput(searchTextarea);
}

let desiredStyle = '';
function selectStyle(name) {
  desiredStyle = name;
  const tabname = getENActiveTab();
  const button = gradioApp().querySelector(`#${tabname}_styles_select`);
  button.click();
}

function applyStyles(styles) {
  let newStyles = [];
  if (styles) newStyles = Array.isArray(styles) ? styles : [styles];
  const index = newStyles.indexOf(desiredStyle);
  if (index > -1) newStyles.splice(index, 1);
  else newStyles.push(desiredStyle);
  return newStyles.join('|');
}

// init

function setupExtraNetworksForTab(tabname) {
  gradioApp().querySelector(`#${tabname}_extra_tabs`).classList.add('extra-networks');
  const en = gradioApp().getElementById(`${tabname}_extra_networks`);
  const tabs = gradioApp().querySelector(`#${tabname}_extra_tabs > div`);

  // buttons
  const btnRefresh = gradioApp().getElementById(`${tabname}_extra_refresh`);
  const btnScan = gradioApp().getElementById(`${tabname}_extra_scan`);
  const btnSave = gradioApp().getElementById(`${tabname}_extra_save`);
  const btnClose = gradioApp().getElementById(`${tabname}_extra_close`);
  const btnModel = gradioApp().getElementById(`${tabname}_extra_model`);
  const btnApply = gradioApp().getElementById(`${tabname}_extra_apply`);
  const buttons = document.createElement('span');
  buttons.classList.add('buttons');
  if (btnRefresh) buttons.appendChild(btnRefresh);
  if (btnModel) buttons.appendChild(btnModel);
  if (btnApply) buttons.appendChild(btnApply);
  if (btnScan) buttons.appendChild(btnScan);
  if (btnSave) buttons.appendChild(btnSave);
  if (btnClose) buttons.appendChild(btnClose);
  btnModel.onclick = () => btnModel.classList.toggle('toolbutton-selected');
  tabs.appendChild(buttons);

  // search and description
  const div = document.createElement('div');
  div.classList.add('second-line');
  tabs.appendChild(div);
  const txtSearch = gradioApp().querySelector(`#${tabname}_extra_search`);
  const txtSearchValue = gradioApp().querySelector(`#${tabname}_extra_search textarea`);
  const txtDescription = gradioApp().getElementById(`${tabname}_description`);
  txtSearch.classList.add('search');
  txtDescription.classList.add('description');
  div.appendChild(txtSearch);
  div.appendChild(txtDescription);
  let searchTimer = null;
  txtSearchValue.addEventListener('input', (evt) => {
    if (searchTimer) clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      filterExtraNetworksForTab(tabname, txtSearchValue.value.toLowerCase());
      searchTimer = null;
    }, 150);
  });

  // card hover
  let hoverTimer = null;
  let previousCard = null;
  gradioApp().getElementById(`${tabname}_extra_tabs`).onmouseover = (e) => {
    const el = e.target.closest('.card'); // bubble-up to card
    if (!el || (el.title === previousCard)) return;
    if (!hoverTimer) {
      hoverTimer = setTimeout(() => {
        readCardDescription(el.dataset.page, el.dataset.name);
        readCardTags(el, el.dataset.tags);
        previousCard = el.title;
      }, 300);
    }
    el.onmouseout = () => {
      clearTimeout(hoverTimer);
      hoverTimer = null;
    };
  };

  // en style
  const intersectionObserver = new IntersectionObserver((entries) => {
    if (!en) return;
    for (const el of Array.from(gradioApp().querySelectorAll('.extra-networks-page'))) {
      el.style.height = `${window.opts.extra_networks_height}vh`;
      el.parentElement.style.width = '-webkit-fill-available';
    }
    if (entries[0].intersectionRatio > 0) {
      if (window.opts.extra_networks_card_cover === 'cover') {
        en.style.transition = '';
        en.style.zIndex = 100;
        en.style.position = 'absolute';
        en.style.right = 'unset';
        en.style.width = 'unset';
        en.style.height = 'unset';
        gradioApp().getElementById(`${tabname}_settings`).parentNode.style.width = 'unset';
      } else if (window.opts.extra_networks_card_cover === 'sidebar') {
        en.style.transition = 'width 0.2s ease';
        en.style.zIndex = 100;
        en.style.position = 'absolute';
        en.style.right = '0';
        en.style.width = `${window.opts.extra_networks_sidebar_width}vw`;
        en.style.height = '-webkit-fill-available';
        gradioApp().getElementById(`${tabname}_settings`).parentNode.style.width = `${100 - 2 - window.opts.extra_networks_sidebar_width}vw`;
      } else {
        en.style.transition = '';
        en.style.zIndex = 0;
        en.style.position = 'relative';
        en.style.right = 'unset';
        en.style.width = 'unset';
        en.style.height = 'unset';
        gradioApp().getElementById(`${tabname}_settings`).parentNode.style.width = 'unset';
      }
    } else {
      en.style.width = 0;
      gradioApp().getElementById(`${tabname}_settings`).parentNode.style.width = 'unset';
    }
  });
  intersectionObserver.observe(en); // monitor visibility of
}

function setupExtraNetworks() {
  setupExtraNetworksForTab('txt2img');
  setupExtraNetworksForTab('img2img');

  function registerPrompt(tabname, id) {
    const textarea = gradioApp().querySelector(`#${id} > label > textarea`);
    if (!activePromptTextarea[tabname]) activePromptTextarea[tabname] = textarea;
    textarea.addEventListener('focus', () => { activePromptTextarea[tabname] = textarea; });
  }

  registerPrompt('txt2img', 'txt2img_prompt');
  registerPrompt('txt2img', 'txt2img_neg_prompt');
  registerPrompt('img2img', 'img2img_prompt');
  registerPrompt('img2img', 'img2img_neg_prompt');
  log('initExtraNetworks');
}

onUiLoaded(setupExtraNetworks);
