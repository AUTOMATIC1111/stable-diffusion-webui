const activePromptTextarea = {};
let sortVal = 0;

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

const getENActiveTab = () => {
  if (gradioApp().getElementById('tab_txt2img').style.display === 'block') return 'txt2img';
  if (gradioApp().getElementById('tab_img2img').style.display === 'block') return 'img2img';
  if (gradioApp().getElementById('tab_control').style.display === 'block') return 'control';
  return '';
};

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
  console.log('showCardDetails', event);
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
  const replaceOutsideBrackets = (input, target, replacement) => input.split(/(<[^>]*>|\{[^}]*\})/g).map((part, i) => {
    if (i % 2 === 0) return part.split(target).join(replacement); // Only replace in the parts that are not inside brackets (which are at even indices)
    return part;
  }).join('');

  const clickTag = (e, tag) => {
    e.preventDefault();
    e.stopPropagation();
    const textarea = activePromptTextarea[getENActiveTab()];
    let new_prompt = textarea.value;
    new_prompt = replaceOutsideBrackets(new_prompt, ` ${tag}`, ''); // try to remove tag
    new_prompt = replaceOutsideBrackets(new_prompt, `${tag} `, '');
    if (new_prompt === textarea.value) new_prompt += ` ${tag}`; // if not removed, then append it
    textarea.value = new_prompt;
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

function getCardsForActivePage() {
  const pagename = getENActivePage();
  if (!pagename) return [];
  const allCards = Array.from(gradioApp().querySelectorAll('.extra-network-cards > .card'));
  const cards = allCards.filter((el) => el.dataset.page.toLowerCase().includes(pagename.toLowerCase()));
  log('getCardsForActivePage', pagename, cards.length);
  return allCards;
}

async function filterExtraNetworksForTab(searchTerm) {
  let found = 0;
  let items = 0;
  const t0 = performance.now();
  const pagename = getENActivePage();
  if (!pagename) return;
  const allPages = Array.from(gradioApp().querySelectorAll('.extra-network-cards'));
  const pages = allPages.filter((el) => el.id.toLowerCase().includes(pagename.toLowerCase()));
  for (const pg of pages) {
    const cards = Array.from(pg.querySelectorAll('.card') || []);

    // We will always have as many items as cards
    items += cards.length;

    // Reset the results to show all cards if the search term is empty
    if (searchTerm === '') {
      cards.forEach((elem) => {
        elem.style.display = '';
      });
    } else {
      // Do not account for case or whitespace
      searchTerm = searchTerm.toLowerCase().trim();

      // If the searchTerm starts with "r#", then we are using regex search
      if (searchTerm.startsWith('r#')) {
        searchTerm = searchTerm.substring(2);

        // Insensitive regex search based on the searchTerm

        // The regex can be invalid -> then it will error out of this function, so the timing log will be missing, instead the error will be logged to console
        const re = new RegExp(searchTerm, 'i');

        cards.forEach((elem) => {
          // Construct the search text, which is the concatenation of all data elements with a prefix to make it unique
          // This combined text allows to exclude search terms for example by using negative lookahead
          if (re.test(`filename: ${elem.dataset.filename}|name: ${elem.dataset.name}|tags: ${elem.dataset.tags}`)) {
            elem.style.display = '';
            found += 1;
          } else {
            elem.style.display = 'none';
          }
        });
      } else {
        // If we are not using regex search, we still use an extended syntax to allow for searching for multiple keywords, or also excluding keywords
        // Keywords are separated by |, and keywords that should be excluded are prefixed with -
        const searchList = searchTerm.split('|').filter((s) => s !== '' && !s.startsWith('-')).map((s) => s.trim());
        const excludeList = searchTerm.split('|').filter((s) => s !== '' && s.trim().startsWith('-')).map((s) => s.trim().substring(1).trim());
        // In addition, both the searchList, and exclude List can be separated by &, which means that all keywords in the searchList must be present, and none of the excludeList
        // So we construct an array of arrays, which we will then use to filter the cards
        const searchListAll = searchList.map((s) => s.split('&').map((t) => t.trim()));
        const excludeListAll = excludeList.map((s) => s.split('&').map((t) => t.trim()));

        cards.forEach((elem) => {
          let text = '';
          if (elem.dataset.filename) text += `${elem.dataset.filename} `;
          if (elem.dataset.name) text += `${elem.dataset.name} `;
          if (elem.dataset.tags) text += `${elem.dataset.tags} `;
          text = text.toLowerCase().replace('models--', 'diffusers').replaceAll('\\', '/');
          if (
            // In searchListAll we have a list of lists, in the sublist, every keyword must be present
            // In the top level list, at least one sublist must be present
            // In excludeListAll we have a list of lists, in the sublist, the keywords may not appear together
            // In the top level list, none of the sublists must be present
            searchListAll.some((sl) => sl.every((st) => text.includes(st))) && !excludeListAll.some((el) => el.every((et) => text.includes(et)))
          ) {
            elem.style.display = '';
            found += 1;
          } else {
            elem.style.display = 'none';
          }
        });
      }
    }
  }
  const t1 = performance.now();
  if (searchTerm !== '') log(`filterExtraNetworks: text=${searchTerm} items=${items} match=${found} time=${Math.round(1000 * (t1 - t0)) / 1000000}`);
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

function sortExtraNetworks() {
  const sortDesc = ['Name [A-Z]', 'Name [Z-A]', 'Date [Newest]', 'Date [Oldest]', 'Size [Largest]', 'Size [Smallest]'];
  const pagename = getENActivePage();
  if (!pagename) return 'sort error: unknown page';
  const allPages = Array.from(gradioApp().querySelectorAll('.extra-network-cards'));
  const pages = allPages.filter((el) => el.id.toLowerCase().includes(pagename.toLowerCase()));
  let num = 0;
  for (const pg of pages) {
    const cards = Array.from(pg.querySelectorAll('.card') || []);
    num = cards.length;
    if (num === 0) return 'sort: no cards';
    cards.sort((a, b) => { // eslint-disable-line no-loop-func
      switch (sortVal) {
        case 0: return a.dataset.name ? a.dataset.name.localeCompare(b.dataset.name) : 0;
        case 1: return b.dataset.name ? b.dataset.name.localeCompare(a.dataset.name) : 0;
        case 2: return a.dataset.mtime && !isNaN(a.dataset.mtime) ? parseFloat(b.dataset.mtime) - parseFloat(a.dataset.mtime) : 0;
        case 3: return b.dataset.mtime && !isNaN(b.dataset.mtime) ? parseFloat(a.dataset.mtime) - parseFloat(b.dataset.mtime) : 0;
        case 4: return a.dataset.size && !isNaN(a.dataset.size) ? parseFloat(b.dataset.size) - parseFloat(a.dataset.size) : 0;
        case 5: return b.dataset.size && !isNaN(b.dataset.size) ? parseFloat(a.dataset.size) - parseFloat(b.dataset.size) : 0;
      }
      return 0;
    });
    for (const card of cards) pg.appendChild(card);
  }
  const desc = sortDesc[sortVal];
  sortVal = (sortVal + 1) % sortDesc.length;
  log('sortExtraNetworks', pagename, num, desc);
  return `sort page ${pagename} cards ${num} by ${desc}`;
}

function refreshENInput(tabname) {
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

  if (button.classList.contains('search-all')) {
    searchTextarea.value = '';
  } else {
    searchTextarea.value = `${button.textContent.trim()}/`;
  }
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

function quickApplyStyle() {
  const tabname = getENActiveTab();
  const btnApply = gradioApp().getElementById(`${tabname}_extra_apply`);
  if (btnApply) btnApply.click();
}

function quickSaveStyle() {
  const tabname = getENActiveTab();
  const btnSave = gradioApp().getElementById(`${tabname}_extra_quicksave`);
  if (btnSave) btnSave.click();
}

let enDirty = false;
function closeDetailsEN(args) {
  // log('closeDetailsEN');
  enDirty = true;
  const tabname = getENActiveTab();
  const btnClose = gradioApp().getElementById(`${tabname}_extra_details_close`);
  if (btnClose) setTimeout(() => btnClose.click(), 100);
  const btnRefresh = gradioApp().getElementById(`${tabname}_extra_refresh`);
  if (btnRefresh && enDirty) setTimeout(() => btnRefresh.click(), 100);
  return args;
}

function refeshDetailsEN(args) {
  log(`refeshDetailsEN: ${enDirty}`);
  const tabname = getENActiveTab();
  const btnRefresh = gradioApp().getElementById(`${tabname}_extra_refresh`);
  if (btnRefresh && enDirty) setTimeout(() => btnRefresh.click(), 100);
  enDirty = false;
  return args;
}

// refresh on en show
function refreshENpage() {
  if (getCardsForActivePage().length === 0) {
    log('refreshENpage');
    const tabname = getENActiveTab();
    const btnRefresh = gradioApp().getElementById(`${tabname}_extra_refresh`);
    if (btnRefresh) btnRefresh.click();
  }
}

// init
function setupExtraNetworksForTab(tabname) {
  let tabs = gradioApp().querySelector(`#${tabname}_extra_tabs`);
  if (tabs) tabs.classList.add('extra-networks');
  const en = gradioApp().getElementById(`${tabname}_extra_networks`);
  tabs = gradioApp().querySelector(`#${tabname}_extra_tabs > div`);
  if (!tabs) return;

  // buttons
  const btnRefresh = gradioApp().getElementById(`${tabname}_extra_refresh`);
  const btnScan = gradioApp().getElementById(`${tabname}_extra_scan`);
  const btnSave = gradioApp().getElementById(`${tabname}_extra_save`);
  const btnClose = gradioApp().getElementById(`${tabname}_extra_close`);
  const btnSort = gradioApp().getElementById(`${tabname}_extra_sort`);
  const btnView = gradioApp().getElementById(`${tabname}_extra_view`);
  const btnModel = gradioApp().getElementById(`${tabname}_extra_model`);
  const btnApply = gradioApp().getElementById(`${tabname}_extra_apply`);
  const buttons = document.createElement('span');
  buttons.classList.add('buttons');
  if (btnRefresh) buttons.appendChild(btnRefresh);
  if (btnModel) buttons.appendChild(btnModel);
  if (btnApply) buttons.appendChild(btnApply);
  if (btnScan) buttons.appendChild(btnScan);
  if (btnSave) buttons.appendChild(btnSave);
  if (btnSort) buttons.appendChild(btnSort);
  if (btnView) buttons.appendChild(btnView);
  if (btnClose) buttons.appendChild(btnClose);
  btnModel.onclick = () => btnModel.classList.toggle('toolbutton-selected');
  tabs.appendChild(buttons);

  // details
  const detailsImg = gradioApp().getElementById(`${tabname}_extra_details_img`);
  const detailsClose = gradioApp().getElementById(`${tabname}_extra_details_close`);
  if (detailsImg && detailsClose) {
    detailsImg.title = 'Close details';
    detailsImg.onclick = () => detailsClose.click();
  }

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
    searchTimer = setTimeout(async () => {
      await filterExtraNetworksForTab(txtSearchValue.value.toLowerCase());
      searchTimer = null;
    }, 100);
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
  if (!en) return;
  const intersectionObserver = new IntersectionObserver((entries) => {
    for (const el of Array.from(gradioApp().querySelectorAll('.extra-networks-page'))) {
      el.style.height = `${window.opts.extra_networks_height}vh`;
      el.parentElement.style.width = '-webkit-fill-available';
    }
    if (entries[0].intersectionRatio > 0) {
      refreshENpage();
      if (window.opts.extra_networks_card_cover === 'cover') {
        en.style.transition = '';
        en.style.zIndex = 100;
        en.style.top = '13em';
        en.style.position = 'absolute';
        en.style.right = 'unset';
        en.style.width = 'unset';
        en.style.height = 'unset';
        gradioApp().getElementById(`${tabname}_settings`).parentNode.style.width = 'unset';
      } else if (window.opts.extra_networks_card_cover === 'sidebar') {
        en.style.zIndex = 100;
        en.style.position = 'absolute';
        en.style.right = '0';
        en.style.top = '13em';
        en.style.height = '-webkit-fill-available';
        en.style.transition = 'width 0.3s ease';
        en.style.width = `${window.opts.extra_networks_sidebar_width}vw`;
        gradioApp().getElementById(`${tabname}_settings`).parentNode.style.width = `${100 - 2 - window.opts.extra_networks_sidebar_width}vw`;
      } else {
        en.style.transition = '';
        en.style.zIndex = 0;
        en.style.top = 0;
        en.style.position = 'relative';
        en.style.right = 'unset';
        en.style.width = 'unset';
        en.style.height = 'unset';
        gradioApp().getElementById(`${tabname}_settings`).parentNode.style.width = 'unset';
      }
    } else {
      if (window.opts.extra_networks_card_cover === 'sidebar') en.style.width = 0;
      gradioApp().getElementById(`${tabname}_settings`).parentNode.style.width = 'unset';
    }
  });
  intersectionObserver.observe(en); // monitor visibility of
}

async function setupExtraNetworks() {
  setupExtraNetworksForTab('txt2img');
  setupExtraNetworksForTab('img2img');
  setupExtraNetworksForTab('control');

  function registerPrompt(tabname, id) {
    const textarea = gradioApp().querySelector(`#${id} > label > textarea`);
    if (!textarea) return;
    if (!activePromptTextarea[tabname]) activePromptTextarea[tabname] = textarea;
    textarea.addEventListener('focus', () => { activePromptTextarea[tabname] = textarea; });
  }

  registerPrompt('txt2img', 'txt2img_prompt');
  registerPrompt('txt2img', 'txt2img_neg_prompt');
  registerPrompt('img2img', 'img2img_prompt');
  registerPrompt('img2img', 'img2img_neg_prompt');
  registerPrompt('control', 'control_prompt');
  registerPrompt('control', 'control_neg_prompt');
  log('initExtraNetworks');
}

onUiLoaded(setupExtraNetworks);
