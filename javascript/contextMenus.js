const contextMenuInit = () => {
  let eventListenerApplied = false;
  const menuSpecs = new Map();

  const uid = () => Date.now().toString(36) + Math.random().toString(36).substring(2);

  function showContextMenu(event, element, menuEntries) {
    const posx = event.clientX + document.body.scrollLeft + document.documentElement.scrollLeft;
    const posy = event.clientY + document.body.scrollTop + document.documentElement.scrollTop;
    const oldMenu = gradioApp().querySelector('#context-menu');
    if (oldMenu) oldMenu.remove();
    const contextMenu = document.createElement('nav');
    contextMenu.id = 'context-menu';
    contextMenu.style.top = `${posy}px`;
    contextMenu.style.left = `${posx}px`;
    const contextMenuList = document.createElement('ul');
    contextMenuList.className = 'context-menu-items';
    contextMenu.append(contextMenuList);
    menuEntries.forEach((entry) => {
      const contextMenuEntry = document.createElement('a');
      contextMenuEntry.innerHTML = entry.name;
      contextMenuEntry.addEventListener('click', (e) => entry.func());
      contextMenuList.append(contextMenuEntry);
    });
    gradioApp().appendChild(contextMenu);
    const menuWidth = contextMenu.offsetWidth + 4;
    const menuHeight = contextMenu.offsetHeight + 4;
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;
    if ((windowWidth - posx) < menuWidth) contextMenu.style.left = `${windowWidth - menuWidth}px`;
    if ((windowHeight - posy) < menuHeight) contextMenu.style.top = `${windowHeight - menuHeight}px`;
  }

  function appendContextMenuOption(targetElementSelector, entryName, entryFunction) {
    let currentItems = menuSpecs.get(targetElementSelector);
    if (!currentItems) {
      currentItems = [];
      menuSpecs.set(targetElementSelector, currentItems);
    }
    const newItem = {
      id: `${targetElementSelector}_${uid()}`,
      name: entryName,
      func: entryFunction,
      isNew: true,
    };
    currentItems.push(newItem);
    return newItem.id;
  }

  function removeContextMenuOption(id) {
    menuSpecs.forEach((v, k) => {
      let index = -1;
      v.forEach((e, ei) => {
        if (e.id === id) { index = ei; }
      });
      if (index >= 0) v.splice(index, 1);
    });
  }

  async function addContextMenuEventListener() {
    if (eventListenerApplied) return;
    log('initContextMenu');
    gradioApp().addEventListener('click', (e) => {
      if (!e.isTrusted) return;
      const oldMenu = gradioApp().querySelector('#context-menu');
      if (oldMenu) oldMenu.remove();
    });
    gradioApp().addEventListener('contextmenu', (e) => {
      const oldMenu = gradioApp().querySelector('#context-menu');
      if (oldMenu) oldMenu.remove();
      menuSpecs.forEach((v, k) => {
        if (e.composedPath()[0].matches(k)) {
          showContextMenu(e, e.composedPath()[0], v);
          e.preventDefault();
        }
      });
    });
    eventListenerApplied = true;
  }
  return [appendContextMenuOption, removeContextMenuOption, addContextMenuEventListener];
};

const initResponse = contextMenuInit();
const appendContextMenuOption = initResponse[0];
const removeContextMenuOption = initResponse[1];
const addContextMenuEventListener = initResponse[2];

function initContextMenu() {
  // Start example Context Menu Items
  const generateOnRepeat = (genbuttonid, interruptbuttonid) => {
    const genbutton = gradioApp().querySelector(genbuttonid);
    const busy = document.getElementById('progressbar')?.style.display === 'block';
    if (!busy) genbutton.click();
    clearInterval(window.generateOnRepeatInterval);
    window.generateOnRepeatInterval = setInterval(() => {
      const pbBusy = document.getElementById('progressbar')?.style.display === 'block';
      if (!pbBusy) genbutton.click();
    }, 500);
  };
  const cancelGenerateForever = () => clearInterval(window.generateOnRepeatInterval);

  appendContextMenuOption('#txt2img_generate', 'Generate forever', () => generateOnRepeat('#txt2img_generate', '#txt2img_interrupt'));
  appendContextMenuOption('#img2img_generate', 'Generate forever', () => generateOnRepeat('#img2img_generate', '#img2img_interrupt'));
  appendContextMenuOption('#txt2img_generate', 'Cancel generate forever', cancelGenerateForever);
  appendContextMenuOption('#img2img_generate', 'Cancel generate forever', cancelGenerateForever);
  appendContextMenuOption('#txt2img_generate', 'Show NVML overlay', initNVML);
  appendContextMenuOption('#txt2img_generate', 'Hide NVML overlay', disableNVML);
}

onUiLoaded(initContextMenu);
onAfterUiUpdate(() => addContextMenuEventListener());
