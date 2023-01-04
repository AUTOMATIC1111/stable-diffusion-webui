
contextMenuInit = function(){
  let eventListenerApplied=false;
  let menuSpecs = new Map();

  const uid = function(){
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
  }

  function showContextMenu(event,element,menuEntries){
    let posx = event.clientX + document.body.scrollLeft + document.documentElement.scrollLeft;
    let posy = event.clientY + document.body.scrollTop + document.documentElement.scrollTop;

    let oldMenu = gradioApp().querySelector('#context-menu')
    if(oldMenu){
      oldMenu.remove()
    }

    let tabButton = uiCurrentTab
    let baseStyle = window.getComputedStyle(tabButton)

    const contextMenu = document.createElement('nav')
    contextMenu.id = "context-menu"
    contextMenu.style.background = baseStyle.background
    contextMenu.style.color = baseStyle.color
    contextMenu.style.fontFamily = baseStyle.fontFamily
    contextMenu.style.top = posy+'px'
    contextMenu.style.left = posx+'px'



    const contextMenuList = document.createElement('ul')
    contextMenuList.className = 'context-menu-items';
    contextMenu.append(contextMenuList);

    menuEntries.forEach(function(entry){
      let contextMenuEntry = document.createElement('a')
      contextMenuEntry.innerHTML = entry['name']
      contextMenuEntry.addEventListener("click", function(e) {
        entry['func']();
      })
      contextMenuList.append(contextMenuEntry);

    })

    gradioApp().getRootNode().appendChild(contextMenu)

    let menuWidth = contextMenu.offsetWidth + 4;
    let menuHeight = contextMenu.offsetHeight + 4;

    let windowWidth = window.innerWidth;
    let windowHeight = window.innerHeight;

    if ( (windowWidth - posx) < menuWidth ) {
      contextMenu.style.left = windowWidth - menuWidth + "px";
    }

    if ( (windowHeight - posy) < menuHeight ) {
      contextMenu.style.top = windowHeight - menuHeight + "px";
    }

  }

  function appendContextMenuOption(targetElementSelector,entryName,entryFunction){

    currentItems = menuSpecs.get(targetElementSelector)

    if(!currentItems){
      currentItems = []
      menuSpecs.set(targetElementSelector,currentItems);
    }
    let newItem = {'id':targetElementSelector+'_'+uid(),
                   'name':entryName,
                   'func':entryFunction,
                   'isNew':true}

    currentItems.push(newItem)
    return newItem['id']
  }

  function removeContextMenuOption(uid){
    menuSpecs.forEach(function(v,k) {
      let index = -1
      v.forEach(function(e,ei){if(e['id']==uid){index=ei}})
      if(index>=0){
        v.splice(index, 1);
      }
    })
  }

  function addContextMenuEventListener(){
    if(eventListenerApplied){
      return;
    }
    gradioApp().addEventListener("click", function(e) {
      let source = e.composedPath()[0]
      if(source.id && source.id.indexOf('check_progress')>-1){
        return
      }

      let oldMenu = gradioApp().querySelector('#context-menu')
      if(oldMenu){
        oldMenu.remove()
      }
    });
    gradioApp().addEventListener("contextmenu", function(e) {
      let oldMenu = gradioApp().querySelector('#context-menu')
      if(oldMenu){
        oldMenu.remove()
      }
      menuSpecs.forEach(function(v,k) {
        if(e.composedPath()[0].matches(k)){
          showContextMenu(e,e.composedPath()[0],v)
          e.preventDefault()
          return
        }
      })
    });
    eventListenerApplied=true

  }

  return [appendContextMenuOption, removeContextMenuOption, addContextMenuEventListener]
}

initResponse = contextMenuInit();
appendContextMenuOption     = initResponse[0];
removeContextMenuOption     = initResponse[1];
addContextMenuEventListener = initResponse[2];

(function(){
  //Start example Context Menu Items
  let generateOnRepeat = function(genbuttonid,interruptbuttonid){
    let genbutton = gradioApp().querySelector(genbuttonid);
    let interruptbutton = gradioApp().querySelector(interruptbuttonid);
    if(!interruptbutton.offsetParent){
      genbutton.click();
    }
    clearInterval(window.generateOnRepeatInterval)
    window.generateOnRepeatInterval = setInterval(function(){
      if(!interruptbutton.offsetParent){
        genbutton.click();
      }
    },
    500)
  }

  appendContextMenuOption('#txt2img_generate','Generate forever',function(){
    generateOnRepeat('#txt2img_generate','#txt2img_interrupt');
  })
  appendContextMenuOption('#img2img_generate','Generate forever',function(){
    generateOnRepeat('#img2img_generate','#img2img_interrupt');
  })

  let cancelGenerateForever = function(){
    clearInterval(window.generateOnRepeatInterval)
  }

  appendContextMenuOption('#txt2img_interrupt','Cancel generate forever',cancelGenerateForever)
  appendContextMenuOption('#txt2img_generate', 'Cancel generate forever',cancelGenerateForever)
  appendContextMenuOption('#img2img_interrupt','Cancel generate forever',cancelGenerateForever)
  appendContextMenuOption('#img2img_generate', 'Cancel generate forever',cancelGenerateForever)

  appendContextMenuOption('#roll','Roll three',
    function(){
      let rollbutton = get_uiCurrentTabContent().querySelector('#roll');
      setTimeout(function(){rollbutton.click()},100)
      setTimeout(function(){rollbutton.click()},200)
      setTimeout(function(){rollbutton.click()},300)
    }
  )
})();
//End example Context Menu Items

onUiUpdate(function(){
  addContextMenuEventListener()
});
