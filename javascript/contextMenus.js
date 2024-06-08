
var contextMenuInit = function() {
    let eventListenerApplied = false;
    let menuSpecs = new Map();

    const uid = function() {
        return Date.now().toString(36) + Math.random().toString(36).substring(2);
    };

    function showContextMenu(event, element, menuEntries) {
        let oldMenu = gradioApp().querySelector('#context-menu');
        if (oldMenu) {
            oldMenu.remove();
        }

        let baseStyle = window.getComputedStyle(uiCurrentTab);

        const contextMenu = document.createElement('nav');
        contextMenu.id = "context-menu";
        contextMenu.style.background = baseStyle.background;
        contextMenu.style.color = baseStyle.color;
        contextMenu.style.fontFamily = baseStyle.fontFamily;
        contextMenu.style.top = event.pageY + 'px';
        contextMenu.style.left = event.pageX + 'px';

        const contextMenuList = document.createElement('ul');
        contextMenuList.className = 'context-menu-items';
        contextMenu.append(contextMenuList);

        menuEntries.forEach(function(entry) {
            let contextMenuEntry = document.createElement('a');
            contextMenuEntry.innerHTML = entry['name'];
            contextMenuEntry.addEventListener("click", function() {
                entry['func']();
            });
            contextMenuList.append(contextMenuEntry);

        });

        gradioApp().appendChild(contextMenu);
    }

    function appendContextMenuOption(targetElementSelector, entryName, entryFunction) {

        var currentItems = menuSpecs.get(targetElementSelector);

        if (!currentItems) {
            currentItems = [];
            menuSpecs.set(targetElementSelector, currentItems);
        }
        let newItem = {
            id: targetElementSelector + '_' + uid(),
            name: entryName,
            func: entryFunction,
            isNew: true
        };

        currentItems.push(newItem);
        return newItem['id'];
    }

    function removeContextMenuOption(uid) {
        menuSpecs.forEach(function(v) {
            let index = -1;
            v.forEach(function(e, ei) {
                if (e['id'] == uid) {
                    index = ei;
                }
            });
            if (index >= 0) {
                v.splice(index, 1);
            }
        });
    }

    function addContextMenuEventListener() {
        if (eventListenerApplied) {
            return;
        }
        gradioApp().addEventListener("click", function(e) {
            if (!e.isTrusted) {
                return;
            }

            let oldMenu = gradioApp().querySelector('#context-menu');
            if (oldMenu) {
                oldMenu.remove();
            }
        });
        ['contextmenu', 'touchstart'].forEach((eventType) => {
            gradioApp().addEventListener(eventType, function(e) {
                let ev = e;
                if (eventType.startsWith('touch')) {
                    if (e.touches.length !== 2) return;
                    ev = e.touches[0];
                }
                let oldMenu = gradioApp().querySelector('#context-menu');
                if (oldMenu) {
                    oldMenu.remove();
                }
                menuSpecs.forEach(function(v, k) {
                    if (e.composedPath()[0].matches(k)) {
                        showContextMenu(ev, e.composedPath()[0], v);
                        e.preventDefault();
                    }
                });
            });
        });
        eventListenerApplied = true;

    }

    return [appendContextMenuOption, removeContextMenuOption, addContextMenuEventListener];
};

var initResponse = contextMenuInit();
var appendContextMenuOption = initResponse[0];
var removeContextMenuOption = initResponse[1];
var addContextMenuEventListener = initResponse[2];

(function() {
    //Start example Context Menu Items
    let generateOnRepeat = function(genbuttonid, interruptbuttonid) {
        let genbutton = gradioApp().querySelector(genbuttonid);
        let interruptbutton = gradioApp().querySelector(interruptbuttonid);
        if (!interruptbutton.offsetParent) {
            genbutton.click();
        }
        clearInterval(window.generateOnRepeatInterval);
        window.generateOnRepeatInterval = setInterval(function() {
            if (!interruptbutton.offsetParent) {
                genbutton.click();
            }
        },
        500);
    };

    let generateOnRepeat_txt2img = function() {
        generateOnRepeat('#txt2img_generate', '#txt2img_interrupt');
    };

    let generateOnRepeat_img2img = function() {
        generateOnRepeat('#img2img_generate', '#img2img_interrupt');
    };

    appendContextMenuOption('#txt2img_generate', 'Generate forever', generateOnRepeat_txt2img);
    appendContextMenuOption('#txt2img_interrupt', 'Generate forever', generateOnRepeat_txt2img);
    appendContextMenuOption('#img2img_generate', 'Generate forever', generateOnRepeat_img2img);
    appendContextMenuOption('#img2img_interrupt', 'Generate forever', generateOnRepeat_img2img);

    let cancelGenerateForever = function() {
        clearInterval(window.generateOnRepeatInterval);
    };

    appendContextMenuOption('#txt2img_interrupt', 'Cancel generate forever', cancelGenerateForever);
    appendContextMenuOption('#txt2img_generate', 'Cancel generate forever', cancelGenerateForever);
    appendContextMenuOption('#img2img_interrupt', 'Cancel generate forever', cancelGenerateForever);
    appendContextMenuOption('#img2img_generate', 'Cancel generate forever', cancelGenerateForever);

})();
//End example Context Menu Items

onAfterUiUpdate(addContextMenuEventListener);
