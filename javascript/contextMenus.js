
let appendContextMenuOption;
let removeContextMenuOption;

onLoad(function(){
	let menuSpecs = new Map();

	let contextMenu, contextMenuList;
	(function() {
		let baseStyle = window.getComputedStyle(get_uiCurrentTab());
		contextMenu = document.createElement('nav');
		contextMenu.id = "context-menu";
		contextMenu.style.background = baseStyle.background;
		contextMenu.style.color = baseStyle.color;
		contextMenu.style.fontFamily = baseStyle.fontFamily;

		contextMenuList = document.createElement('ul');
		contextMenuList.className = 'context-menu-items';
		contextMenu.append(contextMenuList);
	})();

	function display(e,menuEntries) {
		let posx = e.clientX + document.body.scrollLeft + document.documentElement.scrollLeft;
		let posy = e.clientY + document.body.scrollTop + document.documentElement.scrollTop;
		contextMenu.style.top = posy+'px';
		contextMenu.style.left = posx+'px'; 

		contextMenuList.innerHTML="";
		menuEntries.forEach(function(entry){
			let a = document.createElement('a');
			a.innerHTML = entry.name;
			a.addEventListener("click", entry.func);
			contextMenuList.append(a);
		})

		gradioApp().getRootNode().appendChild(contextMenu);

		let menuWidth = contextMenu.offsetWidth + 4;
		let menuHeight = contextMenu.offsetHeight + 4;

		let windowWidth = window.innerWidth;
		let windowHeight = window.innerHeight;

		if ((windowWidth - posx) < menuWidth) {
			contextMenu.style.left = windowWidth - menuWidth + "px";
		}

		if ((windowHeight - posy) < menuHeight) {
			contextMenu.style.top = windowHeight - menuHeight + "px";
		}
	}

	function append(selector,name,func){
		var items = menuSpecs.get(selector);
		if(!items) menuSpecs.set(selector,items=[]);

		let item = {
			owner:selector,
			name:name,
			func:func
		};

		items.push(item);
		return item;
	}

	function remove(item) {
		var items = menuSpecs.get(item.owner);
		if (!items) return;

		var i = items.indexOf(item);
		if (i < 0) return;
		items.splice(i, 1);

		if (!items.length) menuSpecs.remove(item.owner);
	}

	gradioApp().addEventListener("click", function(e) {
		let source = e.composedPath()[0];
		if(source.id && source.id.indexOf('check_progress')>-1) {
			return;
		}

		contextMenu.remove();
	});

	gradioApp().addEventListener("contextmenu", function(e) {
		contextMenu.remove();
		menuSpecs.forEach(function(v,k) {
			if(e.composedPath()[0].matches(k)) {
				display(e,v);
				e.preventDefault();
				return;
			}
		})
	});

	appendContextMenuOption = append;
	removeContextMenuOption = remove;
});


onLoad(function(){
	//Start example Context Menu Items
	var id;
	let generateOnRepeat = function(genId,intId){
		let genBtn = gradioApp().querySelector(genId);
		let intBtn = gradioApp().querySelector(intId);
		if(!intBtn.offsetParent) genBtn.click();

		clearInterval(id);
		id = setInterval(function(){
			if(!intBtn.offsetParent)genBtn.click();
		}, 500);
	}

	appendContextMenuOption('#txt2img_generate','Generate forever',function(){
		generateOnRepeat('#txt2img_generate','#txt2img_interrupt');
	});
	appendContextMenuOption('#img2img_generate','Generate forever',function(){
		generateOnRepeat('#img2img_generate','#img2img_interrupt');
	});

	let cancelGenerateForever = function() {
		clearInterval(id); 
	};

	appendContextMenuOption('#txt2img_interrupt','Cancel generate forever',cancelGenerateForever);
	appendContextMenuOption('#txt2img_generate', 'Cancel generate forever',cancelGenerateForever);
	appendContextMenuOption('#img2img_interrupt','Cancel generate forever',cancelGenerateForever);
	appendContextMenuOption('#img2img_generate', 'Cancel generate forever',cancelGenerateForever);

	appendContextMenuOption('#roll','Roll three',
		function(){ 
			let btn = get_uiCurrentTabContent().querySelector('#roll');
			setTimeout(function(){btn.click()},100);
			setTimeout(function(){btn.click()},200);
			setTimeout(function(){btn.click()},300);
		}
	);

	//End example Context Menu Items
});
