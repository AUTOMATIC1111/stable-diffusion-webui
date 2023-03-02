// various functions for interaction with ui.py not large enough to warrant putting them in separate files

function set_theme(theme){
    gradioURL = window.location.href
    if (!gradioURL.includes('?__theme=')) {
      window.location.replace(gradioURL + '?__theme=' + theme);
    }
}

function selected_gallery_index(){
    var buttons = gradioApp().querySelectorAll('[style="display: block;"].tabitem div[id$=_gallery] .gallery-item')
    var button = gradioApp().querySelector('[style="display: block;"].tabitem div[id$=_gallery] .gallery-item.\\!ring-2')

    var result = -1
    buttons.forEach(function(v, i){ if(v==button) { result = i } })

    return result
}

function extract_image_from_gallery(gallery){
    if(gallery.length == 1){
        return [gallery[0]]
    }

    index = selected_gallery_index()

    if (index < 0 || index >= gallery.length){
        return [null]
    }

    return [gallery[index]];
}

function args_to_array(args){
    res = []
    for(var i=0;i<args.length;i++){
        res.push(args[i])
    }
    return res
}

function switch_to_txt2img(){
    gradioApp().querySelector('#tabs').querySelectorAll('button')[0].click();

    return args_to_array(arguments);
}

function switch_to_img2img_tab(no){
    gradioApp().querySelector('#tabs').querySelectorAll('button')[1].click();
    gradioApp().getElementById('mode_img2img').querySelectorAll('button')[no].click();
}
function switch_to_img2img(){
    switch_to_img2img_tab(0);
    return args_to_array(arguments);
}

function switch_to_sketch(){
    switch_to_img2img_tab(1);
    return args_to_array(arguments);
}

function switch_to_inpaint(){
    switch_to_img2img_tab(2);
    return args_to_array(arguments);
}

function switch_to_inpaint_sketch(){
    switch_to_img2img_tab(3);
    return args_to_array(arguments);
}

function switch_to_inpaint(){
    gradioApp().querySelector('#tabs').querySelectorAll('button')[1].click();
    gradioApp().getElementById('mode_img2img').querySelectorAll('button')[2].click();

    return args_to_array(arguments);
}

function switch_to_extras(){
    gradioApp().querySelector('#tabs').querySelectorAll('button')[2].click();

    return args_to_array(arguments);
}

function get_tab_index(tabId){
    var res = 0

    gradioApp().getElementById(tabId).querySelector('div').querySelectorAll('button').forEach(function(button, i){		
        if(button.className.indexOf('bg-white') != -1)
            res = i
    })

    return res
}

function create_tab_index_args(tabId, args){
    var res = []
    for(var i=0; i<args.length; i++){
        res.push(args[i])
    }

    res[0] = get_tab_index(tabId)

    return res
}

function get_img2img_tab_index() {
    let res = args_to_array(arguments)
    res.splice(-2)
    res[0] = get_tab_index('mode_img2img')
    return res
}

function create_submit_args(args){
    res = []
    for(var i=0;i<args.length;i++){
        res.push(args[i])
    }

    // As it is currently, txt2img and img2img send back the previous output args (txt2img_gallery, generation_info, html_info) whenever you generate a new image.
    // This can lead to uploading a huge gallery of previously generated images, which leads to an unnecessary delay between submitting and beginning to generate.
    // I don't know why gradio is sending outputs along with inputs, but we can prevent sending the image gallery here, which seems to be an issue for some.
    // If gradio at some point stops sending outputs, this may break something
    if(Array.isArray(res[res.length - 3])){
        res[res.length - 3] = null
    }

    return res
}

function showSubmitButtons(tabname, show){
    gradioApp().getElementById(tabname+'_interrupt').style.display = show ? "none" : "block"
    gradioApp().getElementById(tabname+'_skip').style.display = show ? "none" : "block"
}

function submit(){
    rememberGallerySelection('txt2img_gallery')
    showSubmitButtons('txt2img', false)

    var id = randomId()
    requestProgress(id, gradioApp().getElementById('txt2img_gallery_container'), gradioApp().getElementById('txt2img_gallery'), function(){
        showSubmitButtons('txt2img', true)

    })

    var res = create_submit_args(arguments)

    res[0] = id

    return res
}

function submit_img2img(){
    rememberGallerySelection('img2img_gallery')
    showSubmitButtons('img2img', false)

    var id = randomId()
    requestProgress(id, gradioApp().getElementById('img2img_gallery_container'), gradioApp().getElementById('img2img_gallery'), function(){
        showSubmitButtons('img2img', true)
    })

    var res = create_submit_args(arguments)

    res[0] = id
    res[1] = get_tab_index('mode_img2img')

    return res
}

function modelmerger(){
    var id = randomId()
    requestProgress(id, gradioApp().getElementById('modelmerger_results_panel'), null, function(){})

    var res = create_submit_args(arguments)
    res[0] = id
    return res
}


function ask_for_style_name(_, prompt_text, negative_prompt_text) {
    name_ = prompt('Style name:')
    return [name_, prompt_text, negative_prompt_text]
}

function confirm_clear_prompt(prompt, negative_prompt) {
    if(confirm("Delete prompt?")) {
        prompt = ""
        negative_prompt = ""
    }

    return [prompt, negative_prompt]
}


promptTokecountUpdateFuncs = {}

function recalculatePromptTokens(name){
    if(promptTokecountUpdateFuncs[name]){
        promptTokecountUpdateFuncs[name]()
    }
}

function recalculate_prompts_txt2img(){
    recalculatePromptTokens('txt2img_prompt')
    recalculatePromptTokens('txt2img_neg_prompt')
    return args_to_array(arguments);
}

function recalculate_prompts_img2img(){
    recalculatePromptTokens('img2img_prompt')
    recalculatePromptTokens('img2img_neg_prompt')
    return args_to_array(arguments);
}




opts = {}
onUiUpdate(function(){
	if(Object.keys(opts).length != 0) return;

	json_elem = gradioApp().getElementById('settings_json')
	if(json_elem == null) return;

    var textarea = json_elem.querySelector('textarea')
    var jsdata = textarea.value
    opts = JSON.parse(jsdata)
    executeCallbacks(optionsChangedCallbacks);

    Object.defineProperty(textarea, 'value', {
        set: function(newValue) {
            var valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            var oldValue = valueProp.get.call(textarea);
            valueProp.set.call(textarea, newValue);

            if (oldValue != newValue) {
                opts = JSON.parse(textarea.value)
            }

            executeCallbacks(optionsChangedCallbacks);
        },
        get: function() {
            var valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            return valueProp.get.call(textarea);
        }
    });

    json_elem.parentElement.style.display="none"

    function registerTextarea(id, id_counter, id_button){
        var prompt = gradioApp().getElementById(id)
        var counter = gradioApp().getElementById(id_counter)
        var textarea = gradioApp().querySelector("#" + id + " > label > textarea");

        if(counter.parentElement == prompt.parentElement){
            return
        }

        prompt.parentElement.insertBefore(counter, prompt)
        counter.classList.add("token-counter")
        prompt.parentElement.style.position = "relative"

		promptTokecountUpdateFuncs[id] = function(){ update_token_counter(id_button); }
		textarea.addEventListener("input", promptTokecountUpdateFuncs[id]);
    }

    registerTextarea('txt2img_prompt', 'txt2img_token_counter', 'txt2img_token_button')
    registerTextarea('txt2img_neg_prompt', 'txt2img_negative_token_counter', 'txt2img_negative_token_button')
    registerTextarea('img2img_prompt', 'img2img_token_counter', 'img2img_token_button')
    registerTextarea('img2img_neg_prompt', 'img2img_negative_token_counter', 'img2img_negative_token_button')

    show_all_pages = gradioApp().getElementById('settings_show_all_pages')
    settings_tabs = gradioApp().querySelector('#settings div')
    if(show_all_pages && settings_tabs){
        settings_tabs.appendChild(show_all_pages)
        show_all_pages.onclick = function(){
            gradioApp().querySelectorAll('#settings > div').forEach(function(elem){
                elem.style.display = "block";
            })
			gradioApp().querySelectorAll('#settings > div > div > div').forEach(function(elem){
                elem.style.maxHeight  = "none";
            })		
        }
    }
	
	
	
	/* 	
	^ matches the start
	* matches any position
	$ matches the end
	*/
	
	/* auto grow textarea */
	gradioApp().querySelectorAll('[id $= "_prompt"] textarea, [id^="setting_"] textarea').forEach(function (elem) {
		elem.style.boxSizing = 'border-box';
		var offset = elem.offsetHeight - elem.clientHeight;
		elem.addEventListener('input', function (e) {
			e.target.style.minHeight = 'auto';
			e.target.style.minHeight = e.target.scrollHeight + offset + 'px';
		});
		
		elem.addEventListener('focus', function (e) {
			e.target.style.minHeight = 'auto';
			e.target.style.minHeight = e.target.scrollHeight + offset + 'px';
		});
	});
	
	
	/* anapnoe ui start	*/
	/* resizable split view */
	
	const resizeEvent = window.document.createEvent('UIEvents'); 
	resizeEvent.initUIEvent('resize', true, false, window, 0); 	

	gradioApp().querySelectorAll('[id $="2img_splitter"]').forEach((elem) => {
		
		elem.addEventListener("mousedown", function(e) {	

			e.preventDefault();
			
			let resizer = e.currentTarget;
			let container = resizer.parentElement;
			
			let flexDir = window.getComputedStyle(container).getPropertyValue('flex-direction');
		
			let leftSide = resizer.previousElementSibling;
			let rightSide = resizer.nextElementSibling;

			let dir = flexDir == "row-reverse" ? -1.0 : 1.0;

			let x = e.clientX;
			let y = e.clientY;
			let leftWidth = leftSide.getBoundingClientRect().width;		
			
			function mouseMoveHandler(e) {		
				resizer.style.cursor = 'col-resize';
				container.style.cursor = 'col-resize';

				const dx = (e.clientX - x)*dir;
				const dy = (e.clientY - y)*dir;

				const newLeftWidth = ((leftWidth + dx) * 100) / container.getBoundingClientRect().width;
				leftSide.style.flexBasis  = `${newLeftWidth}%`;
				leftSide.style.userSelect = 'none';
				leftSide.style.pointerEvents = 'none';
				rightSide.style.userSelect = 'none';
				rightSide.style.pointerEvents = 'none';
				window.dispatchEvent(resizeEvent);		
			}

			function mouseUpHandler() {
				resizer.style.removeProperty('cursor');
				container.style.removeProperty('cursor');
				leftSide.style.removeProperty('user-select');
				leftSide.style.removeProperty('pointer-events');
				rightSide.style.removeProperty('user-select');
				rightSide.style.removeProperty('pointer-events');
				container.removeEventListener('mousemove', mouseMoveHandler);
				container.removeEventListener('mouseup', mouseUpHandler);
				window.dispatchEvent(resizeEvent);				
			}
			
			container.addEventListener('mousemove', mouseMoveHandler);
			container.addEventListener('mouseup', mouseUpHandler);		
	
		})
		
		let flex_reverse = false;
		elem.addEventListener("dblclick", function(e) {	
			flex_reverse = !flex_reverse;	
			e.preventDefault();
			
			let resizer = e.currentTarget;
			let container = resizer.parentElement;			
			//let flexDir = window.getComputedStyle(container).getPropertyValue('flex-direction');

			if(flex_reverse){
				container.style.flexDirection = 'row-reverse';			
			}else{
				container.style.flexDirection = 'row';	
			}		
		})
		
	})
	
	
	// menu 
	function disableScroll() {         
		scrollTop = window.pageYOffset || document.documentElement.scrollTop;
		scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;
		window.scrollTo(scrollLeft, scrollTop);
		
		window.onscroll = function() {
			window.scrollTo(scrollLeft, scrollTop);
		}
	}
          
	function enableScroll() {
		window.onscroll = function() {}
	}	
	

	function toggleMenu(isopen, icon, panel, func) {
		if(isopen){
			panel.classList.add("open");
			icon.classList.add("fixed");
			gradioApp().addEventListener('click', func);			
			disableScroll();
		}else{
			panel.classList.remove("open");
			icon.classList.remove("fixed");
			gradioApp().removeEventListener('click', func);
			enableScroll();	
		}		
	}
	
	function getPos(el) {
		let rect=el.getBoundingClientRect();
		return {x:rect.left,y:rect.top};
	}
	
	// mobile nav menu
	const tabs_menu = gradioApp().querySelector('#tabs > div:first-child');
	const nav_menu = gradioApp().querySelector('#nav_menu');
	const header = gradioApp().querySelector('#header-top');
	let menu_open = false;
	let z_menu = nav_menu.cloneNode(true);
/* 	let pos = getPos(nav_menu);
	clone.style.position = 'fixed';
	clone.style.left = pos.x+'px';
	clone.style.top = pos.y+'px';
	clone.style.zIndex = 99999; */
	z_menu.id = "clone_nav_menu";
	header.parentElement.append(z_menu);
	function toggleNavMenu(e) {
		//e.preventDefault();
        e.stopPropagation();
		menu_open = !menu_open;
		toggleMenu(menu_open, nav_menu, tabs_menu, toggleNavMenu);
	}
    z_menu.addEventListener('click', toggleNavMenu);
	
	// quicksettings nav menu
	let quick_menu_open = false;
	const quicksettings_overflow = gradioApp().querySelector('#quicksettings_overflow');
	const quick_menu = gradioApp().querySelector('#quick_menu');	
	function toggleQuickMenu(e) {			
		quick_menu_open = !quick_menu_open;
		const withinBoundaries = e.composedPath().includes(quicksettings_overflow);
		if(!quick_menu_open && withinBoundaries){
			quick_menu_open = true;
		}else{
			e.preventDefault();
			e.stopPropagation();
			toggleMenu(quick_menu_open, quick_menu, quicksettings_overflow, toggleQuickMenu);
		}
	}	
    quick_menu.addEventListener('click', toggleQuickMenu);
	
	
	// additional ui styles 
	let styleobj = {};
	const r = gradioApp();		
	const style = document.createElement('style');
	style.id="ui-styles";
	r.appendChild(style);
	
	function updateOpStyles() {		
		let ops_styles = "";	
		for (const key in styleobj) {		
			ops_styles += styleobj[key];
		}	
		const ui_styles = gradioApp().getElementById('ui-styles');		
		ui_styles.innerHTML = ops_styles; 
		//console.log(ui_styles);
	}
	
	// livePreview contain - scale
	function imagePreviewFitMethod(value) {
       styleobj.ui_fit = ".livePreview img {object-fit:" + value + ";}"; 
	}	
	gradioApp().querySelector("#setting_live_preview_image_fit").addEventListener('click', function (e) {
		if (e.target && e.target.matches("input[type='radio']")) {
			imagePreviewFitMethod(e.target.value.toLowerCase());	
			updateOpStyles();			
		}
	})
	imagePreviewFitMethod(opts.live_preview_image_fit.toLowerCase());
	
	// viewports order left - right
	function viewportOrder(value) {
       styleobj.ui_views_order = "[id$=_prompt_image] + div {flex-direction:" + value + ";}";	   
	}
	gradioApp().querySelector("#setting_ui_views_order").addEventListener('click', function (e) {
		if (e.target && e.target.matches("input[type='radio']")) {
			viewportOrder(e.target.value.toLowerCase());
			updateOpStyles();			
		}
	})
	viewportOrder(opts.ui_views_order.toLowerCase());
	

	// sd max resolution output
	function sdMaxOutputResolution(value) {
		gradioApp().querySelectorAll('[id$="2img_width"] input,[id$="2img_height"] input').forEach((elem) => {
			elem.max = value;
		})
	}
	gradioApp().querySelector("#setting_sd_max_resolution").addEventListener('input', function (e) {
		let intvalue = parseInt(e.target.value);
		intvalue = Math.min(Math.max(intvalue, 512), 16384);
		sdMaxOutputResolution(intvalue);					
	})	
	sdMaxOutputResolution(opts.sd_max_resolution);
	
	
	function extra_networks_visibility(value){
		gradioApp().querySelectorAll('[id$="2img_extra_networks_row"]').forEach((elem) => {
			if(value){
				elem.classList.remove("!hidden");		
			}else{
				elem.classList.add("!hidden");
			}
		})
	}
	gradioApp().querySelector("#setting_extra_networks_default_visibility input").addEventListener('click', function (e) {		
		extra_networks_visibility(e.target.checked);
	})
	extra_networks_visibility(opts.extra_networks_default_visibility);
	
	function extra_networks_card_size(value) {
       styleobj.extra_networks_card_size = ":host{--extra-networks-card-size:" + value + ";}";  
	}
	gradioApp().querySelectorAll("#setting_extra_networks_cards_size input").forEach(function (elem){
		elem.addEventListener('input', function (e) {		
			extra_networks_card_size(e.target.value);
			updateOpStyles();
		})	
	})
	extra_networks_card_size(opts.extra_networks_cards_size);
	
	function extra_networks_cards_visible_rows(value) {		
       styleobj.extra_networks_cards_visible_rows = ":host{--extra-networks-visible-rows:" + value + ";}";
	}
	gradioApp().querySelectorAll("#setting_extra_networks_cards_visible_rows input").forEach(function (elem){
		elem.addEventListener('input', function (e) {		
			extra_networks_cards_visible_rows(e.target.value);
			updateOpStyles();
		})		
	})
	extra_networks_cards_visible_rows(opts.extra_networks_cards_visible_rows);

	//hidden ui tabs
	let radio_hidden_html="";
	let radio_header_html="";
	let hiddentabs={};
	let headertabs={};
	const setting_ui_hidden_tabs = gradioApp().querySelector('#setting_ui_hidden_tabs textarea');
	const setting_ui_header_tabs = gradioApp().querySelector('#setting_ui_header_tabs textarea');
	const parent_header_tabs = gradioApp().querySelector('#nav_menu_header_tabs');
	setting_ui_hidden_tabs.style.display = "none";
	setting_ui_header_tabs.style.display = "none";
	
	const maintabs = gradioApp().querySelectorAll('#tabs > div:first-child > button');
	const tabitem = gradioApp().querySelectorAll('#tabs > div.tabitem');
	
	
	function tabOpsSave(setting){
		const iEvent = new Event("input");		
		Object.defineProperty(iEvent, "target", {value: setting})		
		setting.dispatchEvent(iEvent);
	}
	
	function tabsHiddenChange() {		
		const keys = Object.keys(hiddentabs);
		setting_ui_hidden_tabs.value = "";
		keys.forEach((key, index) => {
			//console.log(`${key}: ${hiddentabs[key]} ${index}`);
			if(hiddentabs[key] == true){
				styleobj[key] = "#tabs > div:first-child > button:nth-child("+(index+1)+"){display:none;}";
				setting_ui_hidden_tabs.value += key + ",";
			}else{								
				styleobj[key] = "#tabs > div:first-child  > button:nth-child("+(index+1)+"){display:block;}";
			}
		})
		
		
		tabOpsSave(setting_ui_hidden_tabs);
		tabsHeaderChange();
		
	}
	
	function tabsHeaderChange() {		
		const keys = Object.keys(headertabs);
		setting_ui_header_tabs.value = "";
		keys.forEach((key, index) => {
			//console.log(`${key}: ${hiddentabs[key]} ${index}`);
			let nkey = key+"_hr";
			if(headertabs[key] == true && hiddentabs[key] == false){				
				styleobj[nkey] = "#nav_menu_header_tabs > button:nth-child("+(index+1)+"){display:block;}";				
				setting_ui_header_tabs.value += key + ",";
			}else{				
				styleobj[nkey] = "#nav_menu_header_tabs > button:nth-child("+(index+1)+"){display:none;}";
			}
		})
		
		tabOpsSave(setting_ui_header_tabs);
	}
	

	const Observe = (sel, opt, cb) => {
	  const Obs = new MutationObserver((m) => [...m].forEach(cb));
	  gradioApp().querySelectorAll(sel).forEach(el => Obs.observe(el, opt));
	}
	
	Observe("#tabs > div.tabitem", {
	  attributesList: ["style"], attributeOldValue: true, }, (m) => {		
		if(m.target.style.display === 'block'){
			let idx = parseInt(m.target.getAttribute("tab-item"));
			tabSelected(idx);			
		}		
	})
	
	function tabSelected(idx){

		gradioApp().querySelectorAll('#tabs > div > button.bg-white, #nav_menu_header_tabs > button.bg-white').forEach(function (tab){
			tab.classList.remove("bg-white");
		})
		
		gradioApp().querySelectorAll('#tabs > div > button:nth-child('+(idx+1)+'), #nav_menu_header_tabs > button:nth-child('+(idx+1)+')').forEach(function (tab){
			tab.classList.add("bg-white");			
		}) 
	}
	
	function navigate2Tab(idx){
		gradioApp().querySelectorAll('#tabs > div.tabitem').forEach(function (tabitem, index){			
			if(idx == index){
				tabitem.style.display = "block";
			}else{
				tabitem.style.display = "none";
			}			
		})
	}

	maintabs.forEach(function (elem, index) {
		let tabvalue = elem.innerText.replaceAll(" ", "");
		hiddentabs[tabvalue] = false;
		headertabs[tabvalue] = false;
		let checked_hidden = "";
		let checked_header = "";
		if(setting_ui_hidden_tabs.value.indexOf(tabvalue) != -1){
			hiddentabs[tabvalue] = true;
			checked_hidden = "checked";
		}
		if(setting_ui_header_tabs.value.indexOf(tabvalue) != -1){
			headertabs[tabvalue] = true;
			checked_header = "checked";		
		}			
		radio_hidden_html += '<label class="gr-input-label flex items-center text-gray-700 text-sm space-x-2 border py-1.5 px-3 rounded-lg cursor-pointer bg-white shadow-sm checked:shadow-inner"><input type="checkbox" name="uiha" class="gr-check-radio gr-radio" '+checked_hidden+' value="'+elem.innerText+'"><span class="ml-2" title="">'+elem.innerText+'</span></label>';
		radio_header_html += '<label class="gr-input-label flex items-center text-gray-700 text-sm space-x-2 border py-1.5 px-3 rounded-lg cursor-pointer bg-white shadow-sm checked:shadow-inner"><input type="checkbox" name="uihb" class="gr-check-radio gr-radio" '+checked_header+' value="'+elem.innerText+'"><span class="ml-2" title="">'+elem.innerText+'</span></label>';
		
		tabitem[index].setAttribute("tab-item", index);
		
		let clonetab = elem.cloneNode(true);
		clonetab.id = tabvalue+"_clone";
		parent_header_tabs.append(clonetab);
		clonetab.setAttribute("tab-id", index);
		
		clonetab.addEventListener('click', function (e) {
			const idx = parseInt(e.currentTarget.getAttribute("tab-id"));
			navigate2Tab(idx);			
		})
		
	})
	

	let div = document.createElement("div");
	div.id = "hidden_radio_tabs_container";
	div.classList.add("flex", "flex-wrap", "gap-2");
	div.innerHTML = radio_hidden_html;	
	setting_ui_hidden_tabs.parentElement.appendChild(div);
	
	div = document.createElement("div");
	div.id = "header_radio_tabs_container";
	div.classList.add("flex", "flex-wrap", "gap-2");
	div.innerHTML = radio_header_html;
	setting_ui_header_tabs.parentElement.appendChild(div);
	
	
	// hidden tabs
	gradioApp().querySelector("#hidden_radio_tabs_container").addEventListener('click', function (e) {
		if (e.target && e.target.matches("input[type='checkbox']")) {
			let tabvalue = e.target.value.replaceAll(" ", "");	
			hiddentabs[tabvalue] = e.target.checked				
			tabsHiddenChange();			
			updateOpStyles();
		}
	})
	// header tabs
	gradioApp().querySelector("#header_radio_tabs_container").addEventListener('click', function (e) {
		if (e.target && e.target.matches("input[type='checkbox']")) {
			let tabvalue = e.target.value.replaceAll(" ", "");	
			headertabs[tabvalue] = e.target.checked;			
			tabsHeaderChange();
			updateOpStyles();
		}
	})
	
	tabsHiddenChange();
	
	// add to quick settings
	const settings_submit = gradioApp().querySelector('#settings_submit');
	const quick_parent = gradioApp().querySelector("#quicksettings_overflow_container");
	
	function add2quickSettings(id, section, checked){
		const setting_quicksettings = gradioApp().querySelector('#setting_quicksettings textarea');
		let field_settings = setting_quicksettings.value.replace(" ", "");		
		if(checked){
			field_settings += ","+id;			
			let setting_row = gradioApp().querySelector('#row_setting_'+id);			
			quick_parent.append(setting_row);
		}else{			
			field_settings = field_settings.replaceAll(id, ",");			
			const setting_parent = gradioApp().querySelector("#"+section+"_settings_2img_settings");
			let quick_row = gradioApp().querySelector('#row_setting_'+id);
			//console.log(quick_row);					
			setting_parent.append(quick_row);

		}
		field_settings = field_settings.replace(/,{2,}/g, ',');
		setting_quicksettings.value = field_settings;

		const iEvent = new Event("input");		
		Object.defineProperty(iEvent, "target", {value: setting_quicksettings})		
		setting_quicksettings.dispatchEvent(iEvent);
		
		const cEvent = new Event("click");//submit	
		Object.defineProperty(cEvent, "target", {value: settings_submit})		
		settings_submit.dispatchEvent(cEvent);
		//settings_submit.click();
		//console.log(section + " - "+ id + " - " + checked);
	}
	gradioApp().querySelectorAll('[id*="add2quick_"]').forEach(function (elem){
		let trg = elem.id.split('_add2quick_setting_');
		let sid = trg[0];
		let tid = trg[1];
		let elem_input = gradioApp().querySelector('#'+elem.id+' input');
		elem_input.addEventListener('click', function (e) {		
			add2quickSettings(tid, sid, e.target.checked);
		})
	})
	
	
	
    const container = gradioApp().querySelector("#quicksettings_overflow_container");  
	let draggables;
	// function that gets the element next of cursor/touch
    function getElementAfter(container, y){
        return draggables.reduce((closest, child) => {
            const box = child.getBoundingClientRect();
            const offset = y - box.top - box.height / 2;           
            if(offset < 0 && offset > closest.offset){
                return { offset: offset, element: child};
            } else {
                return closest;
            }       
        }, { offset: Number.NEGATIVE_INFINITY } ).element;
    }
	
	
	let lastElemAfter;
	let islastChild;
	let timeout;
	
	function preventBehavior(e){					
		e.stopPropagation();
		e.preventDefault();
		return false;
	}
		
	
	function dragOrderChange(elementAfter, isComplete, delem) {
				
		if(lastElemAfter !== elementAfter || islastChild){
			if(lastElemAfter != null ){
				lastElemAfter.classList.remove('marker-top', 'marker-bottom');
			}

			if(elementAfter == null){
				islastChild = true;
				lastElemAfter.classList.add('marker-bottom');
			} else {
				islastChild = false;
				elementAfter.classList.add('marker-top');
				lastElemAfter = elementAfter;
				
			} 
		}
		
		if(isComplete){
			
			if(elementAfter == null){
				container.append(delem);
			} else {
				container.insertBefore(delem, elementAfter);
			}	
			
			let order_settings="";
			
			gradioApp().querySelectorAll('#quicksettings_overflow_container > div').forEach(function (el){
				el.classList.remove('marker-top', 'marker-bottom', 'dragging');
				order_settings += el.id.split("row_setting_")[1]+",";							
			})
			//console.log(order_settings);
			const setting_quicksettings = gradioApp().querySelector('#setting_quicksettings textarea');
			setting_quicksettings.value = order_settings;
			const iEvent = new Event("input");		
			Object.defineProperty(iEvent, "target", {value: setting_quicksettings})		
			setting_quicksettings.dispatchEvent(iEvent);
			
			const cEvent = new Event("click");//submit	
			Object.defineProperty(cEvent, "target", {value: settings_submit})		
			settings_submit.dispatchEvent(cEvent);
			
			container.classList.remove('no-scroll');
		}
	}

	function touchmove(e){					
		let y = e.touches[0].clientY;
		let elementAfter = getElementAfter(container, y);					
		dragOrderChange(elementAfter);					
	}

	function touchstart(e){					
		e.currentTarget.draggable = "false";
		let target = e.currentTarget;
		// touch should be hold for 1 second
		timeout = setTimeout(function(){
			target.classList.add('dragging');						
			container.classList.add('no-scroll');
			target.addEventListener('touchmove', touchmove);
			container.addEventListener("touchmove", preventBehavior, {passive: false});			 											
		}, 1000);
		
		target.addEventListener('touchend', touchend);
		target.addEventListener('touchcancel', touchcancel);											  
	}
	
	function touchend(e){
		e.currentTarget.draggable = "true";
		clearTimeout(timeout);					
		e.currentTarget.removeEventListener('touchmove', touchmove);
		container.removeEventListener("touchmove", preventBehavior);
		let y = e.changedTouches[0].clientY;
		let elementAfter = getElementAfter(container, y);					
		dragOrderChange(elementAfter, true, e.currentTarget);				
	}
	
	function touchcancel(e){
		e.currentTarget.draggable = "true";
		clearTimeout(timeout);
		e.currentTarget.classList.remove('dragging');					
		e.currentTarget.removeEventListener('touchmove', touchmove);
		container.removeEventListener("touchmove", preventBehavior);
	}
	
	function dragstart(e){
		e.currentTarget.draggable = "false";
		e.currentTarget.classList.add("dragging");
	}
	
	function dragend(e){					
		e.stopPropagation();
		e.preventDefault();
		e.currentTarget.draggable = "true";
		let y = e.clientY;
		let elementAfter = getElementAfter(container, y);
		dragOrderChange(elementAfter, true, e.currentTarget);
	} 

	function dragOver(e) {					
		e.preventDefault();
		let y = e.clientY;
		let elementAfter = getElementAfter(container, y);
		dragOrderChange(elementAfter);
	}


	function actionQuickSettingsDraggable(checked){
		

		if(checked){
			draggables = Array.from(gradioApp().querySelectorAll('#quicksettings_overflow_container > div:not(.dragging)'));			
			gradioApp().addEventListener('drop', preventBehavior);
		}else{			
			gradioApp().removeEventListener('drop', preventBehavior);
		}
		

		gradioApp().querySelectorAll('#quicksettings_overflow_container > div').forEach(function (elem){
			
			//elem.setAttribute("draggable", checked);
			elem.draggable = checked;
			
			if(checked){
							
				elem.addEventListener('touchstart', touchstart);
				elem.addEventListener('dragstart', dragstart);
				elem.addEventListener('dragend', dragend);
				elem.addEventListener('dragover', dragOver);
				
			}else{
				
				elem.removeEventListener('touchstart', touchstart, false);
				elem.removeEventListener('touchend', touchend, false);
				elem.removeEventListener('touchcancel', touchcancel, false);
				elem.removeEventListener('touchmove', touchmove, false);
				elem.removeEventListener('dragstart', dragstart, false);
				elem.removeEventListener('dragend', dragend, false);
				elem.removeEventListener('dragover', dragOver, false);

			}
		})
	}
	

	gradioApp().querySelector('#quicksettings_draggable').addEventListener('click', function (e) {
		if (e.target && e.target.matches("input[type='checkbox']")) {
			actionQuickSettingsDraggable(e.target.checked);
		}
	})
	


	updateOpStyles();
	

	/* anapnoe ui end */	

	
})

onOptionsChanged(function(){
    elem = gradioApp().getElementById('sd_checkpoint_hash')
    sd_checkpoint_hash = opts.sd_checkpoint_hash || ""
    shorthash = sd_checkpoint_hash.substr(0,10)

	if(elem && elem.textContent != shorthash){
	    elem.textContent = shorthash
	    elem.title = sd_checkpoint_hash
	    elem.href = "https://google.com/search?q=" + sd_checkpoint_hash
	}
})

let txt2img_textarea, img2img_textarea = undefined;
let wait_time = 800
let token_timeouts = {};

function update_txt2img_tokens(...args) {
	update_token_counter("txt2img_token_button")
	if (args.length == 2)
		return args[0]
	return args;
}

function update_img2img_tokens(...args) {
	update_token_counter("img2img_token_button")
	if (args.length == 2)
		return args[0]
	return args;
}

function update_token_counter(button_id) {
	if (token_timeouts[button_id])
		clearTimeout(token_timeouts[button_id]);
	token_timeouts[button_id] = setTimeout(() => gradioApp().getElementById(button_id)?.click(), wait_time);
}

function restart_reload(){
	

	
	//document.html.innerHTML +='<style id="cool">body{background-color:#550000!important;}</style> ';
	document.body.style.backgroundColor = "#1a1a1a";
    document.body.innerHTML='<h1 style="font-family:monospace;margin-top:20%;color:lightgray;text-align:center;">Reloading...</h1>';	
    setTimeout(function(){location.reload()},2000)

    return []
}

// Simulate an `input` DOM event for Gradio Textbox component. Needed after you edit its contents in javascript, otherwise your edits
// will only visible on web page and not sent to python.
function updateInput(target){
	let e = new Event("input", { bubbles: true })
	Object.defineProperty(e, "target", {value: target})
	target.dispatchEvent(e);
}


var desiredCheckpointName = null;
function selectCheckpoint(name){
    desiredCheckpointName = name;
    gradioApp().getElementById('change_checkpoint').click()
}


