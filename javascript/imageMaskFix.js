/**
 * temporary fix for https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/668
 * @see https://github.com/gradio-app/gradio/issues/1721
 */

 /*
window.addEventListener( 'resize', () => imageMaskResize());

function imageMaskResize() {
    const canvases = gradioApp().querySelectorAll('#img2maskimg .touch-none canvas');
    if ( ! canvases.length ) {
    canvases_fixed = false; // TODO: this is unused..?
    window.removeEventListener( 'resize', imageMaskResize );
    return;
    }

    const wrapper = canvases[0].closest('.touch-none');
    const previewImage = wrapper.previousElementSibling;

    if ( ! previewImage.complete ) {
        previewImage.addEventListener( 'load', imageMaskResize);
        return;
    }

    const w = previewImage.width;
    const h = previewImage.height;
    const nw = previewImage.naturalWidth;
    const nh = previewImage.naturalHeight;
    const portrait = nh > nw;

    const wW = Math.min(w, portrait ? h/nh*nw : w/nw*nw);
    const wH = Math.min(h, portrait ? h/nh*nh : w/nw*nh);

    wrapper.style.width = `${wW}px`;
    wrapper.style.height = `${wH}px`;
    wrapper.style.left = `0px`;
    wrapper.style.top = `0px`;

    canvases.forEach( c => {
        c.style.width = c.style.height = '';
        c.style.maxWidth = '100%';
        c.style.maxHeight = '100%';
        c.style.objectFit = 'contain';
    });
 }

 onUiUpdate(() => imageMaskResize());
*/
 
onUiLoaded(function(){
	
	const color_box = '<input type="color">';
	const brush_size = '<input type="range" min="0.75" max="110.0">';
	
	let img_src = [];
	let spl_instances = [];
	let spl_pan_instances = [];
	const container = gradioApp().querySelector(".gradio-container");
    const observer  = new MutationObserver(() => 
		gradioApp().querySelectorAll('div[data-testid="image"]').forEach(function (elem, i){			
			let img_parent = elem.parentElement.querySelector('div[data-testid="image"] > div');
			let img = img_parent.querySelector('img');
			if(img){
				if(img_src[i] != img.src){
					let tool_buttons = img_parent.querySelectorAll('button');
					if(tool_buttons.length > 2){
						let spl_parent = elem.parentElement;
						let spl;
						let spl_pan;
						let isPanning;
						
						if(spl_parent.className != "spl-pane"){							
							spl = new Spotlight();	
							spl.init(spl_parent, "-"+spl_parent.id);
							
							spl.addControl("undo", spl_undo_handler);
							spl_pan = spl.addControl("pan", spl_pan_handler);	
							spl.addControl("brush", spl_brush_handler, brush_size);
							if(tool_buttons.length == 4){
								spl.addControl("color", spl_color_handler, color_box);
							}
							spl.addControl("clear",spl_clear_handler);
							
							spl_instances[i] = spl;
							spl_pan_instances[i] = spl_pan;
							
						}else{
							spl = spl_instances[i];							
							spl_pan = spl_pan_instances[i];							
						}
						
						img_src[i] = img.src;						
						//console.log("NEWIMAGE");

						function spl_undo_handler(e) {
							tool_buttons[0].click();
						}
						
						function spl_clear_handler(e){							
							tool_buttons[1].click();
							spl.panzoom(false);
							img_parent.classList.remove("no-point-events");
							img_parent.parentElement.classList.remove("move");
							document.removeEventListener('wheel', preventDefault, false);							
							setTimeout(function() { 
								spl.close(false, true);
								img_parent.style.flexGrow = "1";
								img_src[i] = "";
								elem.style.transform = "none";
							}, 200);					
							
						}
						
						function spl_color_handler(e){}
						function spl_brush_handler(e){}

						function preventDefault(e) {
							e = e || window.event
							if (e.preventDefault) {
							  e.preventDefault()
							}
							e.returnValue = false
						}

						function pan_toggle(val, target){
							isPanning = val;
							target.classList.toggle("on", isPanning);
							spl.panzoom(val);
							
							if(isPanning){
								img_parent.classList.add("no-point-events");
								img_parent.parentElement.classList.add("move");
								document.addEventListener('wheel', preventDefault, {passive: false});
							}else{
								img_parent.classList.remove("no-point-events");
								img_parent.parentElement.classList.remove("move");
								document.removeEventListener('wheel', preventDefault, false);
							}
							
						}

						function spl_pan_handler(e){
							isPanning = !isPanning;
							pan_toggle(isPanning, this);
						}

						function update_color(listener){
							let input_color = img_parent.querySelector("input[type='color']");
							let spl = img_parent.parentElement.parentElement.parentElement.parentElement.parentElement;
							let spl_color = spl.querySelector(".spl-color input[type='color']");
							spl_color.value = input_color.value;
							
							if(listener){
								spl_color.addEventListener("input", function(ev) {						
									input_color.value = ev.target.value;
									updateInput(input_color);
									pan_toggle(false, spl_pan);					
								})
							}
						}

						function update_brush(listener){
							let input_range = img_parent.querySelector("input[type='range']");
							let spl = img_parent.parentElement.parentElement.parentElement.parentElement.parentElement;
							let spl_brush = spl.querySelector(".spl-brush input[type='range']");
							spl_brush.value = input_range.value;
							
							if(listener){
								spl_brush.addEventListener("input", function(ev) {						
									input_range.value = ev.target.value;
									updateInput(input_range);				
								})
							}
						}

						function init_drawing_tools(){							
							let input_color = img_parent.querySelector("input[type='color']");			
							if(!input_color){
								if(tool_buttons[3]){
									tool_buttons[3].click();	
									setTimeout(function() { update_color(true);	}, 100);
								}
							}else{
								setTimeout(function() { update_color(false);}, 100);		
							}	
							
							let input_range = img_parent.querySelector("input[type='range']");
							if(!input_range){
								if(tool_buttons[2]){
									tool_buttons[2].click();	
									setTimeout(function() { update_brush(true);	}, 100);
								}
							}else{
								setTimeout(function() { update_brush(false);}, 100);		
							}
							
						}

						let w = img.naturalWidth; 
						let h = img.naturalHeight; 
						img_parent.style.width = `${w}px`;
						img_parent.style.height = `${h}px`;
						
						spl.show([{						
							media: "node",																		
							src: elem,
							//autohide: true,
							//control: ["pan","clear","undo","fullscreen","autofit","zoom-in","zoom-out","close"],
							class: "relative",
						}],					
						);

						img_parent.style.flexGrow = "0";						
						pan_toggle(false, spl_pan);					
						spl.panzoom(false);
						setTimeout(function() { init_drawing_tools(); }, 500);
						
					}
				}
			}
		}));
    observer.observe(container, { childList: true, subtree: true });
	
})


/* 
let intervalLastUIUpdate;
function onLastUIUpdate(){
	clearInterval(intervalLastUIUpdate);
	const img2img_tab = gradioApp().querySelector('#img2img_img2img_tab');
	if(img2img_tab && selectedTabItemId=="tab_img2img"){
		const current_img2img_tab_index = get_img2img_tab_index()[0];
		//if(img2img_tab_index != current_img2img_tab_index){
			console.log(current_img2img_tab_index);
			if(current_img2img_tab_index > 3 || current_img2img_tab_index == 0) return;
			img2img_tab_index = current_img2img_tab_index;
			let tabid;
			
			if(img2img_tab_index == 1){
				tabid = "#img2img_img2img_sketch_tab";// #img2img_sketch";			
			}else if(img2img_tab_index == 2){
				tabid = "#img2img_inpaint_tab";// #img2maskimg";
			}else if(img2img_tab_index == 3){
				tabid = "#img2img_inpaint_sketch_tab";// #inpaint_sketch";
			}
			//const parent_img2img_tab_img = gradioApp().querySelector('#mode_img2img > div:nth-child('+(img2img_tab_index+2)+') div[data-testid="image"] > div');
			const parent_img2img_tab_img = gradioApp().querySelector('#mode_img2img '+tabid+' div[data-testid="image"] > div');		
			const img2img_tab_img = parent_img2img_tab_img.querySelector('img');			
			if(img2img_tab_img){			
				parent_img2img_tab_img.style.flexGrow = "0";				
				img2img_tab_img.onload = function() {
					let w = this.naturalWidth; 
					let h = this.naturalHeight; 
					parent_img2img_tab_img.style.width = `${w}px`;
					parent_img2img_tab_img.style.height = `${h}px`;
				}
			}else{
				parent_img2img_tab_img.style.flexGrow = "1";
			}
		//}
	}
}
*/
/* onUiUpdate(function() {	
	if(intervalLastUIUpdate != null) clearInterval(intervalLastUIUpdate);
	intervalLastUIUpdate = setInterval(onLastUIUpdate, 1000);	
}) */
 


