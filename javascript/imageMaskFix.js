/**
 * temporary fix for https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/668
 * @see https://github.com/gradio-app/gradio/issues/1721
 */
 /*
window.addEventListener( 'resize', () => imageMaskResize());
function imageMaskResize() {
    const canvases = gradioApp().querySelectorAll('#img2maskimg .touch-none canvas');
    if ( ! canvases.length ) {
    canvases_fixed = false;
    window.removeEventListener( 'resize', imageMaskResize );
    return;
    }

    const wrapper = canvases[0].closest('.touch-none');
    const previewImage = wrapper.previousElementSibling;

    if ( ! previewImage.complete ) {
        previewImage.addEventListener( 'load', () => imageMaskResize());
        return;
    }

    const w = previewImage.width;
    const h = previewImage.height;
    const nw = previewImage.naturalWidth;
    const nh = previewImage.naturalHeight;
    const portrait = nh > nw;
    const factor = portrait;

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
 


let img2img_tab_index = 0;
let img2img_tab_tool_buttons;

let isPanning;
let parent_container_img;
let spotlight_sketch;
let spotlight_inpaint;
let spotlight_inpaint_sketch;
let spotlight;

let curr_src = [];

let intervalLastUIUpdate;

let sketch_pan;
let inpaint_pan;
let inpaint_sketch_pan;
let curr_pan;

function common_undo_handler(e) {
	img2img_tab_tool_buttons[0].click();
}
function common_clear_handler(e){
	img2img_tab_tool_buttons[1].click();
}

function common_color_handler(e){
	let cb = parent_container_img.querySelector("input[type='color']");
	if(!cb){
		img2img_tab_tool_buttons[3].click();		
		e.target.addEventListener("input", function(ev) {	
			cb = parent_container_img.querySelector("input[type='color']");			
			cb.value = ev.target.value;
			updateInput(cb);
			pan_toggle(false, curr_pan);			
		})
	}
}

function common_brush_handler(e){
	let cb = parent_container_img.querySelector("input[type='range']");
	if(!cb){
		img2img_tab_tool_buttons[2].click();		
		e.target.addEventListener("input", function(ev) {	
			cb = parent_container_img.querySelector("input[type='range']");			
			cb.value = ev.target.value;
			updateInput(cb);
			//pan_toggle(false, curr_pan);			
		})
	}
}

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
	spotlight.panzoom(val);
	
	if(isPanning){
		parent_container_img.classList.add("no-point-events");
		parent_container_img.parentElement.classList.add("move");
		document.addEventListener('wheel', preventDefault, {passive: false});
	}else{
		parent_container_img.classList.remove("no-point-events");
		parent_container_img.parentElement.classList.remove("move");
		document.removeEventListener('wheel', preventDefault, false);
	}
	
}

function common_pan_handler(e){
	isPanning = !isPanning;
	pan_toggle(isPanning, this);
}


onUiLoaded(function(){
	
	const color_box = '<input type="color">';
	const brush_size = '<input type="range" min="0.75" max="110.0">';

	spotlight_sketch = new Spotlight();
	const spotlight_sketch_parent = gradioApp().querySelector("#img2img_sketch");
	spotlight_sketch.init(spotlight_sketch_parent, "-sketch");			
	spotlight_sketch.addControl("undo", common_undo_handler);
	sketch_pan = spotlight_sketch.addControl("pan", common_pan_handler);	
	spotlight_sketch.addControl("brush", common_brush_handler, brush_size);
	spotlight_sketch.addControl("color", common_color_handler, color_box);
	spotlight_sketch.addControl("clear",common_clear_handler);
	
	spotlight_inpaint = new Spotlight();
	const spotlight_inpaint_parent = gradioApp().querySelector("#img2maskimg");
	spotlight_inpaint.init(spotlight_inpaint_parent, "-inpaint");			
	spotlight_inpaint.addControl("undo", common_undo_handler);
	inpaint_pan = spotlight_inpaint.addControl("pan", common_pan_handler);
	spotlight_inpaint.addControl("brush", common_brush_handler, brush_size);
	spotlight_inpaint.addControl("clear",common_clear_handler);
	
	spotlight_inpaint_sketch = new Spotlight();
	const spotlight_inpaint_sketch_parent = gradioApp().querySelector("#inpaint_sketch");
	spotlight_inpaint_sketch.init(spotlight_inpaint_sketch_parent, "-inpaint-sketch");			
	spotlight_inpaint_sketch.addControl("undo", common_undo_handler);
	inpaint_sketch_pan = spotlight_inpaint_sketch.addControl("pan", common_pan_handler);
	spotlight_inpaint_sketch.addControl("brush", common_brush_handler, brush_size);
	spotlight_inpaint_sketch.addControl("color", common_color_handler, color_box);
	spotlight_inpaint_sketch.addControl("clear",common_clear_handler);
})


onUiUpdate(function() {
	//clearInterval(intervalLastUIUpdate);
	const img2img_tab = gradioApp().querySelector('#img2img_img2img_tab');
	if(img2img_tab && selectedTabItemId == "tab_img2img"){
		//console.log("UIMASKUPDATE");
		const img2img_tab_index = get_img2img_tab_index()[0];
		if(img2img_tab_index > 3 || img2img_tab_index == 0) return;
		
		let tabid;
		let copytoId;
		
		if(img2img_tab_index == 1){
			tabid = "#img2img_img2img_sketch_tab";// #img2img_sketch";
			spotlight = spotlight_sketch;			
			curr_pan = sketch_pan;
		}else if(img2img_tab_index == 2){
			tabid = "#img2img_inpaint_tab";// #img2maskimg";
			spotlight = spotlight_inpaint;
			curr_pan = inpaint_pan;
		}else if(img2img_tab_index == 3){
			tabid = "#img2img_inpaint_sketch_tab";// #inpaint_sketch";
			spotlight = spotlight_inpaint_sketch;		
			curr_pan = inpaint_sketch_pan;
		}

		
		const parent_img2img_tab_img = gradioApp().querySelector(tabid);			
		//const parent_img2img_tab_copy_to = parent_img2img_tab_img.querySelector(copytoId);
		const spotlight_parent = parent_img2img_tab_img.querySelector('div[data-testid="image"]');			
		parent_container_img = parent_img2img_tab_img.querySelector('div[data-testid="image"] > div');
		const img2img_tab_img = parent_container_img.querySelector('img');
		img2img_tab_tool_buttons = parent_container_img.querySelectorAll('button');
		
			
		if(img2img_tab_img){		
			pan_toggle(false, curr_pan);					
			spotlight.panzoom(false);
		}
			
		function getImgSync(image){
			let c_src;
			
			if(curr_src[img2img_tab_index] != image.src){ 
				//console.log("NEWIMAGE");
				curr_src[img2img_tab_index] = image.src;		

				let w = image.naturalWidth; 
				let h = image.naturalHeight; 
				parent_container_img.style.width = `${w}px`;
				parent_container_img.style.height = `${h}px`;
				
				spotlight.show([{						
					media: "node",																		
					src: spotlight_parent,
					autohide: true,
					control: ["pan","clear","undo","fullscreen","autofit","zoom-in","zoom-out","close"],
					class: "relative",
				}],					
				);

				parent_container_img.style.flexGrow = "0";
			}
		}
		
		const getImage = async (image) => {
			if(!image){ 						
				if(spotlight.panel){
					spotlight.close(false, true);
					parent_container_img.style.flexGrow = "1";
					curr_src[img2img_tab_index] = "";
					spotlight_parent.style = "";
					
				}
			}else if (image.complete) {							
				return getImgSync(image);				
			}else{
				return new Promise(resolve => {
					image.onload = () => {
						resolve(getImgSync(image));
					};
				});
			}
		};
		
		getImage(img2img_tab_img);

	}
	
})

/* 
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
 
