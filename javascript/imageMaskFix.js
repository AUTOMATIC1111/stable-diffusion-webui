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
onUiUpdate(function() {
	const img2img_tab = gradioApp().querySelector('#img2img_img2img_tab');
	if(img2img_tab){
		const current_img2img_tab_index = get_img2img_tab_index()[0];
		//if(img2img_tab_index != current_img2img_tab_index){
			//console.log(current_img2img_tab_index);
			if(current_img2img_tab_index > 3) return;
			img2img_tab_index = current_img2img_tab_index;
			const parent_img2img_tab_img = gradioApp().querySelector('#mode_img2img > div:nth-child('+(img2img_tab_index+2)+') div[data-testid="image"] > div');		
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
})
