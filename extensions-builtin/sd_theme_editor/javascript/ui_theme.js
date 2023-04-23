function hexToRgb (color) {
  let hex = color[0] === '#' ? color.slice(1) : color;
  let c;

  // expand the short hex by doubling each character, fc0 -> ffcc00
  if (hex.length !== 6) {
    hex = ((() => {
      const result = [];
      for (c of Array.from(hex)) {
        result.push(`${c}${c}`);
      }
      return result;
    })()).join('');
  }
  const colorStr = hex.match(/#?(.{2})(.{2})(.{2})/).slice(1);
  const rgb = colorStr.map(col => parseInt(col, 16));
  rgb.push(1);
  return rgb;
}

function rgbToHsl (rgb) {
  const r = rgb[0] / 255;
  const g = rgb[1] / 255;
  const b = rgb[2] / 255;

  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const diff = max - min;
  const add = max + min;

  const hue =
    min === max ?
      0
    : r === max ?
      (((60 * (g - b)) / diff) + 360) % 360
    : g === max ?
      ((60 * (b - r)) / diff) + 120
    :
      ((60 * (r - g)) / diff) + 240;

  const lum = 0.5 * add;

  const sat =
    lum === 0 ?
      0
    : lum === 1 ?
      1
    : lum <= 0.5 ?
      diff / add
    :
      diff / (2 - add);

  const h = Math.round(hue);
  const s = Math.round(sat * 100);
  const l = Math.round(lum * 100);
  const a = rgb[3] || 1;

  return [h, s, l, a];
}

function hexToHsl (color) {
  const rgb = hexToRgb(color);
  const hsl = rgbToHsl(rgb);
  return "hsl("+hsl[0]+"deg "+ hsl[1] +"% " + hsl[2] + "%)";
}

function hslToHex(h, s, l) {
	
  l /= 100;
  const a = s * Math.min(l, 1 - l) / 100;
  const f = n => {
    const k = (n + h / 30) % 12;
    const color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
    return Math.round(255 * color).toString(16).padStart(2, '0');   // convert to Hex and prefix "0" if needed
  };
  return `#${f(0)}${f(8)}${f(4)}`;
}

function hsl2rgb(h,s,l) {
   let a=s*Math.min(l,1-l);
   let f= (n,k=(n+h/30)%12) => l - a*Math.max(Math.min(k-3,9-k,1),-1);
   return [f(0),f(8),f(4)];
}   

function invertColor(hex) {
    if (hex.indexOf('#') === 0) {
        hex = hex.slice(1);
    }
    // convert 3-digit hex to 6-digits.
    if (hex.length === 3) {
        hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
    }
    if (hex.length !== 6) {
        throw new Error('Invalid HEX color.');
    }
    // invert color components
    var r = (255 - parseInt(hex.slice(0, 2), 16)).toString(16),
        g = (255 - parseInt(hex.slice(2, 4), 16)).toString(16),
        b = (255 - parseInt(hex.slice(4, 6), 16)).toString(16);
    // pad each with zeros and return
    return '#' + padZero(r) + padZero(g) + padZero(b);
}

function padZero(str, len) {
    len = len || 2;
    var zeros = new Array(len).join('0');
    return (zeros + str).slice(-len);
}


function getValsWrappedIn(str,c1,c2){
    var rg = new RegExp("(?<=\\"+c1+")(.*?)(?=\\"+c2+")","g"); 
    return str.match(rg);
}

let styleobj = {};	
let hslobj = {};
let isColorsInv;
	
const toHSLArray = hslStr => hslStr.match(/\d+/g).map(Number);

function offsetColorsHSV(ohsl){
	
	let inner_styles = "";	
	
	for (const key in styleobj) {
		
		let keyVal = styleobj[key];
		
		if( keyVal.indexOf("#") != -1 || keyVal.indexOf("hsl") != -1){
			let colcomp = gradioApp().querySelector('#'+key+' input');
			if(colcomp){
				
				let hsl;
				
				if( keyVal.indexOf("#") != -1){
					keyVal = keyVal.replace(/\s+/g, "");
					//inv ? keyVal = invertColor(keyVal) : 0;
					if(isColorsInv){
						keyVal = invertColor(keyVal);
						styleobj[key] = keyVal;						
					}					
					hsl = rgbToHsl(hexToRgb(keyVal));
				}else{
					if(isColorsInv){
						let c = toHSLArray(keyVal);
						let hex = hslToHex(c[0],c[1],c[2]);							
						keyVal = invertColor(hex);
						styleobj[key] = keyVal;	
						hsl = rgbToHsl(hexToRgb(keyVal));
						
					}else{
						hsl = toHSLArray(keyVal);
					}				
				}
				

				let h = (parseInt(hsl[0]) + parseInt(ohsl[0])) % 360;
				let s = (parseInt(hsl[1]) + parseInt(ohsl[1]));
				let l = (parseInt(hsl[2]) + parseInt(ohsl[2]));
			
				let hex = hslToHex(h, Math.min(Math.max(s, 0), 100), Math.min(Math.max(l, 0), 100));
				
				colcomp.value = hex;				

				hslobj[key] = "hsl(" + h+"deg " + s + "% "+ l +"%)";
				inner_styles += key + ':' + hslobj[key] + ';'; 
			}
		}		
		else
		{			
			inner_styles += key + ':' + styleobj[key] + ';'; 
		}

	}
	
	isColorsInv = false;
	
	const preview_styles = gradioApp().querySelector('#preview-styles');
	preview_styles.innerHTML = ':root {'+ inner_styles +'}';
	preview_styles.innerHTML +='@media only screen and (max-width: 860px) {:root{--ae-outside-gap-size: var(--ae-mobile-outside-gap-size);--ae-inside-padding-size: var(--ae-mobile-inside-padding-size);}}';

	
	const vars_textarea = gradioApp().querySelector('#theme_vars textarea');
	vars_textarea.value = inner_styles;			

	const inputEvent = new Event("input");
    Object.defineProperty(inputEvent, "target", {value: vars_textarea})
	vars_textarea.dispatchEvent(inputEvent);
	
}

function updateTheme(vars){
	
	let inner_styles = "";
	
	for (let i=0; i<vars.length-1; i++) {		
		let key = vars[i].split(":");
		let id = key[0].replace(/\s+/g, "");;		
		let val = key[1].trim();
		
		styleobj[id] = val;
		inner_styles += id + ':' + val + ';';
		
		gradioApp().querySelectorAll('#'+id+' input').forEach((elem) => {			
			if(val.indexOf("hsl") != -1){
				let hsl = toHSLArray(val);
				let hex = hslToHex(hsl[0], hsl[1], hsl[2]);	
				elem.value = hex;
			}else{
				elem.value = val.split("px")[0];
			}
		})	
	}
	
	const preview_styles = gradioApp().querySelector('#preview-styles');
	
	if(preview_styles){
		preview_styles.innerHTML = ':root {'+ inner_styles +'}';
		preview_styles.innerHTML +='@media only screen and (max-width: 860px) {:root{--ae-outside-gap-size: var(--ae-mobile-outside-gap-size);--ae-inside-padding-size: var(--ae-mobile-inside-padding-size);}}';
	}else{
		
		const r = gradioApp();		
		const style = document.createElement('style');
		style.id="preview-styles";
		style.innerHTML = ':root {'+ inner_styles +'}';
		style.innerHTML +='@media only screen and (max-width: 860px) {:root{--ae-outside-gap-size: var(--ae-mobile-outside-gap-size);--ae-inside-padding-size: var(--ae-mobile-inside-padding-size);}}';
		r.appendChild(style);
	}
	
	const vars_textarea = gradioApp().querySelector('#theme_vars textarea');
	const css_textarea = gradioApp().querySelector('#theme_css textarea');
	
	vars_textarea.value = inner_styles;
	css_textarea.value = css_textarea.value;
	
	//console.log(Object);

	const vEvent = new Event("input");
	const cEvent = new Event("input");
    Object.defineProperty(vEvent, "target", {value: vars_textarea})
	Object.defineProperty(cEvent, "target", {value: css_textarea})
	vars_textarea.dispatchEvent(vEvent);
	css_textarea.dispatchEvent(cEvent);
	

}

function applyTheme(){
	console.log("apply");
}

function initTheme() { 


	const current_style = gradioApp().querySelector('.gradio-container > style');
	//console.log(current_style);
	//const head = document.head;		
	//head.appendChild(current_style);

	const css_styles = current_style.innerHTML.split("/*BREAKPOINT_CSS_CONTENT*/");
	let init_css_vars = css_styles[0].split("}")[0].split("{")[1];
	init_css_vars = init_css_vars.replace(/\n|\r/g, "");

	let init_vars = init_css_vars.split(";");
	let vars = init_vars;
	
	//console.log(vars);
	
	const vars_textarea = gradioApp().querySelector('#theme_vars textarea');
	const css_textarea = gradioApp().querySelector('#theme_css textarea');
	//const result_textarea = gradioApp().querySelector('#theme_result textarea');
	vars_textarea.value = init_css_vars;
	css_textarea.value = "/*BREAKPOINT_CSS_CONTENT*/" + css_styles[1] + "/*BREAKPOINT_CSS_CONTENT*/";
	
	updateTheme(vars);
	
	//vars_textarea.addEventListener("change", function(e) {				
		//e.preventDefault();	
		//e.stopPropagation();
		//vars = vars_textarea.value.split(";");
		//console.log(e);
		//updateTheme(vars);
	
	//})

	const preview_styles = gradioApp().querySelector('#preview-styles');
	let intervalChange;



	gradioApp().querySelectorAll('#ui_theme_settings input').forEach((elem) => {
		
		elem.addEventListener("input", function(e) {

			let celem = e.currentTarget;			
			let val = e.currentTarget.value;
			let curr_val;
			
			switch(e.currentTarget.type) {
				case "range":
					celem = celem.parentElement;				
					val = e.currentTarget.value+"px";					
					break;
				case "color":
					celem = celem.parentElement.parentElement;
					val = e.currentTarget.value;						
					break;
				case "number":
					celem = celem.parentElement.parentElement.parentElement;
					val = e.currentTarget.value+"px";					
					break;			
			}
					
			styleobj[celem.id] = val;
			
			//console.log(styleobj);
			
				
			if(intervalChange != null) clearInterval(intervalChange);
			intervalChange = setTimeout(() => {
				let inner_styles = "";
				
				for (const key in styleobj) {
					inner_styles += key + ':' + styleobj[key] + ';'; 			
				}
				
				vars = inner_styles.split(";");			
				preview_styles.innerHTML = ':root {'+ inner_styles +'}';
				preview_styles.innerHTML +='@media only screen and (max-width: 860px) {:root{--ae-outside-gap-size: var(--ae-mobile-outside-gap-size);--ae-inside-padding-size: var(--ae-mobile-inside-padding-size);}}';

				vars_textarea.value = inner_styles;			
				const vEvent = new Event("input");
				Object.defineProperty(vEvent, "target", {value: vars_textarea})
				vars_textarea.dispatchEvent(vEvent);
				
				offsetColorsHSV(hsloffset);
			}, 1000)
			
		})
	})
	

	
	const reset_btn = gradioApp().getElementById('theme_reset_btn');	
	reset_btn.addEventListener("click", function(e) {
		e.preventDefault();
		e.stopPropagation();
		gradioApp().querySelectorAll('#ui_theme_hsv input').forEach((elem) => {
			elem.value = 0;
		})
		hsloffset = [0,0,0];
		updateTheme(init_vars);		
	})
	
	/*
	const apply_btn = gradioApp().getElementById('theme_apply_btn');	
	apply_btn.addEventListener("click", function(e) {
		e.preventDefault();
		e.stopPropagation();				
		init_css_vars = vars_textarea.value.replace(/\n|\r/g, "");		
		vars_textarea.value = init_css_vars;
		
		init_vars = init_css_vars.split(";");
		vars = init_vars;		
		updateTheme(vars);
	})
	*/
	
	let intervalCheck;
	function dropDownOnChange(){
		
		if(init_css_vars != vars_textarea.value){			
			clearInterval(intervalCheck);
			init_css_vars = vars_textarea.value.replace(/\n|\r/g, "");		
			vars_textarea.value = init_css_vars;		
			init_vars = init_css_vars.split(";");
			vars = init_vars;		
			updateTheme(vars);			
		}
		
	}

	
	
	const drop_down = gradioApp().querySelector('#themes_drop_down');	
	drop_down.addEventListener("click", function(e) {
		if(intervalCheck != null) clearInterval(intervalCheck);
		intervalCheck = setInterval(dropDownOnChange, 100);
		console.log("ok");
	})
	
	let hsloffset = [0,0,0];
	
	const hue = gradioApp().querySelectorAll('#theme_hue input').forEach((elem) => {		
		elem.addEventListener("change", function(e) {
			e.preventDefault();
			e.stopPropagation();
			hsloffset[0] = e.currentTarget.value;
			offsetColorsHSV(hsloffset);		
		})			
	})
	
	const sat = gradioApp().querySelectorAll('#theme_sat input').forEach((elem) => {		
		elem.addEventListener("change", function(e) {
			e.preventDefault();
			e.stopPropagation();
			hsloffset[1] = e.currentTarget.value;
			offsetColorsHSV(hsloffset);		
		})			
	})
	
	const brt = gradioApp().querySelectorAll('#theme_brt input').forEach((elem) => {		
		elem.addEventListener("change", function(e) {
			e.preventDefault();
			e.stopPropagation();
			hsloffset[2] = e.currentTarget.value;
			offsetColorsHSV(hsloffset);		
		})			
	})
	
	const inv_btn = gradioApp().getElementById('theme_invert_btn');	
	inv_btn.addEventListener("click", function(e) {
		e.preventDefault();
		e.stopPropagation();
		isColorsInv = !isColorsInv;
		offsetColorsHSV(hsloffset);	
	})
	
}


function observeGradioApp() {
	const observer = new MutationObserver(() => {

		const block = gradioApp().getElementById('tab_ui_theme');
		if (block) {
			observer.disconnect();
			
			setTimeout(() => {
			  initTheme();
			}, "500")
			
			
		}
	});
	observer.observe(gradioApp(), { childList: true, subtree: true });
}

document.addEventListener("DOMContentLoaded", () => {
    observeGradioApp();
});

