//called from progressbar
function showGalleryImage() {
  //need to clean up the old code
}

let like;
let tile;

let slide = 0;
let gallery = [];
let fullImg_src;
//let control = ["pan","undo","like","tile","page","fullscreen","autofit","zoom-in","zoom-out","clear","close","download","prev","next"];
let control = [
  "like",
  "tile",
  "page",
  "fullscreen",
  "autofit",
  "zoom-in",
  "zoom-out",
  "clear",
  "close",
  "download",
  "prev",
  "next",
];

let img_browser;
let img_file_name;

let spl_pane;
let spl_zoom_out;
let spl_zoom_in;
let spotlight_gallery;

function tile_zoom_update(val) {
  let current_tile_state_size = gallery[slide].tile_size;
  current_tile_state_size += val * 5;
  current_tile_state_size = Math.max(5, Math.min(100, current_tile_state_size));
  spl_pane.style.setProperty("background-size", current_tile_state_size + "%");
  gallery[slide].tile_size = current_tile_state_size;
}

function tile_wheel(event) {
  let delta = event["deltaY"];
  delta = (delta < 0 ? 1 : delta ? -1 : 0) * 0.5;
  tile_zoom_update(delta);
}
function tile_zoom_in(event) {
  tile_zoom_update(1);
}
function tile_zoom_out(event) {
  tile_zoom_update(-1);
}

function removeTile() {
  spl_pane.removeEventListener("wheel", tile_wheel);
  spl_zoom_out.removeEventListener("click", tile_zoom_out);
  spl_zoom_in.removeEventListener("click", tile_zoom_in);

  spl_pane.classList.remove("hide");
  spl_pane.style.setProperty("background-image", "none");
  spotlight_gallery.zoom(0.0);
}

function addTile(spl_img) {
  spl_pane.addEventListener("wheel", tile_wheel);
  spl_zoom_out.addEventListener("click", tile_zoom_out);
  spl_zoom_in.addEventListener("click", tile_zoom_in);

  const current_tile_state_size = gallery[slide].tile_size;
  spl_pane.classList.add("hide");
  spl_pane.style.setProperty("background-position", "center");
  spl_pane.style.setProperty("background-size", current_tile_state_size + "%");
  if (spl_img) {
    spl_pane.style.setProperty("background-image", `url(${spl_img.src})`);
  }
}

function tile_handler(event) {
  const current_tile_state = !gallery[slide].tile;
  gallery[slide].tile = current_tile_state;

  this.classList.toggle("on");

  if (current_tile_state) {
    const spl_img = gradioApp().querySelector("#spotlight-gal .spl-pane img");
    addTile(spl_img);
  } else {
    removeTile();
  }
}
function like_handler(event) {
  const current_like_state = !gallery[slide].like;
  gallery[slide].like = current_like_state;
  this.classList.toggle("on");

  if (current_like_state) {
    // add to favorites ...
    //img_file_name.value = gallery[slide].src;
    //console.log(gallery[slide].src);
  } else {
    // remove from favorites ...
  }
}

function createGallerySpotlight() {
  //console.log("clicked");
  slide = 0;
  gallery = [];

  gradioApp()
    .querySelectorAll("#" + selectedTabItemId + " .thumbnails img")
    .forEach(function (elem, i) {
      elem.setAttribute("gal-id", i);
      //if(fullImg_src == elem.src) slide = parseInt(i+1);
      if (elem.parentElement.className.indexOf("selected") != -1)
        slide = parseInt(i + 1);
      //console.log(slide);
      gallery[i] = {
        src: elem.src,
        title: "Seed:" + elem.src,
        //description: "This is a description.",
        like: false,
        tile: false,
        tile_size: 50,
      };
    });

  const options = {
    class: "sd-gallery",
    index: slide,
    //control: ["like","page","theme","fullscreen","autofit","zoom-in","zoom-out","close","download","play","prev","next"],
    control: control,
    //animation: animation,
    onshow: function (index) {},
    onchange: function (index, options) {
      slide = index - 1;
      tile.classList.toggle("on", gallery[slide].tile);
      //if(img_browser){
      like.classList.toggle("on", gallery[slide].like);

      //}

      spl_pane = gradioApp().querySelector(
        "#spotlight-gal .spl-pane:nth-child(" + index + ")"
      );
      spl_zoom_out = gradioApp().querySelector("#spotlight-gal .spl-zoom-out");
      spl_zoom_in = gradioApp().querySelector("#spotlight-gal .spl-zoom-in");

      const current_tile_state = gallery[slide].tile;
      if (current_tile_state) {
        addTile();
      } else {
        removeTile();
      }
    },
    onclose: function (index) {
      gradioApp()
        .querySelector(
          "#" +
            selectedTabItemId +
            " .thumbnails .thumbnail-item:nth-child(" +
            (slide + 1) +
            ")"
        )
        .click();
    },
  };

  //assign(options, modifier);

  spotlight_gallery.show(gallery, options);
  spotlight_gallery.panzoom(true);
}

function fullImg_click_handler(e) {
  e.stopPropagation();
  e.preventDefault();
  createGallerySpotlight();
}

let intervalUiUpdateIViewer;
function onUiHeaderTabUpdate() {
  if (intervalUiUpdateIViewer != null) clearInterval(intervalUiUpdateIViewer);
  intervalUiUpdateIViewer = setInterval(onUiUpdateIViewer, 500);
}

let fullImg_preview;
function onUiUpdateIViewer() {
  clearInterval(intervalUiUpdateIViewer);
  //update_performant_inputs(selectedTabItemId);

  //fullImg_preview = gradioApp().querySelector('#'+selectedTabItemId+' [id$="2img_results"] .modify-upload + img.w-full.object-contain');
  fullImg_preview = gradioApp().querySelector(
    "#" + selectedTabItemId + " .preview > img"
  );
  if (opts.js_modal_lightbox && fullImg_preview) {
    fullImg_src = fullImg_preview.src;
    fullImg_preview.removeEventListener("click", fullImg_click_handler);
    fullImg_preview.addEventListener("click", fullImg_click_handler, true); //bubbling phase

    /*
		// this is an idea to integrate image browser extension seamlesly, 
		// without the need to change to the image browser tab extension users will be able to review images after generation
		// and add them to favorites or delete the ones that don't like on the spot
		const img_browser = gradioApp().querySelector('[id$="2img_images_history"]');
		const tbname = selectedTabItemId.split("_")[1];
		if(img_browser && tbname ==("txt2img" || "img2img")){
			const images_history = gradioApp().querySelector('[id$="'+tbname+'_images_history"]');
			const history_button_panel = images_history.querySelector('[id$="'+tbname+'_images_history_button_panel"]');
			const labels = images_history.querySelectorAll('label.block span');
			for(let i=0;i<labels.length;i++){
				//console.log(labels[i].innerHTML)
				if(labels[i].innerHTML == 'File Name'){
					img_file_name = labels[i].parentElement.querySelector("textarea");
					console.log(img_file_name.value);
					break;
				}
			}
		}
		*/
  }
}

onUiUpdate(function () {
  if (intervalUiUpdateIViewer != null) clearInterval(intervalUiUpdateIViewer);
  intervalUiUpdateIViewer = setInterval(onUiUpdateIViewer, 500);

});

onUiLoaded(function () {
  spotlight_gallery = new Spotlight();
  spotlight_gallery.init(
    gradioApp().querySelector(".gradio-container"),
    "-gal"
  );
  tile = spotlight_gallery.addControl("tile", tile_handler);
  like = spotlight_gallery.addControl("like", like_handler);
});

document.addEventListener("DOMContentLoaded", function () {});
