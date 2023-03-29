/**
 * Spotlight.js v0.7.8
 * Copyright 2019-2021 Nextapps GmbH
 * Author: Thomas Wilkerling
 * Licence: Apache-2.0
 * https://github.com/nextapps-de/spotlight
 */
 /* 
 this library was rewriten by anapnoe to work with https://github.com/anapnoe/stable-diffusion-webui-ux
 it supports multi-instancing, enable-disable panzoom functionallity and input type tools
 */

(function (window) {
    'use strict';
    const Spotlight = function (selector, options = {}) {
        this.controls = [
            "info",
            "theme",
            "download",
            "play",
            "page",
            "close",
            "autofit",
            "zoom-in",
            "zoom-out",
            "prev",
            "next",
            "fullscreen"
        ];
        this.controls_default = {
            //"info": 1,
            "page": 1,
            "close": 1,
            "autofit": 1,
            "zoom-in": 1,
            "zoom-out": 1,
            "prev": 1,
            "next": 1,
            "fullscreen": 1
        };
        this.keycodes = {
            BACKSPACE: 8,
            ESCAPE: 27,
            SPACEBAR: 32,
            LEFT: 37,
            RIGHT: 39,
            UP: 38,
            NUMBLOCK_PLUS : 107,
            PLUS: 187,
            DOWN: 40,
            NUMBLOCK_MINUS: 109,
            MINUS: 189,
            INFO: 73
        };
		
		const self = this;
		
		Object.assign || (Object.assign =
			function(target, source){
				//console.log("assign", target, source);
				const keys = Object.keys(/** @type {!Object} */(source));
				for(let i = 0, key; i < keys.length; i++){
					key = keys[i];
					target[key] = /*"" +*/ source[key];
				}
			return target;
			}
		);

		Element.prototype.closest || (Element.prototype.closest = function(classname){
			//console.log("closest", classname);
			classname = classname.substring(1);
			let node = this;
			while(node && (node.nodeType === 1)){
				if(node.classList.contains(classname)){
					return (node);
				}
				node = node.parentElement /* || node.parentNode */;
			}
			return null;
		});
		
        this.addClass = (node, class_name) => {
            self.toggleClass(node, class_name, true);
        }
        this.removeClass = (node, class_name) =>{       
            self.toggleClass(node, class_name);
        }
        this.toggleClass = (node, class_name, state) =>{
            node.classList[state ? "add" : "remove"](class_name);
        }
        this.hasClass = (node, class_name) =>{
            return node.classList.contains(class_name);
        }
        this.setStyle = (node, style, value) =>{
            value = "" + value;
            if(node["_s_" + style] !== value){
                node.style.setProperty(style, value);
                node["_s_" + style] = value;
            }
        }
        let tmp = 0;
        this.prepareStyle = (node, fn) => {
            if(fn){
                self.setStyle(node, "transition", "none");
                fn();
            }
            // force applying styles (quick-fix for closure compiler):
            tmp || (tmp = node.clientTop && 0); // clientWidth      
            fn && self.setStyle(node, "transition", "");
        }
        this.setText = (node, text) => {
            node.firstChild.nodeValue = text;
        }
        this.getByClass = (classname, context) => {
            return (context || document).getElementsByClassName(classname);
        }
        this.getByTag = (tag, context) => {
            return (context || document).getElementsByTagName(tag);
        }
        this.addListener = (node, event, fn, mode) => {       
            self.toggleListener(true, node, event, fn, mode);
        }
        this.removeListener = (node, event, fn, mode) => {
            self.toggleListener(false, node, event, fn, mode);
        }
        this.toggleListener = (state, node, event, fn, mode) => {  
            node[(state ? "add" : "remove") + "EventListener"](event, fn, mode || (mode === false) ? mode : true);
        }
        this.cancelEvent = (event, prevent) => {
            event.stopPropagation();
            //event.stopImmediatePropagation();
            prevent && event.preventDefault();
        }
        this.downloadImage = (body, image) => {      
            const link = self.createElement("a");
            const src = image.src;
            link.href = src;
            link.download = src.substring(src.lastIndexOf("/") + 1);
            self.body.appendChild(link);
            link.click();
            self.body.removeChild(link);
        }
        this.createElement = (element) => {
            return document.createElement(element);
        }
        this.toggleDisplay = (node, state) => {
            self.setStyle(node, "display", state ? "" : "none");
        }
        this.toggleVisibility = (node, state) => {       
            self.setStyle(node, "visibility", state ? "" : "hidden");
        }
        this.toggleAnimation = (node, state) => {
            self.setStyle(node, "transition", state ? "" : "none");
        }
        this.widget = this.createElement("div");     
        this.widget.innerHTML = (
            '<div class=spl-spinner></div>' +
            '<div class=spl-track>' +
                '<div class=spl-scene>' +
                    '<div class=spl-pane></div>' +
                '</div>' +
            '</div>' +
            '<div class=spl-header>' +
                '<div class=spl-page> </div>' +
            '</div>' +
            '<div class=spl-progress></div>' +
            '<div class=spl-footer>' +
                '<div class=spl-title> </div>' +
                '<div class=spl-description> </div>' +
                '<div class=spl-button> </div>' +
            '</div>' +
            '<div class=spl-prev></div>' +
            '<div class=spl-next></div>'
        );
		
		this.video_support = {};
		this.tpl_video = this.createElement("video");

		this.parse_src = (anchor, size, options, media) => {
			let src, diff;
			if(media !== "node"){
				const keys = Object.keys((options));
				for(let x = 0, key; x < keys.length; x++){
					key = keys[x];
					if((key.length > 3) && (key.indexOf("src") === 0)){
						if(media === "video"){
							const cache = self.video_support[key];
							if(cache){
								if(cache > 0){
									src = options[key];
									break;
								}
							}
							else if(self.tpl_video.canPlayType("video/" + key.substring(3).replace("-", "").toLowerCase())){
								self.video_support[key] = 1;
								src = options[key];
								break;
							}
							else{
								self.video_support[key] = -1;
							}
						}
						else{
							// Image Media:
							const res = parseInt(key.substring(4), 10);
							if(res){
								const abs = Math.abs(size - res);
								if(!diff || (abs < diff)){
									diff = abs;
									src = options[key];
								}
							}
						}
					}
				}
			}
			return src || options["src"] || options["href"] || anchor["src"] || anchor["href"];
		}

        this.controls_dom = {};
        const connection = navigator["connection"];
        const dpr = window["devicePixelRatio"] || 1;
		
		
        this.x;
        this.y;
        this.startX;
        this.startY;
        this.viewport_w;
        this.viewport_h;
        this.media_w;
        this.media_h;
        this.scale;

        this.is_down;
        this.dragged;
        this.slidable;
        this.toggle_autofit;
        this.toggle_theme;

        this.current_slide;
        this.slide_count;
        this.anchors;
        this.options;
        this.options_media;
        this.options_group;
        this.options_infinite;
        this.options_progress;
        this.options_onshow;
        this.options_onchange;
        this.options_onclose;
        this.options_fit;
        this.options_autohide;
        this.options_autoslide;
        this.options_theme;
        this.options_preload;
        this.options_href;
        this.options_click;
        this.options_class;
        this.delay;

        this.animation_scale;
        this.animation_fade;
        this.animation_slide;
        this.animation_custom;

        this.body;
        this.doc;
        this.panel;
        this.panes;
        this.media;
        this.media_next = this.createElement("img");
        this.slider;
        this.header;
        this.footer;
        this.footer_visible = 0;
        this.title;
        this.description;
        this.button;
        this.page_prev;
        this.page_next;
        this.maximize;
        this.page;
        this.player;
        this.progress;
        this.spinner;

        this.gallery;
        this.gallery_next;
        this.playing;
        this.hide;
        this.hide_cooldown;
        this.id = "spotlight";

        this.prefix_request;
        this.prefix_exit;
		
		this.track;

        this.addListener(document, "click", this.dispatch);

        this.init = (selector, instanceId) => {

            //setUserOptions();
            //setSliderOptions();

            if (self.body) {
                return;
            }

            self.id = "spotlight" + instanceId;
            self.widget.id = "spotlight" + instanceId;            
            self.doc = document.body;
            self.body = selector;
            self.slider = self.getOneByClass("scene");
            self.header = self.getOneByClass("header");
            self.footer = self.getOneByClass("footer");
            self.title = self.getOneByClass("title");
            self.description = self.getOneByClass("description");
            self.button = self.getOneByClass("button");
            self.page_prev = self.getOneByClass("prev");
            self.page_next = self.getOneByClass("next");
            self.page = self.getOneByClass("page");
            self.progress = self.getOneByClass("progress");
            self.spinner = self.getOneByClass("spinner");
            self.panes = [self.getOneByClass("pane")];
			
			//console.log(selector);

            self.addControl("close", self.close);

            self.doc[self.prefix_request = "requestFullscreen"] ||
                self.doc[self.prefix_request = "msRequestFullscreen"] ||
                self.doc[self.prefix_request = "webkitRequestFullscreen"] ||
                self.doc[self.prefix_request = "mozRequestFullscreen"] ||
                (self.prefix_request = "");

            if (self.prefix_request) {

                self.prefix_exit = (

                    self.prefix_request.replace("request", "exit")
                        .replace("mozRequest", "mozCancel")
                        .replace("Request", "Exit")
                );

                self.maximize = self.addControl("fullscreen", self.fullscreen);
            }
            else {

                self.controls.pop(); 
            }

            self.addControl("info", self.info);
            self.addControl("autofit", self.autofit);
            self.addControl("zoom-in", self.zoom_in);
            self.addControl("zoom-out", self.zoom_out);
            self.addControl("theme", self.theme);
            self.player = self.addControl("play", self.play);
            self.addControl("download", self.download);

            self.addListener(self.page_prev, "click", self.prev);
            self.addListener(self.page_next, "click", self.next);

            self.track = self.getOneByClass("track");

            self.addListener(self.track, "mousedown", self.start);
            self.addListener(self.track, "mousemove", self.move);
            self.addListener(self.track, "mouseleave", self.end);
            self.addListener(self.track, "mouseup", self.end);

            self.addListener(self.track, "touchstart", self.start, { "passive": false });
            self.addListener(self.track, "touchmove", self.move, { "passive": true });
            //addListener(self.track, "touchcancel", end);
            self.addListener(self.track, "touchend", self.end);
            // click listener for the wrapper "track" is already covered
            //addListener(self.track, "click", menu);

            self.addListener(self.button, "click", function () {
                if (self.options_click) {
                    self.options_click(self.current_slide, self.options);
                }
                else if (self.options_href) {
                    location.href = self.options_href;
                }
            });
            //this.onInit();
        }


        this.getOneByClass = (classname) => {
            return this.controls_dom[classname] = this.getByClass("spl-" + classname, this.widget)[0];
        }

        this.addControl = (classname, fn, html="") => {
            //console.log("addControl", classname, fn);
            const div = self.createElement("div");
            div.className = "spl-" + classname;
			div.innerHTML = html; 
            self.addListener(div, "click", fn);
            self.header.appendChild(div);
            return self.controls_dom[classname] = div;
        }

        this.removeControl = (classname) => {
            //console.log("dispatch", classname);
            const div = self.controls_dom[classname];
            if (div) {
                self.header.removeChild(div);
                self.controls_dom[classname] = null;
            }
        }

        this.dispatch = (event) => {
            //console.log("dispatch");
            const target = event.target.closest(".spotlight");
            if (target) {
                self.cancelEvent(event, true);
                const group = target.closest(".spotlight-group");
                self.anchors = self.getByClass("spotlight", group);
                // determine current selected index
                for (let i = 0; i < self.anchors.length; i++) {
                    if (self.anchors[i] === target) {
                        self.options_group = group && group.dataset;
                        self.init_gallery(i + 1);
                        break;
                    }
                }
            }
        }

        this.show = (gallery, group, index) => {        
			//console.log(gallery, group, index);
			//console.log(group);
            self.anchors = gallery;
            if (group) {

                self.options_group = group;
                self.options_onshow = group["onshow"];
                self.options_onchange = group["onchange"];
                self.options_onclose = group["onclose"];
                index = index || group["index"];
            }
			
            self.init_gallery(index);
        }
        //state, node, event, fn, mode
        this.panzoom = (state) => {
            self.toggleListener(state, self.track, "wheel", self.wheel_listener, { passive: false });
            self.toggleListener(state, self.track, "mousedown", self.start);
            self.toggleListener(state, self.track, "mousemove", self.move);
            self.toggleListener(state, self.track, "mouseleave", self.end);
            self.toggleListener(state, self.track, "mouseup", self.end);
            self.toggleListener(state, self.track, "touchstart", self.start, { passive: false });
            self.toggleListener(state, self.track, "touchmove", self.move, { passive: true });
            self.toggleListener(state, self.track, "touchend", self.end);
        }

        this.init_gallery = (index) => {
            //console.log("init_gallery", index);
            self.slide_count = self.anchors.length;
            if (self.slide_count) {

                self.body || self.init();

                self.options_onshow && self.options_onshow(index);
                const pane = self.panes[0];
                //const parent = pane.parentNode;
				const parent = pane.parentElement;

                for (let i = self.panes.length; i < self.slide_count; i++) {
                    const clone = pane.cloneNode(false);
                    self.setStyle(clone, "left", (i * 100) + "%");
                    parent.appendChild(clone);
                    self.panes[i] = clone;
                }

                if (!self.panel) {
                    self.body.appendChild(self.widget);
                    self.update_widget_viewport();
                    //resize_listener();
                }
                self.current_slide = index || 1;
                self.toggleAnimation(self.slider);
                self.setup_page(true);
                self.prefix_request && self.detect_fullscreen();
                self.show_gallery();
            }
        }

        this.parse_option = (key, is_default) => {
            //console.log("parse_option", key, is_default);
            let val = options[key];
            if (typeof val !== "undefined") {
                val = "" + val;
                return (val !== "false") && (val || is_default);
            }
            return is_default;
        }

        this.apply_options = (anchor) => {
            //console.log("apply_options", anchor);
            self.options = {};
            self.options_group && Object.assign(self.options, self.options_group);
            Object.assign(self.options, anchor.dataset || anchor);

            // TODO: theme is icon and option field!

            self.options_media = self.options["media"];
            self.options_click = self.options["onclick"];
            self.options_theme = self.options["theme"];
            self.options_class = self.options["class"];
            self.options_autohide = self.parse_option("autohide", false);
            self.options_infinite = self.parse_option("infinite");
            self.options_progress = self.parse_option("progress", true);
            self.options_autoslide = self.parse_option("autoslide");
            self.options_preload = self.parse_option("preload", true);
            self.options_href = self.options["buttonHref"];
            self.delay = (self.options_autoslide && parseFloat(self.options_autoslide)) || 7;
            self.toggle_theme || (self.options_theme && self.theme(self.options_theme));
            self.options_class && self.addClass(self.widget, self.options_class);
            self.options_class && self.prepareStyle(self.widget)

            const control = self.options["control"];

            // determine controls
            if (control) {
                const whitelist = (
                    typeof control === "string" ? control.split(",") : control
                );
                // prepare to false when using whitelist
                for (let i = 0; i < self.controls.length; i++) {
                    self.options[self.controls[i]] = false;
                }
                // apply whitelist
                for (let i = 0; i < whitelist.length; i++) {
                    const option = whitelist[i].trim();
                    // handle shorthand "zoom"
                    if (option === "zoom") {
                        self.options["zoom-in"] = self.options["zoom-out"] = true;
                    }
                    else {
                        self.options[option] = true;
                    }
                }
            }

            // determine animations
            const animation = self.options["animation"];
            self.animation_scale = self.animation_fade = self.animation_slide = !animation;
            self.animation_custom = false;

            if (animation) {
                const whitelist = (
                    typeof animation === "string" ?
                        animation.split(",") : animation
                );

                // apply whitelist
                for (let i = 0; i < whitelist.length; i++) {
                    const option = whitelist[i].trim();
                    if (option === "scale") {
                        self.animation_scale = true;
                    }
                    else if (option === "fade") {
                        self.animation_fade = true;
                    }
                    else if (option === "slide") {
                        self.animation_slide = true;
                    }
                    else if (option) {
                        self.animation_custom = option;
                    }
                }
            }
            self.options_fit = self.options["fit"];
        }

        this.prepare_animation = (prepare) => {
            //console.log("prepare_animation", prepare);
            if (prepare) {
                self.prepareStyle(self.media, self.prepare_animation);
            }
            else {
                self.toggleAnimation(self.slider, self.animation_slide);
                self.setStyle(self.media, "opacity", self.animation_fade ? 0 : 1);
                self.update_scroll(self.animation_scale && 0.8);
                self.animation_custom && self.addClass(self.media, self.animation_custom);
            }
        }

        this.init_slide = (index) => {
            //console.log("init_slide", index);
            self.panel = self.panes[index - 1];
            self.media = (self.panel.firstChild);
            self.current_slide = index;

            if (self.media) {
                self.disable_autoresizer();
                if (self.options_fit) {
                    self.addClass(self.media, self.options_fit);
                }
                self.prepare_animation(true);
                self.animation_custom && self.removeClass(self.media, self.animation_custom);
                self.animation_fade && self.setStyle(self.media, "opacity", 1);
                self.animation_scale && self.setStyle(self.media, "transform", "");
                self.setStyle(self.media, "visibility", "visible");

                self.gallery_next && (self.media_next.src = self.gallery_next);
                self.options_autoslide && self.animate_bar(self.playing);
            }
            else {

                const type = self.gallery.media;
                const options_spinner = self.parse_option("spinner", true);

                if (type === "node") {
                    self.media = self.gallery.src;
                    if (typeof self.media === "string") {
                        self.media = gradioApp().querySelector(self.media);
                    }

                    if (self.media) {
                        //self.media._root || (self.media._root = self.media.parentNode);
						self.media._root || (self.media._root = self.media.parentElement);
                        self.update_media_viewport();
                        self.panel.appendChild(self.media);
                        self.init_slide(index);
                    }
                    return;
                }
                else {
                    self.toggle_spinner(self.options_spinner, true);
                    self.media = (self.createElement("img"));
                    self.media.onload = function () {
                        if (self.media === self) {
                            self.media.onerror = null;
                            self.toggle_spinner(self.options_spinner);
                            self.init_slide(index);
                            self.update_media_viewport();
                        }
                    };
                    //media.crossOrigin = "anonymous";
                    self.media.src = self.gallery.src;
                    self.panel.appendChild(self.media);
                }

                if (self.media) {
                    self.options_spinner || self.setStyle(self.media, "visibility", "visible");
                    self.media.onerror = function () {
                        if (self.media === self) {
                            self.checkout(self.media);
                            self.addClass(self.spinner, "error");
                            self.toggle_spinner(self.options_spinner);
                        }
                    };
                }
            }
        }

        this.toggle_spinner = (options_spinner, is_on) => {         
            options_spinner && self.toggleClass(self.spinner, "spin", is_on);
        }

        this.has_fullscreen = () => {
            return (
                document["fullscreen"] ||
                document["fullscreenElement"] ||
                document["webkitFullscreenElement"] ||
                document["mozFullScreenElement"]
            );
        }

        this.resize_listener = () => {
            self.update_widget_viewport()
            self.media && self.update_media_viewport();
            if (self.prefix_request) {
                const is_fullscreen = self.has_fullscreen();
                self.toggleClass(self.maximize, "on", is_fullscreen)
                is_fullscreen || self.detect_fullscreen();
            }
            //update_scroll();
        }

        this.detect_fullscreen = () => {
            self.toggleDisplay(self.maximize, (screen.availHeight - window.innerHeight) > 0);
        }

        this.update_widget_viewport = () => {
            self.viewport_w = self.widget.clientWidth;
            self.viewport_h = self.widget.clientHeight;
        }

        this.update_media_viewport = () => {
            self.media_w = self.media.clientWidth;
            self.media_h = self.media.clientHeight;
        }

        // function update_media_dimension(){
        //
        //     media_w = media.width;
        //     media_h = media.height;
        // }

        this.update_scroll = (force_scale) => {
            self.setStyle(self.media, "transform", "translate(-50%, -50%) scale(" + (force_scale || self.scale) + ")");
        }

        this.update_panel = (x, y) => {
            self.setStyle(self.panel, "transform", x || y ? "translate(" + x + "px, " + y + "px)" : "");
        }

        this.update_slider = (index, prepare, offset) => {
            if (prepare) {
                self.prepareStyle(self.slider, function () {
                    self.update_slider(index, false, offset);
                });
            }
            else {
                self.setStyle(self.slider, "transform", "translateX(" + (-index * 100 + (offset || 0)) + "%)");
            }
        }

        this.toggle_listener = (install) => {
            //console.log("toggle_listener", install);
            self.toggleListener(install, window, "keydown", self.key_listener);
            //toggleListener(install, window, "wheel", wheel_listener);
            self.toggleListener(install, window, "resize", self.resize_listener);
            //self.toggleListener(install, window, "popstate", self.history_listener);
        }

        this.history_listener = (event) => {
            //console.log("history_listener");
            if (self.panel && event.state["spl"]) {
                self.close(true);
            }
        }

        this.key_listener = (event) => {
            //console.log("key_listener");
            if (self.panel) {
                const zoom_enabled = self.options["zoom-in"] !== false;
                switch (event.keyCode) {
                    case self.keycodes.BACKSPACE:
                        zoom_enabled && self.autofit();
                        break;
                    case self.keycodes.ESCAPE:
                        self.close();
                        break;
                    case self.keycodes.SPACEBAR:
                        self.options_autoslide && self.play();
                        break;
                    case self.keycodes.LEFT:
                        self.prev();
                        break;
                    case self.keycodes.RIGHT:
                        self.next();
                        break;
                    case self.keycodes.UP:
                    case self.keycodes.NUMBLOCK_PLUS:
                    case self.keycodes.PLUS:
                        zoom_enabled && self.zoom_in();
                        break;
                    case self.keycodes.DOWN:
                    case self.keycodes.NUMBLOCK_MINUS:
                    case self.keycodes.MINUS:
                        zoom_enabled && self.zoom_out();
                        break;
                    case self.keycodes.INFO:
                        self.info();
                        break;
                }
            }
        }

        this.wheel_listener = (event) => {
            //console.log("wheel_listener");
            if (self.panel && (self.options["zoom-in"] !== false)) {
                let delta = event["deltaY"];
                delta = (delta < 0 ? 1 : delta ? -1 : 0) * 0.5;
                if (delta < 0) {
                    self.zoom_out();
                }
                else {
                    self.zoom_in();
                }
            }
        }

        this.play = (init, _skip_animation) => {
            //console.log("play", init);
            const state = (typeof init === "boolean" ? init : !this.playing);
            if (state === !self.playing) {
                self.playing = self.playing ? clearTimeout(self.playing) : 1;
                self.toggleClass(self.player, "on", self.playing);
                _skip_animation || self.animate_bar(self.playing);
            }
        }

        this.animate_bar = (start) => {
            //console.log("animate_bar", start);
            if (self.options_progress) {
                self.prepareStyle(self.progress, function () {
                    self.setStyle(self.progress, "transition-duration", "");
                    self.setStyle(self.progress, "transform", "");
                });

                if (start) {
                    self.setStyle(self.progress, "transition-duration", self.delay + "s");
                    self.setStyle(self.progress, "transform", "translateX(0)");
                }
            }

            if (start) {
                self.playing = setTimeout(self.next, self.delay * 1000);
            }
        }

        this.autohide = () => {
            //console.log("autohide");
            if (self.options_autohide) {
                self.hide_cooldown = Date.now() + 2950;
                if (!self.hide) {
                    self.addClass(self.widget, "menu");
                    self.schedule(3000);
                }
            }
        }

        this.schedule = (cooldown) => {
            //console.log("schedule", cooldown);
			
            self.hide = setTimeout(function () {
                const now = Date.now();
                if (now >= self.hide_cooldown) {
                    self.removeClass(self.widget, "menu");
                    self.hide = 0;
                }
                else {
                    self.schedule(self.hide_cooldown - now);
                }
            }, cooldown);
        }

        this.menu = (state) => {
            //console.log("menu");
            if (typeof state === "boolean") {
                self.hide = state ? self.hide : 0;
            }
            if (self.hide) {
                self.hide = clearTimeout(self.hide);
                self.removeClass(self.widget, "menu");
            }
            else {
                self.autohide();
            }
        }

        this.start = (e) => {
            //console.log("start");
            self.cancelEvent(e, true);
            self.is_down = true;
            self.dragged = false;
            let touches = e.touches;
            if (touches && (touches = touches[0])) {
                e = touches;
            }
            self.slidable =  (self.media_w * self.scale) <= self.viewport_w;
            self.startX = e.pageX;
            self.startY = e.pageY;
            self.toggleAnimation(self.panel);
        }

        this.end = (e) => {
            //console.log("end");
            self.cancelEvent(e);
            if (self.is_down) {
                if (!self.dragged) {
                    self.menu();
                }
                else {
                    if (self.slidable && self.dragged) {
                        const has_next = (self.x < -(self.viewport_w / 7)) && ((self.current_slide < self.slide_count) || self.options_infinite);
                        const has_prev = has_next || (self.x > (self.viewport_w / 7)) && ((self.current_slide > 1) || self.options_infinite);
                        if (has_next || has_prev) {
                            self.update_slider(self.current_slide - 1, true, self.x / self.viewport_w * 100);
                            (has_next && self.next()) ||
                                (has_prev && self.prev());
                        }
                        self.x = 0;
                        self.update_panel();
                    }
                    self.toggleAnimation(self.panel, true);
                }
                self.is_down = false;
            }
        }

        this.move = (e) => {
            //console.log("move");
            self.cancelEvent(e);
            if (self.is_down) {
                let touches = e.touches;
                if (touches && (touches = touches[0])) {
                    e = touches;
                }
                // handle x-axis in slide mode and in drag mode
                let diff = (self.media_w * self.scale - self.viewport_w) / 2;
                self.x -= self.startX - (self.startX = e.pageX);
                if (!self.slidable) {
                    if (self.x > diff) {
                        self.x = diff;
                    }
                    else if (self.x < -diff) {
                        self.x = -diff
                    }
                    // handle y-axis in drag mode
                    if ((self.media_h * self.scale) > self.viewport_h) {
                        diff = (self.media_h * self.scale - self.viewport_h) / 2;
                        self.y -= self.startY - (self.startY = e.pageY);
                        if (self.y > diff) {
                            self.y = diff;
                        }
                        else if (self.y < -diff) {
                            self.y = -diff;
                        }
                    }
                }
                self.dragged = true;
                self.update_panel(self.x, self.y);
            }
            else {
                self.autohide();
            }
        }

        this.fullscreen = (init) => {
            //console.log("fullscreen", init);
            const is_fullscreen = self.has_fullscreen();
            if ((typeof init !== "boolean") || (init !== !!is_fullscreen)) {
                if (is_fullscreen) {
                    document[self.prefix_exit]();
                    //removeClass(maximize, "on");
                }
                else {
                    self.widget[self.prefix_request]();
                    //addClass(maximize, "on");
                }
            }
        }

        this.theme = (theme) => {
            //console.log("theme", theme);
            if (typeof theme !== "string") {
                // toggle:
                theme = self.toggle_theme ? "" : self.options_theme || "white";
            }

            if (self.toggle_theme !== theme) {
                // set:
                self.toggle_theme && self.removeClass(self.widget, self.toggle_theme);
                theme && self.addClass(self.widget, theme);
                self.toggle_theme = theme;
            }
        }

        this.autofit = (init) => {

            //console.log("autofit", init);
            if (typeof init === "boolean") {
                self.toggle_autofit = !init;
            }

            self.toggle_autofit = (self.scale === 1) && !self.toggle_autofit;

            self.toggleClass(self.media, "autofit", self.toggle_autofit);
            self.setStyle(self.media, "transform", "");

            self.scale = 1;
            self.x = 0;
            self.y = 0;

            self.update_media_viewport();
            self.toggleAnimation(self.panel);
            self.update_panel();
            //autohide();
        }

        this.zoom_in = (e) => {
            //console.log("zoom_in");
            let value = self.scale / 0.65;
            if (value <= 50) {
                //console.log(toggle_autofit);
                self.disable_autoresizer();
                // if(options_fit){
                //     removeClass(media, options_fit);
                // }
                self.x /= 0.65;
                self.y /= 0.65;
                self.update_panel(self.x, self.y);
                self.zoom(value);
            }
            //e && autohide();
        }

        this.zoom_out = (e) => {
            //console.log("zoom_out");
            let value = self.scale * 0.65;
            self.disable_autoresizer();
            if (value >= 1) {
                if (value === 1) {
                    self.x = self.y = 0;
                    // if(options_fit){
                    //     addClass(media, options_fit);
                    // }
                }
                else {
                    self.x *= 0.65;
                    self.y *= 0.65;
                }
                self.update_panel(self.x, self.y);
                self.zoom(value);
            }
            //e && autohide();
        }

        this.zoom = (factor) => {
            //console.log("zoom", factor);
            self.scale = factor || 1;
            self.update_scroll();
        }

        this.info = () => {
            //console.log("info");
            self.footer_visible = !self.footer_visible;
            self.toggleVisibility(self.footer, self.footer_visible);

        }

        this.disable_autoresizer = () => {
            //console.log("disable_autoresizer");
            //update_media_dimension();
            if (self.toggle_autofit) {
                // removeClass(media, "autofit");
                // toggle_autofit = false;
                self.autofit();
            }
        }

        this.show_gallery = () => {

            //console.log("show_gallery");

            //history.pushState({ "spl": 1 }, "");
            //history.pushState({ "spl": 2 }, "");

            self.toggleAnimation(self.widget, true);
            self.addClass(self.body, "hide-scrollbars");
            self.addClass(self.widget, "show");

            self.toggle_listener(true);
            self.update_widget_viewport();
			//investigate resize_listener
            self.resize_listener();
            self.autohide();

            self.options_autoslide && self.play(true, true);
        }

        this.download = () => {
            //console.log("download", media);
            self.downloadImage(self.body, self.media);
        }

        this.close = (hashchange, nofade) => {
            //console.log("close", hashchange);
			if(self.hasClass(self.body,"relative") && !nofade){
				self.fullscreen(false);
				return;
			}
			
			if(!nofade){
				setTimeout(function () {
					self.body.removeChild(self.widget);
					self.panel = self.media = self.gallery = self.options = self.options_group = self.anchors = self.options_onshow = self.options_onchange = self.options_onclose = self.options_click = null;

				}, 200);
			}

            self.removeClass(self.body, "hide-scrollbars");
            self.removeClass(self.widget, "show");			

            self.fullscreen(false);
            self.toggle_listener();

            //history.go(hashchange === true ? -1 : -2);

            // teardown
            self.gallery_next && (self.media_next.src = "");
            self.playing && self.play();
            self.media && self.checkout(self.media);
            self.hide && (self.hide = clearTimeout(self.hide));
            self.toggle_theme && self.theme();
            self.options_class && self.removeClass(self.widget, self.options_class);
            self.options_onclose && self.options_onclose();
			
			if(nofade){				
				self.body.removeChild(self.widget);
				self.panel = self.media = self.gallery = self.options = self.options_group = self.anchors = self.options_onshow = self.options_onchange = self.options_onclose = self.options_click = null;
			}
        }

        this.checkout = (media) => {
            //console.log("checkout");
            if (media._root) {
                media._root.appendChild(media);
                media._root = null;
            }
            else {
                const parent = media.parentNode;
                parent && parent.removeChild(media);
                media = media.src = media.onerror = "";
            }
        }

        this.prev = (e) => {
            //console.log("prev");
            e && self.autohide();
            if (self.slide_count > 1) {
                if (self.current_slide > 1) {
                    return self.goto(self.current_slide - 1);
                }
                else if (self.options_infinite) {
                    self.update_slider(self.slide_count, true);
                    return self.goto(self.slide_count);
                }
            }
        }

        this.next = (e) => {
            //console.log("next");
            e && self.autohide();
            if (self.slide_count > 1) {
                if (self.current_slide < self.slide_count) {
                    return self.goto(self.current_slide + 1);
                }
                else if (self.options_infinite) {
                    self.update_slider(-1, true);
                    return self.goto(1);
                }
                else if (self.playing) {
                    self.play();
                }
            }
        }

        this.goto = (slide) => {
            //console.log("goto", slide);
            if (slide !== self.current_slide) {
                if (self.playing) {
                    clearTimeout(self.playing);
                    self.animate_bar();
                }
                else {
                    self.autohide();
                }

                //playing ? animate_bar() : autohide();
                const direction = slide > self.current_slide;
                self.current_slide = slide;
                self.setup_page(direction);
                //options_autoslide && play(true, true);
                return true;
            }
        }

        this.prepare = (direction) => {

            //console.log("prepare", direction);

            let anchor = self.anchors[self.current_slide - 1];

            self.apply_options(anchor);
            const speed = connection && connection["downlink"];
            let size = Math.max(self.viewport_h, self.viewport_w) * dpr;
            if (speed && ((speed * 1200) < size)) {
                size = speed * 1200;
            }
            let tmp;
            self.gallery = {
                media: self.options_media,
                src: self.parse_src(anchor, size, self.options, self.options_media),
                title: self.parse_option("title",
                    anchor["alt"] || anchor["title"] ||
                    // inherit title from a direct child only
                    ((tmp = anchor.firstElementChild) && (tmp["alt"] || tmp["title"]))
                )
            };

            self.gallery_next && (self.media_next.src = self.gallery_next = "");

            if (self.options_preload && direction) {
                if ((anchor = self.anchors[self.current_slide])) {
                    const options_next = anchor.dataset || anchor;
                    const next_media = options_next["media"];
                    if (!next_media || (next_media === "image")) {
                        self.gallery_next = self.parse_src(anchor, size, options_next, next_media);
                    }
                }
            }

            // apply controls

            for (let i = 0; i < self.controls.length; i++) {
                const option = self.controls[i];
                //console.log(option + ": ", options[option]);
                self.toggleDisplay(self.controls_dom[option], self.parse_option(option, self.controls_default[option]));
            }
        }

        this.setup_page = (direction) => {
            //console.log("setup_page", direction);
            self.x = 0;
            self.y = 0;
            self.scale = 1;

            if (self.media) {
                // Note: the onerror callback was removed when the image was fully loaded (also for video)
                if (self.media.onerror) {
                    self.checkout(self.media);
                }
                else {
                    let ref = self.media;
                    setTimeout(function () {
                        if (ref && (self.media !== ref)) {
                            self.checkout(ref);
                            ref = null;
                        }
                    }, 650);

                    // animate out the old image
                    self.prepare_animation();
                    self.update_panel();
                }
            }

            self.footer && self.toggleVisibility(self.footer, 0);

            self.prepare(direction);
            self.update_slider(self.current_slide - 1);
            self.removeClass(self.spinner, "error");
            self.init_slide(self.current_slide);
            self.toggleAnimation(self.panel);
            self.update_panel();

            const str_title = self.gallery.title;
            const str_description = self.parse_option("description");
            const str_button = self.parse_option("button");
            const has_content = str_title || str_description || str_button;

            if (has_content) {

                str_title && self.setText(self.title, str_title);
                str_description && self.setText(self.description, str_description);
                str_button && self.setText(self.button, str_button);

                self.toggleDisplay(self.title, str_title);
                self.toggleDisplay(self.description, str_description);
                self.toggleDisplay(self.button, str_button);

                self.setStyle(self.footer, "transform", self.options_autohide === "all" ? "" : "none");
            }

            self.options_autohide || self.addClass(self.widget, "menu");

            self.toggleVisibility(self.footer, self.footer_visible && has_content);
            self.toggleVisibility(self.page_prev, self.options_infinite || (self.current_slide > 1));
            self.toggleVisibility(self.page_next, self.options_infinite || (self.current_slide < self.slide_count));
            self.setText(self.page, self.slide_count > 1 ? self.current_slide + " / " + self.slide_count : "");

            self.options_onchange && self.options_onchange(self.current_slide, self.options);
        }
        /*
        const isWebkit = property => {
            if (typeof document.documentElement.style[property] === 'string') {
                return property;
            }
            // Capitalize the first letter
            property = property.charAt(0).toUpperCase() + property.slice(1);
            return `webkit${property}`;
        };
        */

        /* const extendDefaults = (defaults, properties) => {
            let property, propertyDeep;
            if (properties != undefined && properties != 'undefined') {
                for (property in properties) {
                    const propObj = properties[property];

                    if (typeof propObj === 'object') {
                        for (propertyDeep in propObj) {
                            defaults[property][propertyDeep] = propObj[propertyDeep];
                        }
                    } else {
                        defaults[property] = propObj;
                    }
                }
            }
            return defaults;
        }; */

        //this.init();
    };

    if (typeof module !== 'undefined' && typeof module.exports !== 'undefined') {
        module.exports = Spotlight;
    } else {
        window.Spotlight = Spotlight;
    }

})(window);