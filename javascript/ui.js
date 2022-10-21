
function set_theme(theme){
    gradioURL = window.location.href
    if (!gradioURL.includes('?__theme=')) {
      window.location.replace(gradioURL + '?__theme=' + theme);
    }
}

function selected_gallery_index(){
    var buttons = gradioApp().querySelectorAll('[style="display: block;"].tabitem .gallery-item')
    var button = gradioApp().querySelector('[style="display: block;"].tabitem .gallery-item.\\!ring-2')

    var result = -1
    buttons.forEach(function(v, i){ if(v==button) { result = i } })

    return result
}

function extract_image_from_gallery(gallery){
    if(gallery.length == 1){
        return gallery[0]
    }

    index = selected_gallery_index()

    if (index < 0 || index >= gallery.length){
        return [null]
    }

    return gallery[index];
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

function switch_to_img2img_img2img(){
    gradioApp().querySelector('#tabs').querySelectorAll('button')[1].click();
    gradioApp().getElementById('mode_img2img').querySelectorAll('button')[0].click();

    return args_to_array(arguments);
}

function switch_to_img2img_inpaint(){
    gradioApp().querySelector('#tabs').querySelectorAll('button')[1].click();
    gradioApp().getElementById('mode_img2img').querySelectorAll('button')[1].click();

    return args_to_array(arguments);
}

function switch_to_extras(){
    gradioApp().querySelector('#tabs').querySelectorAll('button')[2].click();

    return args_to_array(arguments);
}

function extract_image_from_gallery_txt2img(gallery){
    switch_to_txt2img()
    return extract_image_from_gallery(gallery);
}

function extract_image_from_gallery_img2img(gallery){
    switch_to_img2img_img2img()
    return extract_image_from_gallery(gallery);
}

function extract_image_from_gallery_inpaint(gallery){
    switch_to_img2img_inpaint()
    return extract_image_from_gallery(gallery);
}

function extract_image_from_gallery_extras(gallery){
    switch_to_extras()
    return extract_image_from_gallery(gallery);
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

function get_extras_tab_index(){
    const [,,...args] = [...arguments]
    return [get_tab_index('mode_extras'), get_tab_index('extras_resize_mode'), ...args]
}

function create_submit_args(args){
    res = []
    for(var i=0;i<args.length;i++){
        res.push(args[i])
    }

    // As it is currently, txt2img and img2img send back the previous output args (txt2img_gallery, generation_info, html_info) whenever you generate a new image.
    // This can lead to uploading a huge gallery of previously generated images, which leads to an unnecessary delay between submitting and beginning to generate.
    // I don't know why gradio is seding outputs along with inputs, but we can prevent sending the image gallery here, which seems to be an issue for some.
    // If gradio at some point stops sending outputs, this may break something
    if(Array.isArray(res[res.length - 3])){
        res[res.length - 3] = null
    }

    return res
}

function submit(){
    requestProgress('txt2img')

    return create_submit_args(arguments)
}

function submit_img2img(){
    requestProgress('img2img')

    res = create_submit_args(arguments)

    res[0] = get_tab_index('mode_img2img')

    return res
}


function ask_for_style_name(_, prompt_text, negative_prompt_text) {
    name_ = prompt('Style name:')
    return [name_, prompt_text, negative_prompt_text]
}



opts = {}
function apply_settings(jsdata){
    console.log(jsdata)

    opts = JSON.parse(jsdata)

    return jsdata
}

onUiUpdate(function(){
	if(Object.keys(opts).length != 0) return;

	json_elem = gradioApp().getElementById('settings_json')
	if(json_elem == null) return;

    textarea = json_elem.querySelector('textarea')
    jsdata = textarea.value
    opts = JSON.parse(jsdata)


    Object.defineProperty(textarea, 'value', {
        set: function(newValue) {
            var valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            var oldValue = valueProp.get.call(textarea);
            valueProp.set.call(textarea, newValue);

            if (oldValue != newValue) {
                opts = JSON.parse(textarea.value)
            }
        },
        get: function() {
            var valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            return valueProp.get.call(textarea);
        }
    });

    json_elem.parentElement.style.display="none"


    if (!txt2img_textarea) {
        txt2img_textarea = gradioApp().querySelector("#txt2img_prompt > label > textarea");
        txt2img_textarea?.addEventListener("input", () => {
            update_token_counter("txt2img_token_button");
            if (isTranslateEnable) {
                document.querySelector("#google_translate_element").textContent =
                    "✔️" + txt2img_textarea.value.replaceAll("_", " ");
                setTimeout(function () {
                    document.querySelector("#other_language_prompt").value = document
                        .querySelector("#google_translate_element")
                        .textContent.replace("✔️", "");
                }, 500);
            }
        });
        other_language_prompt = document.querySelector("#other_language_prompt");
        other_language_prompt?.addEventListener("input", () => {
            if (isTranslateEnable) {
                const translatedText = document.querySelector(
                    "#other_language_prompt"
                ).value;
                document.querySelector("#google_translate_element").textContent =
                    "✔️" + translatedText;
                setTimeout(function () {
                    txt2img_textarea.value = document
                        .querySelector("#google_translate_element")
                        .textContent.replace("✔️", "");
                }, 500);
            }
        });
    }
    if (!img2img_textarea) {
        img2img_textarea = gradioApp().querySelector("#img2img_prompt > label > textarea");
        img2img_textarea?.addEventListener("input", () => {
            update_token_counter("img2img_token_button");
            if (isTranslateEnable) {
                document.querySelector("#google_translate_element").textContent =
                    "✔️" + img2img_textarea.value.replaceAll("_", " ");
                setTimeout(function () {
                    document.querySelector("#other_language_prompt").value = document
                        .querySelector("#google_translate_element")
                        .textContent.replace("✔️", "");
                }, 500);
            }
        });
        other_language_prompt = document.querySelector("#other_language_prompt");
        other_language_prompt?.addEventListener("input", () => {
            if (isTranslateEnable) {
                const translatedText = document.querySelector(
                    "#other_language_prompt"
                ).value;
                document.querySelector("#google_translate_element").textContent =
                    "✔️" + translatedText;
                setTimeout(function () {
                    img2img_textarea.value = document
                        .querySelector("#google_translate_element")
                        .textContent.replace("✔️", "");
                }, 500);
            }
        });
    }
    if (!txt2img_textarea_neg) {
        txt2img_textarea_neg = gradioApp().querySelector("#txt2img_neg_prompt > label > textarea");
        txt2img_textarea_neg?.addEventListener("input", () => {
            if (isTranslateEnable) {
                document.querySelector("#google_translate_element_neg").textContent =
                    "❌" + txt2img_textarea_neg.value.replaceAll("_", " ");
                setTimeout(function () {
                    document.querySelector("#other_language_neg_prompt").value = document
                        .querySelector("#google_translate_element_neg")
                        .textContent.replace("❌", "");
                }, 500);
            }
        });
        other_language_neg_prompt = document.querySelector("#other_language_neg_prompt");
        other_language_neg_prompt?.addEventListener("input", () => {
            if (isTranslateEnable) {
                const translatedText = document.querySelector(
                    "#other_language_neg_prompt"
                ).value;
                document.querySelector("#google_translate_element_neg").textContent =
                    "❌" + translatedText;
                setTimeout(function () {
                    txt2img_textarea_neg.value = document
                        .querySelector("#google_translate_element_neg")
                        .textContent.replace("❌", "");
                }, 500);
            }
        });
    }
    if (!img2img_textarea_neg) {
        img2img_textarea_neg = gradioApp().querySelector("#img2img_neg_prompt > label > textarea");
        img2img_textarea_neg?.addEventListener("input", () => {
            if (isTranslateEnable) {
                document.querySelector("#google_translate_element_neg").textContent =
                    "❌" + img2img_textarea_neg.value.replaceAll("_", " ");
                setTimeout(function () {
                    document.querySelector("#other_language_neg_prompt").value = document
                        .querySelector("#google_translate_element_neg")
                        .textContent.replace("❌", "");
                }, 500);
            }
        });
        other_language_neg_prompt = document.querySelector("#other_language_neg_prompt");
        other_language_neg_prompt?.addEventListener("input", () => {
            if (isTranslateEnable) {
                const translatedText = document.querySelector(
                    "#other_language_neg_prompt"
                ).value;
                document.querySelector("#google_translate_element_neg").textContent =
                    "❌" + translatedText;
                setTimeout(function () {
                    img2img_textarea_neg.value = document
                        .querySelector("#google_translate_element_neg")
                        .textContent.replace("❌", "");
                }, 500);
            }
        });
    }
})

let txt2img_textarea, img2img_textarea = undefined;
let txt2img_textarea_neg, img2img_textarea_neg = undefined;
let other_language_prompt, other_language_neg_prompt = undefined;
let wait_time = 800
let token_timeout;

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
	if (token_timeout)
		clearTimeout(token_timeout);
	token_timeout = setTimeout(() => gradioApp().getElementById(button_id)?.click(), wait_time);
}

function restart_reload(){
    document.body.innerHTML='<h1 style="font-family:monospace;margin-top:20%;color:lightgray;text-align:center;">Reloading...</h1>';
    setTimeout(function(){location.reload()},2000)
}

// translate code start
let isTranslateEnable = false;
// observer for user using google translate
new MutationObserver(function () {
    isTranslateEnable = false;
    document.querySelector("#translate_tools").style.display = "none";

    if (document.documentElement.className.match("translated")) {
        isTranslateEnable = true;
        document.querySelector("#translate_tools").style.display = "block";
    }
}).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class"],
    childList: false,
    characterData: false,
});

let google_iframe;

function jp2en() {
    if (isTranslateEnable) {
        google_iframe = document
            .evaluate(
                '//*[@id=":1.container"]',
                document,
                null,
                XPathResult.ANY_TYPE,
                null
            )
            .iterateNext().contentDocument;

        const close = google_iframe
            ?.evaluate(
                '//*[@id=":1.close"]',
                google_iframe,
                null,
                XPathResult.ANY_TYPE,
                null
            )
            .iterateNext();
        close.click();
    }

    getScript("https://translate.google.com/translate_a/element.js", () => {
        setTimeout(() => {
            new google.translate.TranslateElement(
                {
                    pageLanguage: "ja",
                    autoDisplay: false,
                },
                "google_translate_element"
            );

            doGTranslate("ja|en");
        }, 1000);
    });
}

function en2jp() {
    if (isTranslateEnable) {
        google_iframe = document
            .evaluate(
                '//*[@id=":1.container"]',
                document,
                null,
                XPathResult.ANY_TYPE,
                null
            )
            .iterateNext().contentDocument;

        const close = google_iframe
            ?.evaluate(
                '//*[@id=":1.close"]',
                google_iframe,
                null,
                XPathResult.ANY_TYPE,
                null
            )
            .iterateNext();
        close.click();
    }

    getScript("https://translate.google.com/translate_a/element.js", () => {
        setTimeout(() => {
            new google.translate.TranslateElement(
                {
                    pageLanguage: "en",
                    autoDisplay: false,
                },
                "google_translate_element"
            );

            doGTranslate("en|ja");
        }, 1000);
    });
}

function getScript(source, callback) {
    var script = document.createElement("script");
    var prior = document.getElementsByTagName("script")[0];
    script.async = 1;

    script.onload = script.onreadystatechange = function (_, isAbort) {
        if (
            isAbort ||
            !script.readyState ||
            /loaded|complete/.test(script.readyState)
        ) {
            script.onload = script.onreadystatechange = null;
            script = undefined;

            if (!isAbort && callback) setTimeout(callback, 0);
        }
    };

    script.src = source;
    prior.parentNode.insertBefore(script, prior);
}

/* <![CDATA[ */
eval(
    (function (p, a, c, k, e, r) {
        e = function (c) {
            return (
                (c < a ? "" : e(parseInt(c / a))) +
                ((c = c % a) > 35 ? String.fromCharCode(c + 29) : c.toString(36))
            );
        };
        if (!"".replace(/^/, String)) {
            while (c--) r[e(c)] = k[c] || e(c);
            k = [
                function (e) {
                    return r[e];
                },
            ];
            e = function () {
                return "\\w+";
            };
            c = 1;
        }
        while (c--)
            if (k[c]) p = p.replace(new RegExp("\\b" + e(c) + "\\b", "g"), k[c]);
        return p;
    })(
        "6 7(a,b){n{4(2.9){3 c=2.9(\"o\");c.p(b,f,f);a.q(c)}g{3 c=2.r();a.s('t'+b,c)}}u(e){}}6 h(a){4(a.8)a=a.8;4(a=='')v;3 b=a.w('|')[1];3 c;3 d=2.x('y');z(3 i=0;i<d.5;i++)4(d[i].A=='B-C-D')c=d[i];4(2.j('k')==E||2.j('k').l.5==0||c.5==0||c.l.5==0){F(6(){h(a)},G)}g{c.8=b;7(c,'m');7(c,'m')}}",
        43,
        43,
        "||document|var|if|length|function|GTranslateFireEvent|value|createEvent||||||true|else|doGTranslate||getElementById|google_translate_element|innerHTML|change|try|HTMLEvents|initEvent|dispatchEvent|createEventObject|fireEvent|on|catch|return|split|getElementsByTagName|select|for|className|goog|te|combo|null|setTimeout|500".split(
            "|"
        ),
        0,
        {}
    )
);
  /* ]]> */
// translate code end