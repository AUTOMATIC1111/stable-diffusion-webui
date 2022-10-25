let rtt_translate_div, rtt_translate_reversed_div;
let rtt_gr_txt2image_textarea, rtt_gr_img2img_textarea;
let rtt_gr_txt2image_textarea_neg, rtt_gr_img2img_textarea_neg;
let rtt_reversed_prompt_textarea, rtt_reversed_prompt_textarea_neg;
let rtt_translate_from, rtt_translate_to;
let rtt_is_translate_enable = false;
let rtt_is_translate_reversed = false;

// observer for user using google translate
new MutationObserver(function () {
    rtt_is_translate_enable = false;
    if (rtt_translate_div)
        rtt_translate_div.style.display = "none";
    if (rtt_translate_reversed_div)
        rtt_translate_reversed_div.style.display = "none";

    if (document.documentElement.className.match("translated")) {
        rtt_is_translate_enable = true;
        if (rtt_translate_div)
            rtt_translate_div.style.display = "block";
        if (rtt_translate_reversed_div && rtt_is_translate_reversed)
            rtt_translate_reversed_div.style.display = "block";
    }
}).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["class"],
    childList: false,
    characterData: false,
});

onUiUpdate(function () {
    if (!rtt_translate_reversed_div) {
        rtt_translate_reversed_div = document.createElement("div");
        rtt_translate_reversed_div.innerHTML = '<textarea  id="rtt_reversed_prompt_textarea" placeholder="In put Prompt here to translate" rows="2" style="overflow-y: scroll; height: 63px; width:90%" spellcheck="false"></textarea><br /><textarea id="rtt_reversed_prompt_textarea_neg" rows="2" style="overflow-y: scroll; height: 63px; width:90%" spellcheck="false"></textarea>';
        rtt_translate_reversed_div.style.display = "none";
        document.body.insertBefore(rtt_translate_reversed_div, document.body.firstChild);
    }

    if (!rtt_translate_div) {
        rtt_translate_div = document.createElement("div");
        rtt_translate_div.innerHTML = '<span id="google_translate_element"></span><br /><span id="google_translate_element_neg"></span><br />';
        rtt_translate_div.style.display = "none";
        document.body.insertBefore(rtt_translate_div, document.body.firstChild);
    }

    if (!rtt_translate_from) {
        rtt_translate_from = gradioApp().querySelector("#rtt_translate_from");
    }

    if (!rtt_translate_to) {
        rtt_translate_to = gradioApp().querySelector("#rtt_translate_to");
    }

    if (!rtt_reversed_prompt_textarea) {
        rtt_reversed_prompt_textarea = document.querySelector("#rtt_reversed_prompt_textarea");
        rtt_reversed_prompt_textarea?.addEventListener("input", () => {
            if (rtt_is_translate_enable) {
                const translatedText = rtt_reversed_prompt_textarea.value;
                document.querySelector("#google_translate_element").textContent =
                    "✔️" + translatedText;
                setTimeout(function () {
                    rtt_gr_txt2image_textarea.value = document
                        .querySelector("#google_translate_element")
                        .textContent.replace("✔️", "");
                }, 500);
            }
        });
        rtt_reversed_prompt_textarea?.addEventListener("input", () => {
            if (rtt_is_translate_enable) {
                const translatedText = rtt_reversed_prompt_textarea.value;
                document.querySelector("#google_translate_element").textContent =
                    "✔️" + translatedText;
                setTimeout(function () {
                    rtt_gr_img2img_textarea.value = document
                        .querySelector("#google_translate_element")
                        .textContent.replace("✔️", "");
                }, 500);
            }
        });
    }

    if (!rtt_reversed_prompt_textarea_neg) {
        rtt_reversed_prompt_textarea_neg = document.querySelector("#rtt_reversed_prompt_textarea_neg");
        rtt_reversed_prompt_textarea_neg?.addEventListener("input", () => {
            if (rtt_is_translate_enable) {
                const translatedText = rtt_reversed_prompt_textarea_neg.value;
                document.querySelector("#google_translate_element_neg").textContent =
                    "❌" + translatedText;
                setTimeout(function () {
                    rtt_gr_txt2image_textarea_neg.value = document
                        .querySelector("#google_translate_element_neg")
                        .textContent.replace("❌", "");
                }, 500);
            }
        });
        rtt_reversed_prompt_textarea_neg?.addEventListener("input", () => {
            if (rtt_is_translate_enable) {
                const translatedText = document.rtt_reversed_prompt_textarea_neg.value;
                document.querySelector("#google_translate_element_neg").textContent =
                    "❌" + translatedText;
                setTimeout(function () {
                    rtt_gr_img2img_textarea_neg.value = document
                        .querySelector("#google_translate_element_neg")
                        .textContent.replace("❌", "");
                }, 500);
            }
        });
    }

    if (!rtt_gr_txt2image_textarea) {
        rtt_gr_txt2image_textarea = gradioApp().querySelector("#txt2img_prompt > label > textarea");
        rtt_gr_txt2image_textarea?.addEventListener("input", () => {
            if (rtt_is_translate_enable) {
                document.querySelector("#google_translate_element").textContent =
                    "✔️" + rtt_gr_txt2image_textarea.value.replaceAll("_", " ");
                setTimeout(function () {
                    document.querySelector("#rtt_reversed_prompt_textarea").value = document
                        .querySelector("#google_translate_element")
                        .textContent.replace("✔️", "");
                }, 500);
            }
        });
    }
    if (!rtt_gr_img2img_textarea) {
        rtt_gr_img2img_textarea = gradioApp().querySelector("#img2img_prompt > label > textarea");
        rtt_gr_img2img_textarea?.addEventListener("input", () => {
            if (rtt_is_translate_enable) {
                document.querySelector("#google_translate_element").textContent =
                    "✔️" + rtt_gr_img2img_textarea.value.replaceAll("_", " ");
                setTimeout(function () {
                    document.querySelector("#rtt_reversed_prompt_textarea").value = document
                        .querySelector("#google_translate_element")
                        .textContent.replace("✔️", "");
                }, 500);
            }
        });
    }
    if (!rtt_gr_txt2image_textarea_neg) {
        rtt_gr_txt2image_textarea_neg = gradioApp().querySelector("#txt2img_neg_prompt > label > textarea");
        rtt_gr_txt2image_textarea_neg?.addEventListener("input", () => {
            if (rtt_is_translate_enable) {
                document.querySelector("#google_translate_element_neg").textContent =
                    "❌" + rtt_gr_txt2image_textarea_neg.value.replaceAll("_", " ");
                setTimeout(function () {
                    document.querySelector("#rtt_reversed_prompt_textarea_neg").value = document
                        .querySelector("#google_translate_element_neg")
                        .textContent.replace("❌", "");
                }, 500);
            }
        });
    }
    if (!rtt_gr_img2img_textarea_neg) {
        rtt_gr_img2img_textarea_neg = gradioApp().querySelector("#img2img_neg_prompt > label > textarea");
        rtt_gr_img2img_textarea_neg?.addEventListener("input", () => {
            if (rtt_is_translate_enable) {
                document.querySelector("#google_translate_element_neg").textContent =
                    "❌" + rtt_gr_img2img_textarea_neg.value.replaceAll("_", " ");
                setTimeout(function () {
                    document.querySelector("#rtt_reversed_prompt_textarea_neg").value = document
                        .querySelector("#google_translate_element_neg")
                        .textContent.replace("❌", "");
                }, 500);
            }
        });
    }
});

function rtt_translate() {
    rtt_is_translate_reversed = false;
    if (rtt_is_translate_enable) {
        const rtt_google_iframe = document.evaluate(
            '//*[@id=":1.container"]',
            document,
            null,
            XPathResult.ANY_TYPE,
            null
        ).iterateNext().contentDocument;

        rtt_google_iframe?.evaluate(
            '//*[@id=":1.close"]',
            rtt_google_iframe,
            null,
            XPathResult.ANY_TYPE,
            null
        ).iterateNext().click();
    }

    getScript("https://translate.google.com/translate_a/element.js", () => {
        setTimeout(() => {
            const temp = new google.translate.TranslateElement(
                {
                    pageLanguage: rtt_translate_from.value,
                    autoDisplay: false,
                },
                "google_translate_element"
            );

            doGTranslate(rtt_translate_from.value + "|" + rtt_translate_to.value);
        }, 1000);
    });
}

function rtt_translate_reversed() {
    rtt_is_translate_reversed = true;
    if (rtt_is_translate_enable) {
        const rtt_google_iframe = document.evaluate(
            '//*[@id=":1.container"]',
            document,
            null,
            XPathResult.ANY_TYPE,
            null
        ).iterateNext().contentDocument;

        rtt_google_iframe?.evaluate(
            '//*[@id=":1.close"]',
            rtt_google_iframe,
            null,
            XPathResult.ANY_TYPE,
            null
        ).iterateNext().click();
    }

    getScript("https://translate.google.com/translate_a/element.js", () => {
        setTimeout(() => {
            const temp = new google.translate.TranslateElement(
                {
                    pageLanguage: rtt_translate_to.value,
                    autoDisplay: false,
                },
                "google_translate_element"
            );

            doGTranslate(rtt_translate_to.value + "|" + rtt_translate_from.value);
        }, 1000);
    });
}

function getScript(source, callback) {
    let script = document.createElement("script");
    let prior = document.getElementsByTagName("script")[0];
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