function hexToRgb(color) {
  var hex = color[0] === "#" ? color.slice(1) : color;
  var c; // expand the short hex by doubling each character, fc0 -> ffcc00

  if (hex.length !== 6) {
    hex = (function() {
      var result = [];

      for (
        var _i = 0, _Array$from = Array.from(hex);
        _i < _Array$from.length;
        _i++
      ) {
        c = _Array$from[_i];
        result.push("".concat(c).concat(c));
      }

      return result;
    })().join("");
  }

  var colorStr = hex.match(/#?(.{2})(.{2})(.{2})/).slice(1);
  var rgb = colorStr.map(function(col) {
    return parseInt(col, 16);
  });
  rgb.push(1);
  return rgb;
}

function rgbToHsl(rgb) {
  var r = rgb[0] / 255;
  var g = rgb[1] / 255;
  var b = rgb[2] / 255;
  var max = Math.max(r, g, b);
  var min = Math.min(r, g, b);
  var diff = max - min;
  var add = max + min;
  var hue =
    min === max
      ? 0
      : r === max
        ? (60 * (g - b) / diff + 360) % 360
        : g === max ? 60 * (b - r) / diff + 120 : 60 * (r - g) / diff + 240;
  var lum = 0.5 * add;
  var sat =
    lum === 0 ? 0 : lum === 1 ? 1 : lum <= 0.5 ? diff / add : diff / (2 - add);
  var h = Math.round(hue);
  var s = Math.round(sat * 100);
  var l = Math.round(lum * 100);
  var a = rgb[3] || 1;
  return [h, s, l, a];
}

function hexToHsl(color) {
  var rgb = hexToRgb(color);
  var hsl = rgbToHsl(rgb);
  return "hsl(" + hsl[0] + "deg " + hsl[1] + "% " + hsl[2] + "%)";
}

function hslToHex(h, s, l) {
  l /= 100;
  var a = s * Math.min(l, 1 - l) / 100;

  var f = function f(n) {
    var k = (n + h / 30) % 12;
    var color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
    return Math.round(255 * Math.max(0, Math.min(color, 1)))
      .toString(16)
      .padStart(2, "0"); // convert to Hex and prefix "0" if needed
  };

  return "#".concat(f(0)).concat(f(8)).concat(f(4));
}

function hsl2rgb(h, s, l) {
  var a = s * Math.min(l, 1 - l);

  var f = function f(n) {
    var k =
      arguments.length > 1 && arguments[1] !== undefined
        ? arguments[1]
        : (n + h / 30) % 12;
    return l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
  };

  return [f(0), f(8), f(4)];
}

function invertColor(hex) {
  if (hex.indexOf("#") === 0) {
    hex = hex.slice(1);
  } // convert 3-digit hex to 6-digits.

  if (hex.length === 3) {
    hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
  }

  if (hex.length !== 6) {
    throw new Error("Invalid HEX color.");
  } // invert color components

  var r = (255 - parseInt(hex.slice(0, 2), 16)).toString(16),
    g = (255 - parseInt(hex.slice(2, 4), 16)).toString(16),
    b = (255 - parseInt(hex.slice(4, 6), 16)).toString(16); // pad each with zeros and return

  return "#" + padZero(r) + padZero(g) + padZero(b);
}

function padZero(str, len) {
  len = len || 2;
  var zeros = new Array(len).join("0");
  return (zeros + str).slice(-len);
}

function getValsWrappedIn(str, c1, c2) {
  var rg = new RegExp("(?<=\\" + c1 + ")(.*?)(?=\\" + c2 + ")", "g");
  return str.match(rg);
}

var styleobj = {};
var hslobj = {};
var isColorsInv;

var toHSLArray = function toHSLArray(hslStr) {
  return hslStr.match(/\d+/g).map(Number);
};

function offsetColorsHSV(ohsl) {
  var inner_styles = "";

  for (var key in styleobj) {
    var keyVal = styleobj[key];

    if (keyVal.indexOf("#") != -1 || keyVal.indexOf("hsl") != -1) {
      var colcomp = document.body.querySelector("#" + key + " input");

      if (colcomp) {
        var hsl = void 0;

        if (keyVal.indexOf("#") != -1) {
          keyVal = keyVal.replace(/\s+/g, ""); //inv ? keyVal = invertColor(keyVal) : 0;

          if (isColorsInv) {
            keyVal = invertColor(keyVal);
            styleobj[key] = keyVal;
          }

          hsl = rgbToHsl(hexToRgb(keyVal));
        } else {
          if (isColorsInv) {
            var c = toHSLArray(keyVal);

            var _hex = hslToHex(c[0], c[1], c[2]);

            keyVal = invertColor(_hex);
            styleobj[key] = keyVal;
            hsl = rgbToHsl(hexToRgb(keyVal));
          } else {
            hsl = toHSLArray(keyVal);
          }
        }

        var h = (parseInt(hsl[0]) + parseInt(ohsl[0])) % 360;
        var s = parseInt(hsl[1]) + parseInt(ohsl[1]);
        var l = parseInt(hsl[2]) + parseInt(ohsl[2]);
        var hex = hslToHex(
          h,
          Math.min(Math.max(s, 0), 100),
          Math.min(Math.max(l, 0), 100)
        );
        colcomp.value = hex;
        hslobj[key] = "hsl(" + h + "deg " + s + "% " + l + "%)";
        inner_styles += key + ":" + hslobj[key] + ";";
      }
    } else {
      inner_styles += key + ":" + styleobj[key] + ";";
    }
  }

  isColorsInv = false;
  var preview_styles = document.body.querySelector("#preview-styles");
  preview_styles.innerHTML = ":root {" + inner_styles + "}";
  preview_styles.innerHTML +=
    "@media only screen and (max-width: 860px) {:root{--ae-outside-gap-size: var(--ae-mobile-outside-gap-size);--ae-inside-padding-size: var(--ae-mobile-inside-padding-size);}}";
  var vars_textarea = document.body.querySelector("#theme_vars textarea");
  vars_textarea.value = inner_styles;
  window.updateInput(vars_textarea);
}

function updateTheme(vars) {
  var inner_styles = "";

  var _loop = function _loop(i) {
    var key = vars[i].split(":");
    var id = key[0].replace(/\s+/g, "");
    var val = key[1].trim();
    styleobj[id] = val;
    inner_styles += id + ":" + val + ";";
    document.body.querySelectorAll("#" + id + " input").forEach(function(elem) {
      if (val.indexOf("hsl") != -1) {
        var hsl = toHSLArray(val);
        var hex = hslToHex(hsl[0], hsl[1], hsl[2]);
        elem.value = hex;
      } else {
        elem.value = val.split("px")[0];
      }
    });
  };

  for (var i = 0; i < vars.length - 1; i++) {
    _loop(i);
  }

  var preview_styles = document.body.querySelector("#preview-styles");

  if (preview_styles) {
    preview_styles.innerHTML = ":root {" + inner_styles + "}";
    preview_styles.innerHTML +=
      "@media only screen and (max-width: 860px) {:root{--ae-outside-gap-size: var(--ae-mobile-outside-gap-size);--ae-inside-padding-size: var(--ae-mobile-inside-padding-size);}}";
  } else {
    var r = document.body;
    var style = document.createElement("style");
    style.id = "preview-styles";
    style.innerHTML = ":root {" + inner_styles + "}";
    style.innerHTML +=
      "@media only screen and (max-width: 860px) {:root{--ae-outside-gap-size: var(--ae-mobile-outside-gap-size);--ae-inside-padding-size: var(--ae-mobile-inside-padding-size);}}";
    r.appendChild(style);
  }

  var vars_textarea = document.body.querySelector("#theme_vars textarea");
  var css_textarea = document.body.querySelector("#theme_css textarea");
  vars_textarea.value = inner_styles;
  css_textarea.value = css_textarea.value; //console.log(Object);

  window.updateInput(vars_textarea);
  window.updateInput(css_textarea);
}

function applyTheme() {
  console.log("apply");
}

function initTheme(styles) {
  var css_styles = styles.split("/*BREAKPOINT_CSS_CONTENT*/");
  var init_css_vars = css_styles[0].split("}")[0].split("{")[1];
  init_css_vars = init_css_vars.replace(/\n|\r/g, "");
  var init_vars = init_css_vars.split(";");
  var vars = init_vars; //console.log(vars);

  var vars_textarea = document.body.querySelector("#theme_vars textarea");
  var css_textarea = document.body.querySelector("#theme_css textarea");
  vars_textarea.value = init_css_vars;
  var additional_styles = css_styles[1] !== undefined ? css_styles[1] : "";
  css_textarea.value =
    "/*BREAKPOINT_CSS_CONTENT*/" +
    additional_styles +
    "/*BREAKPOINT_CSS_CONTENT*/";
  updateTheme(vars);
  var preview_styles = document.body.querySelector("#preview-styles");

  if (!preview_styles) {
    var style = document.createElement("style");
    style.id = "preview-styles";
    style.innerHTML = styles;
    document.body.appendChild(style);
  }

  var intervalChange;
  document.body
    .querySelectorAll("#ui_theme_settings input")
    .forEach(function(elem) {
      elem.addEventListener("input", function(e) {
        var celem = e.currentTarget;
        var val = e.currentTarget.value;
        var curr_val;

        switch (e.currentTarget.type) {
          case "range":
            celem = celem.parentElement;
            val = e.currentTarget.value + "px";
            break;

          case "color":
            celem = celem.parentElement.parentElement;
            val = e.currentTarget.value;
            break;

          case "number":
            celem = celem.parentElement.parentElement.parentElement;
            val = e.currentTarget.value + "px";
            break;
        }

        if (celem.id === "--ae-input-slider-height") {
          val = e.currentTarget.value;
        }

        styleobj[celem.id] = val; //console.log(styleobj);

        if (intervalChange != null) clearInterval(intervalChange);
        intervalChange = setTimeout(function() {
          var inner_styles = "";

          for (var key in styleobj) {
            inner_styles += key + ":" + styleobj[key] + ";";
          }

          vars = inner_styles.split(";");
          preview_styles.innerHTML = ":root {" + inner_styles + "}";
          preview_styles.innerHTML +=
            "@media only screen and (max-width: 860px) {:root{--ae-outside-gap-size: var(--ae-mobile-outside-gap-size);--ae-inside-padding-size: var(--ae-mobile-inside-padding-size);}}";
          vars_textarea.value = inner_styles;
          window.updateInput(vars_textarea);

          offsetColorsHSV(hsloffset);
        }, 1000);
      });
    });
  var reset_btn = document.getElementById("theme_reset_btn");
  reset_btn.addEventListener("click", function(e) {
    e.preventDefault();
    e.stopPropagation();
    document.body
      .querySelectorAll("#ui_theme_hsv input")
      .forEach(function(elem) {
        elem.value = 0;
      });
    hsloffset = [0, 0, 0];
    updateTheme(init_vars);
  });
  var intervalCheck;

  function dropDownOnChange() {
    if (init_css_vars !== vars_textarea.value) {
      clearInterval(intervalCheck);
      init_css_vars = vars_textarea.value.replace(/\n|\r/g, "");
      vars_textarea.value = init_css_vars;
      init_vars = init_css_vars.split(";");
      vars = init_vars;
      updateTheme(vars);
    }
  }

  var drop_down = document.body.querySelector("#themes_drop_down input");
  drop_down.addEventListener("click", function(e) {
    if (intervalCheck !== null) clearInterval(intervalCheck);
    vars_textarea.value = init_css_vars;
    intervalCheck = setInterval(dropDownOnChange, 500);
  });
  var hsloffset = [0, 0, 0];
  var hue = document.body
    .querySelectorAll("#theme_hue input")
    .forEach(function(elem) {
      elem.addEventListener("change", function(e) {
        e.preventDefault();
        e.stopPropagation();
        hsloffset[0] = e.currentTarget.value;
        offsetColorsHSV(hsloffset);
      });
    });
  var sat = document.body
    .querySelectorAll("#theme_sat input")
    .forEach(function(elem) {
      elem.addEventListener("change", function(e) {
        e.preventDefault();
        e.stopPropagation();
        hsloffset[1] = e.currentTarget.value;
        offsetColorsHSV(hsloffset);
      });
    });
  var brt = document.body
    .querySelectorAll("#theme_brt input")
    .forEach(function(elem) {
      elem.addEventListener("change", function(e) {
        e.preventDefault();
        e.stopPropagation();
        hsloffset[2] = e.currentTarget.value;
        offsetColorsHSV(hsloffset);
      });
    });
  var inv_btn = document.getElementById("theme_invert_btn");
  inv_btn.addEventListener("click", function(e) {
    e.preventDefault();
    e.stopPropagation();
    isColorsInv = !isColorsInv;
    offsetColorsHSV(hsloffset);
  });
}

function observeGradioApp() {
  var observer = new MutationObserver(function() {
    var css = document.querySelector('[rel="stylesheet"][href*="user"]');
    var block = gradioApp().getElementById("tab_ui_theme");

    if (css && block) {
      observer.disconnect();
      setTimeout(function() {
        var rootRules = Array.from(css.sheet.cssRules).filter(function(
          cssRule
        ) {
          return (
            cssRule instanceof CSSStyleRule && cssRule.selectorText === ":root"
          );
        });
        var rootCssText = rootRules[0].cssText; //console.log("Loading theme vars", rootCssText);

        initTheme(rootCssText);
      }, "500");
    }
  });
  observer.observe(gradioApp(), {
    childList: true,
    subtree: true
  });
}

document.addEventListener("DOMContentLoaded", function() {
  observeGradioApp();
});
