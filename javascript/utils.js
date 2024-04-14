/** Collators used for sorting. */
const INT_COLLATOR = new Intl.Collator([], {numeric: true});
const STR_COLLATOR = new Intl.Collator("en", {numeric: true, sensitivity: "base"});

/** Helper functions for checking types and simplifying logging/error handling. */
function isNumber(x) {
    return typeof x === "number" && isFinite(x);
}
function isNumberLogError(x) {
    if (isNumber(x)) {
        return true;
    }
    console.error(`expected number, got: ${typeof x}`);
    return false;
}
function isNumberThrowError(x) {
    if (isNumber(x)) {
        return;
    }
    throw new Error(`expected number, got: ${typeof x}`);
}

function isString(x) {
    return typeof x === "string" || x instanceof String;
}
function isStringLogError(x) {
    if (isString(x)) {
        return true;
    }
    console.error(`expected string, got: ${typeof x}`);
    return false;
}
function isStringThrowError(x) {
    if (isString(x)) {
        return;
    }
    throw new Error(`expected string, got: ${typeof x}`);
}

function isNull(x) {
    return x === null;
}
function isUndefined(x) {
    return typeof x === "undefined" || x === undefined;
}
// checks both null and undefined for simplicity sake.
function isNullOrUndefined(x) {
    return isNull(x) || isUndefined(x);
}
function isNullOrUndefinedLogError(x) {
    if (isNullOrUndefined(x)) {
        console.error("Variable is null/undefined.");
        return true;
    }
    return false;
}
function isNullOrUndefinedThrowError(x) {
    if (!isNullOrUndefined(x)) {
        return;
    }
    throw new Error("Variable is null/undefined.");
}

function isElement(x) {
    return x instanceof Element;
}
function isElementLogError(x) {
    if (isElement(x)) {
        return true;
    }
    console.error(`expected element type, got: ${typeof x}`);
    return false;
}
function isElementThrowError(x) {
    if (isElement(x)) {
        return;
    }
    throw new Error(`expected element type, got: ${typeof x}`);
}

function isFunction(x) {
    return typeof x === "function";
}
function isFunctionLogError(x) {
    if (isFunction(x)) {
        return true;
    }
    console.error(`expected function type, got: ${typeof x}`);
    return false;
}
function isFunctionThrowError(x) {
    if (isFunction(x)) {
        return;
    }
    throw new Error(`expected function type, got: ${typeof x}`);
}

function isObject(x) {
    return typeof x === "object" && !Array.isArray(x);
}
function isObjectLogError(x) {
    if (isObject(x)) {
        return true;
    }
    console.error(`expected object type, got: ${typeof x}`);
    return false;
}
function isObjectThrowError(x) {
    if (isObject(x)) {
        return;
    }
    throw new Error(`expected object type, got: ${typeof x}`);
}

function keyExists(obj, k) {
    return isObject(obj) && isString(k) && k in obj;
}
function keyExistsLogError(obj, k) {
    if (keyExists(obj, k)) {
        return true;
    }
    console.error(`key does not exist in object: ${k}`);
    return false;
}
function keyExistsThrowError(obj, k) {
    if (keyExists(obj, k)) {
        return;
    }
    throw new Error(`key does not exist in object: ${k}`);
}

function getValue(obj, k) {
    /** Returns value of object for given key if it exists, otherwise returns null. */
    if (keyExists(obj, k)) {
        return obj[k];
    }
    return null;
}
function getValueLogError(obj, k) {
    if (keyExistsLogError(obj, k)) {
        return obj[k];
    }
    return null;
}
function getValueThrowError(obj, k) {
    keyExistsThrowError(obj, k);
    return obj[k];
}

function getElementByIdLogError(selector) {
    const elem = gradioApp().getElementById(selector);
    isElementLogError(elem);
    return elem;
}
function getElementByIdThrowError(selector) {
    const elem = gradioApp().getElementById(selector);
    isElementThrowError(elem);
    return elem;
}

function querySelectorLogError(selector) {
    const elem = gradioApp().querySelector(selector);
    isElementLogError(elem);
    return elem;
}
function querySelectorThrowError(selector) {
    const elem = gradioApp().querySelector(selector);
    isElementThrowError(elem);
    return elem;
}

/** Functions for getting dimensions of elements. */
function getStyle(elem) {
    return window.getComputedStyle ? window.getComputedStyle(elem) : elem.currentStyle;
}

function getComputedProperty(elem, prop) {
    return getStyle(elem)[prop];
}

function getComputedPropertyDims(elem, prop) {
    /** Returns the top/left/bottom/right float dimensions of an element for the specified property. */
    const style = getStyle(elem);
    return {
        top: parseFloat(style.getPropertyValue(`${prop}-top`)),
        left: parseFloat(style.getPropertyValue(`${prop}-left`)),
        bottom: parseFloat(style.getPropertyValue(`${prop}-bottom`)),
        right: parseFloat(style.getPropertyValue(`${prop}-right`)),
    };
}

function getComputedMarginDims(elem) {
    /** Returns the width/height of the computed margin of an element. */
    const dims = getComputedPropertyDims(elem, "margin");
    return {
        width: dims.left + dims.right,
        height: dims.top + dims.bottom,
    };
}

function getComputedPaddingDims(elem) {
    /** Returns the width/height of the computed padding of an element. */
    const dims = getComputedPropertyDims(elem, "padding");
    return {
        width: dims.left + dims.right,
        height: dims.top + dims.bottom,
    };
}

function getComputedBorderDims(elem) {
    /** Returns the width/height of the computed border of an element. */
    // computed border will always start with the pixel width so thankfully
    // the parseFloat() conversion will just give us the width and ignore the rest.
    // Otherwise we'd have to use border-<pos>-width instead.
    const dims = getComputedPropertyDims(elem, "border");
    return {
        width: dims.left + dims.right,
        height: dims.top + dims.bottom,
    };
}

function getComputedDims(elem) {
    /** Returns the full width and height of an element including its margin, padding, and border. */
    const width = elem.scrollWidth;
    const height = elem.scrollHeight;
    const margin = getComputedMarginDims(elem);
    const padding = getComputedPaddingDims(elem);
    const border = getComputedBorderDims(elem);
    return {
        width: width + margin.width + padding.width + border.width,
        height: height + margin.height + padding.height + border.height,
    };
}

/** Functions for asynchronous operations. */

function debounce(handler, timeout_ms) {
    /** Debounces a function call.
     *
     *  NOTE: This will NOT work if called from within a class.
     *  It will drop `this` from scope.
     *
     *  Repeated calls to the debounce handler will not call the handler until there are
     *  no new calls to the debounce handler for timeout_ms time.
     *
     *  Example:
     *  function add(x, y) { return x + y; }
     *  let debounce_handler = debounce(add, 5000);
     *  let res;
     *  for (let i = 0; i < 10; i++) {
     *      res = debounce_handler(i, 100);
     *  }
     *  console.log("Result:", res);
     *
     *  This example will print "Result: 109".
     */
    let timer = null;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => handler(...args), timeout_ms);
    };
}

function waitForElement(selector) {
    /** Promise that waits for an element to exist in DOM. */
    return new Promise(resolve => {
        if (document.querySelector(selector)) {
            return resolve(document.querySelector(selector));
        }

        const observer = new MutationObserver(mutations => {
            if (document.querySelector(selector)) {
                observer.disconnect();
                resolve(document.querySelector(selector));
            }
        });

        observer.observe(document.documentElement, {
            childList: true,
            subtree: true
        });
    });
}

function waitForBool(o) {
    /** Promise that waits for a boolean to be true.
     *
     *  `o` must be an Object of the form:
     *  { state: <bool value> }
     *
     *  Resolves when (state === true)
     */
    return new Promise(resolve => {
        (function _waitForBool() {
            if (o.state) {
                return resolve();
            }
            setTimeout(_waitForBool, 100);
        })();
    });
}

function waitForKeyInObject(o) {
    /** Promise that waits for a key to exist in an object.
     *
     *  `o` must be an Object of the form:
     *  {
     *      obj: <object to watch for key>,
     *      k: <key to watch for>,
     *  }
     *
     *  Resolves when (k in obj)
     */
    return new Promise(resolve => {
        (function _waitForKeyInObject() {
            if (o.k in o.obj) {
                return resolve();
            }
            setTimeout(_waitForKeyInObject, 100);
        })();
    });
}

function waitForValueInObject(o) {
    /** Promise that waits for a key value pair in an Object.
     *
     *  `o` must be an Object of the form:
     *  {
     *      obj: <object containing value>,
     *      k: <key in object>,
     *      v: <value at key for comparison>
     *  }
     *
     *  Resolves when obj[k] == v
     */
    return new Promise(resolve => {
        waitForKeyInObject({k: o.k, obj: o.obj}).then(() => {
            (function _waitForValueInObject() {

                if (o.k in o.obj && o.obj[o.k] == o.v) {
                    return resolve();
                }
                setTimeout(_waitForValueInObject, 100);
            })();
        });
    });
}

/** Requests */

function requestGet(url, data, handler, errorHandler) {
    var xhr = new XMLHttpRequest();
    var args = Object.keys(data).map(function(k) {
        return encodeURIComponent(k) + '=' + encodeURIComponent(data[k]);
    }).join('&');
    xhr.open("GET", url + "?" + args, true);

    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                try {
                    var js = JSON.parse(xhr.responseText);
                    handler(js);
                } catch (error) {
                    console.error(error);
                    errorHandler();
                }
            } else {
                errorHandler();
            }
        }
    };
    var js = JSON.stringify(data);
    xhr.send(js);
}

function requestGetPromise(url, data) {
    return new Promise((resolve, reject) => {
        let xhr = new XMLHttpRequest();
        let args = Object.keys(data).map(k => {
            return encodeURIComponent(k) + "=" + encodeURIComponent(data[k]);
        }).join("&");
        xhr.open("GET", url + "?" + args, true);

        xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                resolve(xhr.responseText);
            } else {
                reject({status: xhr.status, response: xhr.responseText});
            }
        };

        xhr.onerror = () => {
            reject({status: xhr.status, response: xhr.responseText});
        };
        const payload = JSON.stringify(data);
        xhr.send(payload);
    });
}

/** Misc helper functions. */

function clamp(x, min, max) {
    return Math.max(min, Math.min(x, max));
}

function htmlStringToElement(s) {
    /** Converts an HTML string into an Element type. */
    let parser = new DOMParser();
    let tmp = parser.parseFromString(s, "text/html");
    return tmp.body.firstElementChild;
}

function toggleCss(key, css, enable) {
    var style = document.getElementById(key);
    if (enable && !style) {
        style = document.createElement('style');
        style.id = key;
        style.type = 'text/css';
        document.head.appendChild(style);
    }
    if (style && !enable) {
        document.head.removeChild(style);
    }
    if (style) {
        style.innerHTML == '';
        style.appendChild(document.createTextNode(css));
    }
}

function copyToClipboard(s) {
    /** Copies the passed string to the clipboard. */
    isStringThrowError(s);
    navigator.clipboard.writeText(s);
}
