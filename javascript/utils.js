/** Helper functions for checking types and simplifying logging/error handling. */

const isNumber = x => typeof x === "number" && isFinite(x);
const isNumberLogError = x => {
    if (isNumber(x)) {
        return true;
    }
    console.error(`expected number, got: ${typeof x}`);
    return false;
};
const isNumberThrowError = x => {
    if (isNumber(x)) {
        return;
    }
    throw new Error(`expected number, got: ${typeof x}`);
};

const isString = x => typeof x === "string" || x instanceof String;
const isStringLogError = x => {
    if (isString(x)) {
        return true;
    }
    console.error(`expected string, got: ${typeof x}`);
    return false;
};
const isStringThrowError = x => {
    if (isString(x)) {
        return;
    }
    throw new Error(`expected string, got: ${typeof x}`);
};

const isNull = x => x === null;
const isUndefined = x => typeof x === "undefined" || x === undefined;
// checks both null and undefined for simplicity sake.
const isNullOrUndefined = x => isNull(x) || isUndefined(x);
const isNullOrUndefinedLogError = x => {
    if (isNullOrUndefined(x)) {
        console.error("Variable is null/undefined.");
        return true;
    }
    return false;
};
const isNullOrUndefinedThrowError = x => {
    if (!isNullOrUndefined(x)) {
        return;
    }
    throw new Error("Variable is null/undefined.");
};

const isElement = x => x instanceof Element;
const isElementLogError = x => {
    if (isElement(x)) {
        return true;
    }
    console.error(`expected element type, got: ${typeof x}`);
    return false;
};
const isElementThrowError = x => {
    if (isElement(x)) {
        return;
    }
    throw new Error(`expected element type, got: ${typeof x}`);
};

const isFunction = x => typeof x === "function";
const isFunctionLogError = x => {
    if (isFunction(x)) {
        return true;
    }
    console.error(`expected function type, got: ${typeof x}`);
    return false;
};
const isFunctionThrowError = x => {
    if (isFunction(x)) {
        return;
    }
    throw new Error(`expected function type, got: ${typeof x}`);
};

const isObject = x => typeof x === "object" && !Array.isArray(x);
const isObjectLogError = x => {
    if (isObject(x)) {
        return true;
    }
    console.error(`expected object type, got: ${typeof x}`);
    return false;
};
const isObjectThrowError = x => {
    if (isObject(x)) {
        return;
    }
    throw new Error(`expected object type, got: ${typeof x}`);
};

const keyExists = (obj, k) => isObject(obj) && isString(k) && k in obj;
const keyExistsLogError = (obj, k) => {
    if (keyExists(obj, k)) {
        return true;
    }
    console.error(`key does not exist in object: ${k}`);
    return false;
};
const keyExistsThrowError = (obj, k) => {
    if (keyExists(obj, k)) {
        return;
    }
    throw new Error(`key does not exist in object: ${k}`)
};

const getValue = (obj, k) => {
    /** Returns value of object for given key if it exists, otherwise returns null. */
    if (keyExists(obj, k)) {
        return obj[k];
    }
    return null;
};
const getValueLogError = (obj, k) => {
    if (keyExistsLogError(obj, k)) {
        return obj[k];
    }
    return null;
};
const getValueThrowError = (obj, k) => {
    keyExistsThrowError(obj, k);
    return obj[k];
};

const getElementByIdLogError = selector => {
    const elem = gradioApp().getElementById(selector);
    isElementLogError(elem);
    return elem;
};
const getElementByIdThrowError = selector => {
    const elem = gradioApp().getElementById(selector);
    isElementThrowError(elem);
    return elem;
};

const querySelectorLogError = selector => {
    const elem = gradioApp().querySelector(selector);
    isElementLogError(elem);
    return elem;
};
const querySelectorThrowError = selector => {
    const elem = gradioApp().querySelector(selector);
    isElementThrowError(elem);
    return elem;
};

/** Functions for getting dimensions of elements. */

const getComputedPropertyDims = (elem, prop) => {
    /** Returns the top/left/bottom/right float dimensions of an element for the specified property. */
    const style = window.getComputedStyle(elem, null);
    return {
        top: parseFloat(style.getPropertyValue(`${prop}-top`)),
        left: parseFloat(style.getPropertyValue(`${prop}-left`)),
        bottom: parseFloat(style.getPropertyValue(`${prop}-bottom`)),
        right: parseFloat(style.getPropertyValue(`${prop}-right`)),
    };
};

const getComputedMarginDims = elem => {
    /** Returns the width/height of the computed margin of an element. */
    const dims = getComputedPropertyDims(elem, "margin");
    return {
        width: dims.left + dims.right,
        height: dims.top + dims.bottom,
    };
};

const getComputedPaddingDims = elem => {
    /** Returns the width/height of the computed padding of an element. */
    const dims = getComputedPropertyDims(elem, "padding");
    return {
        width: dims.left + dims.right,
        height: dims.top + dims.bottom,
    };
};

const getComputedBorderDims = elem => {
    /** Returns the width/height of the computed border of an element. */
    // computed border will always start with the pixel width so thankfully
    // the parseFloat() conversion will just give us the width and ignore the rest.
    // Otherwise we'd have to use border-<pos>-width instead.
    const dims = getComputedPropertyDims(elem, "border");
    return {
        width: dims.left + dims.right,
        height: dims.top + dims.bottom,
    };
};

const getComputedDims = elem => {
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
};

const calcColsPerRow = function (parent, child) {
    /** Calculates the number of columns of children that can fit in a parent's visible width. */
    const parent_inner_width = parent.offsetWidth - getComputedPaddingDims(parent).width;
    return parseInt(parent_inner_width / getComputedDims(child).width, 10);

};

const calcRowsPerCol = function (parent, child) {
    /** Calculates the number of rows of children that can fit in a parent's visible height. */
    const parent_inner_height = parent.offsetHeight - getComputedPaddingDims(parent).height;
    return parseInt(parent_inner_height / getComputedDims(child).height, 10);
};

/** Functions for asynchronous operations. */

const debounce = (handler, timeout_ms) => {
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
};

const waitForElement = selector => {
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
};

const waitForBool = o => {
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
};

const waitForKeyInObject = o => {
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
};

const waitForValueInObject = o => {
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
        waitForKeyInObject({ k: o.k, obj: o.obj }).then(() => {
            (function _waitForValueInObject() {

                if (o.k in o.obj && o.obj[o.k] == o.v) {
                    return resolve();
                }
                setTimeout(_waitForValueInObject, 100);
            })();
        });
    });
};

/** Requests */

const requestGet = (url, data, handler, errorHandler) => {
    var xhr = new XMLHttpRequest();
    var args = Object.keys(data).map(function (k) {
        return encodeURIComponent(k) + '=' + encodeURIComponent(data[k]);
    }).join('&');
    xhr.open("GET", url + "?" + args, true);

    xhr.onreadystatechange = function () {
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
};

const requestGetPromise = (url, data) => {
    return new Promise((resolve, reject) => {
        let xhr = new XMLHttpRequest();
        let args = Object.keys(data).map(k => {
            return encodeURIComponent(k) + "=" + encodeURIComponent(data[k]);
        }).join("&");
        xhr.open("GET", url + "?" + args, true);

        xhr.onreadystatechange = function () {
            if (xhr.readyState === 4) {
                if (xhr.status === 200) {
                    try {
                        resolve(xhr.responseText);
                    } catch (error) {
                        reject(error);
                    }
                } else {
                    reject({ status: this.status, statusText: xhr.statusText });
                }
            }
        };
        xhr.send(JSON.stringify(data));
    });
};

/** Misc helper functions. */

const clamp = (x, min, max) => Math.max(min, Math.min(x, max));

const getStyle = (prop, elem) => {
    return window.getComputedStyle ? window.getComputedStyle(elem)[prop] : elem.currentStyle[prop];
};

const htmlStringToElement = function (str) {
    /** Converts an HTML string into an Element type. */
    let parser = new DOMParser();
    let tmp = parser.parseFromString(str, "text/html");
    return tmp.body.firstElementChild;
};

const toggleCss = (key, css, enable) => {
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
};

const copyToClipboard = s => {
    /** Copies the passed string to the clipboard. */
    isStringThrowError(s);
    navigator.clipboard.writeText(s);
};