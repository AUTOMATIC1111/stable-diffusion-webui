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

function waitForElement(selector, timeout_ms) {
    /** Promise that waits for an element to exist in DOM. */
    return new Promise((resolve, reject) => {
        if (document.querySelector(selector)) {
            return resolve(document.querySelector(selector));
        }

        const observer = new MutationObserver(mutations => {
            if (document.querySelector(selector)) {
                observer.disconnect();
                return resolve(document.querySelector(selector));
            }
        });

        observer.observe(document.documentElement, {
            childList: true,
            subtree: true
        });

        if (isNumber(timeout_ms) && timeout_ms !== 0) {
            setTimeout(() => {
                observer.takeRecords();
                observer.disconnect();
                return reject(`timed out waiting for element: "${selector}"`);
            }, timeout_ms);
        }
    });
}

function waitForBool(o, timeout_ms) {
    /** Promise that waits for a boolean to be true.
     *
     *  `o` must be an Object of the form:
     *  { state: <bool value> }
     *
     *  If timeout_ms is null/undefined or 0, waits forever.
     *
     *  Resolves when (state === true)
     *  Rejects when state is not True before timeout_ms.
     */
    let wait_timer;
    return new Promise((resolve, reject) => {
        if (isNumber(timeout_ms) && timeout_ms !== 0) {
            setTimeout(() => {
                clearTimeout(wait_timer);
                return reject("timed out waiting for bool");
            }, timeout_ms);
        }
        (function _waitForBool() {
            if (o.state) {
                return resolve();
            }
            wait_timer = setTimeout(_waitForBool, 100);
        })();
    });
}

function waitForKeyInObject(o, timeout_ms) {
    /** Promise that waits for a key to exist in an object.
     *
     *  `o` must be an Object of the form:
     *  {
     *      obj: <object to watch for key>,
     *      k: <key to watch for>,
     *  }
     *
     *  If timeout_ms is null/undefined or 0, waits forever.
     *
     *  Resolves when (k in obj).
     *  Rejects when k is not found in obj before timeout_ms.
     */
    let wait_timer;
    return new Promise((resolve, reject) => {
        if (isNumber(timeout_ms) && timeout_ms !== 0) {
            setTimeout(() => {
                clearTimeout(wait_timer);
                return reject(`timed out waiting for key: ${o.k}`);
            }, timeout_ms);
        }
        (function _waitForKeyInObject() {
            if (o.k in o.obj) {
                return resolve();
            }
            wait_timer = setTimeout(_waitForKeyInObject, 100);
        })();
    });
}

function waitForValueInObject(o, timeout_ms) {
    /** Promise that waits for a key value pair in an Object.
     *
     *  `o` must be an Object of the form:
     *  {
     *      obj: <object containing value>,
     *      k: <key in object>,
     *      v: <value at key for comparison>
     *  }
     *
     *  If timeout_ms is null/undefined or 0, waits forever.
     *
     *  Resolves when obj[k] == v
     */
    let wait_timer;
    return new Promise((resolve, reject) => {
        if (isNumber(timeout_ms) && timeout_ms !== 0) {
            setTimeout(() => {
                clearTimeout(wait_timer);
                return reject(`timed out waiting for value: ${o.k}: ${o.v}`);
            }, timeout_ms);
        }
        waitForKeyInObject({k: o.k, obj: o.obj}, timeout_ms).then(() => {
            (function _waitForValueInObject() {

                if (o.k in o.obj && o.obj[o.k] == o.v) {
                    return resolve();
                }
                setTimeout(_waitForValueInObject, 100);
            })();
        }).catch((error) => {
            return reject(error);
        });
    });
}

/** Requests */

class FetchError extends Error {
    constructor(...args) {
        super(...args);
        this.name = this.constructor.name;
    }
}

class Fetch4xxError extends FetchError {
    constructor(...args) {
        super(...args);
    }
}

class Fetch5xxError extends FetchError {
    constructor(...args) {
        super(...args);
    }
}

class FetchRetryLimitError extends FetchError {
    constructor(...args) {
        super(...args);
    }
}

class FetchTimeoutError extends FetchError {
    constructor(...args) {
        super(...args);
    }
}

class FetchWithRetryAndBackoffTimeoutError extends FetchError {
    constructor(...args) {
        super(...args);
    }
}

async function fetchWithRetryAndBackoff(url, data, args = {}) {
    /** Wrapper around `fetch` with retries, backoff, and timeout.
     *
     *  Uses a Decorrelated jitter backoff strategy.
     *  https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
     *
     *  Args:
     *      url:                    Primary URL to fetch.
     *      data:                   Data to append to the URL when making the request.
     *      opts:
     *          method:             The HTTP request method to use.
     *          timeout_ms:         Max allowed time before this function fails.
     *          fetch_timeout_ms:   Max allowed time for individual `fetch` calls.
     *          min_delay_ms:       Min time that a delay between requests can be.
     *          max_delay_ms:       Max time that a delay between reqeusts can be.
     *          response_handler:   A callback function that returns a promise.
     *              This function is sent the response from the `fetch` call.
     *              If not specified, all status codes >= 400 are handled as errors.
     *              This is useful for handling requests whose responses from the server
     *              are erronious but the HTTP status is 200.
     */
    args.method = args.method || "GET";
    args.timeout_ms = args.timeout_ms || 30000;
    args.min_delay_ms = args.min_delay_ms || 100;
    args.max_delay_ms = args.max_delay_ms || 3000;
    args.fetch_timeout_ms = args.fetch_timeout_ms || 10000;
    // The default response handler function for `fetch` call responses.
    const response_handler = (response) => new Promise((resolve, reject) => {
        if (response.ok) {
            return response.json().then(json => {
                return resolve(json);
            });
        } else {
            if (response.status >= 400 && response.status < 500) {
                throw new Fetch4xxError("client error:", response);
            }
            if (response.status >= 500 && response.status < 600) {
                throw new Fetch5xxError("server error:", response);
            }
            return reject(response);
        }
    });
    args.response_handler = args.response_handler || response_handler;

    const url_args = Object.entries(data).map(([k, v]) => {
        return `${encodeURIComponent(k)}=${encodeURIComponent(v)}`;
    }).join("&");
    url = `${url}?${url_args}`;

    let controller;
    let retry = true;

    const randrange = (min, max) => {
        return Math.floor(Math.random() * (max - min + 1) + min);
    };
    const get_jitter = (base, max, prev) => {
        return Math.min(max, randrange(base, prev * 3));
    };

    const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

    // Timeout for individual `fetch` calls.
    const fetch_timeout = (ms, promise) => {
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                return reject(new FetchTimeoutError("Fetch timed out."));
            }, ms);
            return promise.then(resolve, reject);
        });
    };

    // Timeout for all retries.
    const run_timeout = (ms, promise) => {
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                retry = false;
                controller.abort();
                return reject(new FetchWithRetryAndBackoffTimeoutError("Request timed out."));
            }, ms);
            return promise.then(resolve, reject);
        });
    };

    const run = async(delay_ms) => {
        if (!retry) {
            // Retry is controlled externally via `run_timeout`. This function's promise
            // is also handled via that timeout so we can just return here.
            return;
        }
        try {
            controller = new AbortController();
            const fetch_opts = {method: args.method, signal: controller.signal};
            const response = await fetch_timeout(args.fetch_timeout_ms, fetch(url, fetch_opts));
            return await args.response_handler(response);
        } catch (error) {
            controller.abort();
            // dont bother with anything else if told to not retry.
            if (!retry) {
                return;
            }

            if (error instanceof Fetch4xxError) {
                throw error;
            }
            if (error instanceof Fetch5xxError) {
                throw error;
            }
            // Any other errors mean we need to retry the request.
            delay_ms = get_jitter(args.min_delay_ms, args.max_delay_ms, delay_ms);
            await delay(delay_ms);
            return await run(delay_ms);
        }
    };
    return await run_timeout(args.timeout_ms, run(args.min_delay_ms));
}

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

function requestGetPromise(url, data, timeout_ms) {
    /**Asynchronous `GET` request that returns a promise.
     *
     * The result will be of the format {status: int, response: JSON object}.
     * Thus, the xhr.responseText that we receive is expected to be a JSON string.
     * Acceptable status codes for successful requests are 200 <= status < 300.
    */
    if (!isNumber(timeout_ms)) {
        timeout_ms = 1000;
    }
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const args = Object.entries(data).map(([k, v]) => {
            return `${encodeURIComponent(k)}=${encodeURIComponent(v)}`;
        }).join("&");

        xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                return resolve({status: xhr.status, response: JSON.parse(xhr.responseText)});
            } else {
                return reject({status: xhr.status, response: JSON.parse(xhr.responseText)});
            }
        };

        xhr.onerror = () => {
            return reject({status: xhr.status, response: JSON.parse(xhr.responseText)});
        };

        xhr.ontimeout = () => {
            return reject({status: 408, response: {detail: `Request timeout: ${url}`}});
        };

        const payload = JSON.stringify(data);
        xhr.open("GET", `${url}?${args}`, true);
        xhr.timeout = timeout_ms;
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

function htmlStringToFragment(s) {
    /** Converts an HTML string into a DocumentFragment. */
    return document.createRange().createContextualFragment(s);
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

function attrPromise({elem, attr, timeout_ms} = {}) {
    timeout_ms = timeout_ms || 0;
    return new Promise((resolve, reject) => {
        let res = false;
        const observer_config = {attributes: true, attributeOldValue: true};
        const observer = new MutationObserver(mutations => {
            mutations.forEach(mutation => {
                if (isString(attr) && mutation.attributeName === attr) {
                    res = true;
                    observer.disconnect();
                    resolve(elem, elem.getAttribute(attr));
                }
                if (!isString(attr)) {
                    res = true;
                    observer.disconnect();
                    resolve(elem);
                }
            });
        });

        if (timeout_ms > 0) {
            setTimeout(() => {
                if (!res) {
                    reject(elem);
                }
            }, timeout_ms);
        }

        if (isString(attr)) {
            observer_config.attributeFilter = [attr];
        }
        observer.observe(elem, observer_config);
    });
}

function waitForVisible(elem, callback) {
    new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.intersectionRatio > 0) {
                callback(elem);
                observer.disconnect();
            }
        });
    }).observe(elem);
    if (!callback) return new Promise(resolve => callback = resolve);
}
