// Prevent eslint errors on functions defined in other files.
/*global
    isNumberThrowError,
    isNullOrUndefined,
*/
/*eslint no-undef: "error"*/

const LRU_CACHE_MAX_ITEMS_DEFAULT = 250;
class LRUCache {
    /** Least Recently Used cache implementation.
     *
     *  Source: https://stackoverflow.com/a/46432113
    */
    constructor(max = LRU_CACHE_MAX_ITEMS_DEFAULT) {
        isNumberThrowError(max);

        this.max = max;
        this.cache = new Map();
    }

    clear() {
        this.cache.clear();
    }

    destroy() {
        this.clear();
        this.cache = null;
    }

    size() {
        return this.cache.size;
    }

    get(key) {
        let item = this.cache.get(key);
        if (!isNullOrUndefined(item)) {
            this.cache.delete(key);
            this.cache.set(key, item);
        }
        return item;
    }

    set(key, val) {
        if (this.cache.has(key)) {
            this.cache.delete(key);
        } else if (this.cache.size === this.max) {
            this.cache.delete(this.first());
        }
        this.cache.set(key, val);
    }

    has(key) {
        return this.cache.has(key);
    }

    first() {
        return this.cache.keys().next().value;
    }
}
