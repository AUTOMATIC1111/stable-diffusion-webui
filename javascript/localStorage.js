
function localSet(k, v) {
    try {
        localStorage.setItem(k, v);
    } catch (e) {
        console.warn(`Failed to save ${k} to localStorage: ${e}`);
    }
}

function localGet(k, def) {
    try {
        return localStorage.getItem(k);
    } catch (e) {
        console.warn(`Failed to load ${k} from localStorage: ${e}`);
    }

    return def;
}

function localRemove(k) {
    try {
        return localStorage.removeItem(k);
    } catch (e) {
        console.warn(`Failed to remove ${k} from localStorage: ${e}`);
    }
}
