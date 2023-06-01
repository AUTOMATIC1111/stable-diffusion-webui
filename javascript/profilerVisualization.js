
function createRow(table, cellName, items) {
    var tr = document.createElement('tr');
    var res = [];

    items.forEach(function(x, i) {
        if (x === undefined) {
            res.push(null);
            return;
        }

        var td = document.createElement(cellName);
        td.textContent = x;
        tr.appendChild(td);
        res.push(td);

        var colspan = 1;
        for (var n = i + 1; n < items.length; n++) {
            if (items[n] !== undefined) {
                break;
            }

            colspan += 1;
        }

        if (colspan > 1) {
            td.colSpan = colspan;
        }
    });

    table.appendChild(tr);

    return res;
}

function showProfile(path, cutoff = 0.05) {
    requestGet(path, {}, function(data) {
        var table = document.createElement('table');
        table.className = 'popup-table';

        data.records['total'] = data.total;
        var keys = Object.keys(data.records).sort(function(a, b) {
            return data.records[b] - data.records[a];
        });
        var items = keys.map(function(x) {
            return {key: x, parts: x.split('/'), time: data.records[x]};
        });
        var maxLength = items.reduce(function(a, b) {
            return Math.max(a, b.parts.length);
        }, 0);

        var cols = createRow(table, 'th', ['record', 'seconds']);
        cols[0].colSpan = maxLength;

        function arraysEqual(a, b) {
            return !(a < b || b < a);
        }

        var addLevel = function(level, parent, hide) {
            var matching = items.filter(function(x) {
                return x.parts[level] && !x.parts[level + 1] && arraysEqual(x.parts.slice(0, level), parent);
            });
            var sorted = matching.sort(function(a, b) {
                return b.time - a.time;
            });
            var othersTime = 0;
            var othersList = [];
            var othersRows = [];
            var childrenRows = [];
            sorted.forEach(function(x) {
                var visible = x.time >= cutoff && !hide;

                var cells = [];
                for (var i = 0; i < maxLength; i++) {
                    cells.push(x.parts[i]);
                }
                cells.push(x.time.toFixed(3));
                var cols = createRow(table, 'td', cells);
                for (i = 0; i < level; i++) {
                    cols[i].className = 'muted';
                }

                var tr = cols[0].parentNode;
                if (!visible) {
                    tr.classList.add("hidden");
                }

                if (x.time >= cutoff) {
                    childrenRows.push(tr);
                } else {
                    othersTime += x.time;
                    othersList.push(x.parts[level]);
                    othersRows.push(tr);
                }

                var children = addLevel(level + 1, parent.concat([x.parts[level]]), true);
                if (children.length > 0) {
                    var cell = cols[level];
                    var onclick = function() {
                        cell.classList.remove("link");
                        cell.removeEventListener("click", onclick);
                        children.forEach(function(x) {
                            x.classList.remove("hidden");
                        });
                    };
                    cell.classList.add("link");
                    cell.addEventListener("click", onclick);
                }
            });

            if (othersTime > 0) {
                var cells = [];
                for (var i = 0; i < maxLength; i++) {
                    cells.push(parent[i]);
                }
                cells.push(othersTime.toFixed(3));
                cells[level] = 'others';
                var cols = createRow(table, 'td', cells);
                for (i = 0; i < level; i++) {
                    cols[i].className = 'muted';
                }

                var cell = cols[level];
                var tr = cell.parentNode;
                var onclick = function() {
                    tr.classList.add("hidden");
                    cell.classList.remove("link");
                    cell.removeEventListener("click", onclick);
                    othersRows.forEach(function(x) {
                        x.classList.remove("hidden");
                    });
                };

                cell.title = othersList.join(", ");
                cell.classList.add("link");
                cell.addEventListener("click", onclick);

                if (hide) {
                    tr.classList.add("hidden");
                }

                childrenRows.push(tr);
            }

            return childrenRows;
        };

        addLevel(0, []);

        popup(table);
    });
}

