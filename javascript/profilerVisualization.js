
function createRow(table, cellName, items) {
    var tr = document.createElement('tr');
    var res = [];

    items.forEach(function(x) {
        var td = document.createElement(cellName);
        td.textContent = x;
        tr.appendChild(td);
        res.push(td);
    });

    table.appendChild(tr);

    return res;
}

function showProfile(path, cutoff = 0.0005) {
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

        var addLevel = function(level, parent) {
            var matching = items.filter(function(x) {
                return x.parts[level] && !x.parts[level + 1] && arraysEqual(x.parts.slice(0, level), parent);
            });
            var sorted = matching.sort(function(a, b) {
                return b.time - a.time;
            });
            var othersTime = 0;
            var othersList = [];
            sorted.forEach(function(x) {
                if (x.time < cutoff) {
                    othersTime += x.time;
                    othersList.push(x.parts[level]);
                    return;
                }

                var cells = [];
                for (var i = 0; i < maxLength; i++) {
                    cells.push(x.parts[i]);
                }
                cells.push(x.time.toFixed(3));
                var cols = createRow(table, 'td', cells);
                for (i = 0; i < level; i++) {
                    cols[i].className = 'muted';
                }

                addLevel(level + 1, parent.concat([x.parts[level]]));
            });

            if (othersTime > 0) {
                var cells = [];
                for (var i = 0; i < maxLength; i++) {
                    cells.push(parent[i]);
                }
                cells.push(othersTime.toFixed(3));
                var cols = createRow(table, 'td', cells);
                for (i = 0; i < level; i++) {
                    cols[i].className = 'muted';
                }

                cols[level].textContent = 'others';
                cols[level].title = othersList.join(", ");
            }
        };

        addLevel(0, []);

        popup(table);
    });
}

