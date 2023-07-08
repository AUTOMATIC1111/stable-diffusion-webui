// various hints and extra info for the settings tab

var settingsHintsSetup = false;

onOptionsChanged(function() {
    if (settingsHintsSetup) return;
    settingsHintsSetup = true;

    gradioApp().querySelectorAll('#settings [id^=setting_]').forEach(function(div) {
        var name = div.id.substr(8);
        var commentBefore = opts._comments_before[name];
        var commentAfter = opts._comments_after[name];

        if (!commentBefore && !commentAfter) return;

        var span = null;
        if (div.classList.contains('gradio-checkbox')) span = div.querySelector('label span');
        else if (div.classList.contains('gradio-checkboxgroup')) span = div.querySelector('span').firstChild;
        else if (div.classList.contains('gradio-radio')) span = div.querySelector('span').firstChild;
        else span = div.querySelector('label span').firstChild;

        if (!span) return;

        if (commentBefore) {
            var comment = document.createElement('DIV');
            comment.className = 'settings-comment';
            comment.innerHTML = commentBefore;
            span.parentElement.insertBefore(document.createTextNode('\xa0'), span);
            span.parentElement.insertBefore(comment, span);
            span.parentElement.insertBefore(document.createTextNode('\xa0'), span);
        }
        if (commentAfter) {
            comment = document.createElement('DIV');
            comment.className = 'settings-comment';
            comment.innerHTML = commentAfter;
            span.parentElement.insertBefore(comment, span.nextSibling);
            span.parentElement.insertBefore(document.createTextNode('\xa0'), span.nextSibling);
        }
    });
});

function settingsHintsShowQuicksettings() {
    requestGet("./internal/quicksettings-hint", {}, function(data) {
        var table = document.createElement('table');
        table.className = 'popup-table';

        data.forEach(function(obj) {
            var tr = document.createElement('tr');
            var td = document.createElement('td');
            td.textContent = obj.name;
            tr.appendChild(td);

            td = document.createElement('td');
            td.textContent = obj.label;
            tr.appendChild(td);

            table.appendChild(tr);
        });

        popup(table);
    });
}
