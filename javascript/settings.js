let settingsExcludeTabsFromShowAll = {
    settings_tab_defaults: 1,
    settings_tab_sysinfo: 1,
    settings_tab_actions: 1,
    settings_tab_licenses: 1,
};

function settingsShowAllTabs() {
    gradioApp().querySelectorAll('#settings > div').forEach(function(elem) {
        if (settingsExcludeTabsFromShowAll[elem.id]) return;

        elem.style.display = "block";
    });
}

function settingsShowOneTab() {
    gradioApp().querySelector('#settings_show_one_page').click();
}

onUiLoaded(function() {
    var edit = gradioApp().querySelector('#settings_search');
    var editTextarea = gradioApp().querySelector('#settings_search > label > input');
    var buttonShowAllPages = gradioApp().getElementById('settings_show_all_pages');
    var settings_tabs = gradioApp().querySelector('#settings div');

    onEdit('settingsSearch', editTextarea, 250, function() {
        var searchText = (editTextarea.value || "").trim().toLowerCase();

        gradioApp().querySelectorAll('#settings > div[id^=settings_] div[id^=column_settings_] > *').forEach(function(elem) {
            var visible = elem.textContent.trim().toLowerCase().indexOf(searchText) != -1;
            elem.style.display = visible ? "" : "none";
        });

        if (searchText != "") {
            settingsShowAllTabs();
        } else {
            settingsShowOneTab();
        }
    });

    settings_tabs.insertBefore(edit, settings_tabs.firstChild);
    settings_tabs.appendChild(buttonShowAllPages);


    buttonShowAllPages.addEventListener("click", settingsShowAllTabs);
});
