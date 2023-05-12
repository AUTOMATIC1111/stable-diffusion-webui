// various hints and extra info for the settings tab

onUiLoaded(function(){
    createLink = function(elem_id, text, href){
        var a = document.createElement('A')
        a.textContent = text
        a.target = '_blank';

        elem = gradioApp().querySelector('#'+elem_id)
        elem.insertBefore(a, elem.querySelector('label'))

        return a
    }

    createLink("setting_samples_filename_pattern", "[wiki] ").href = "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"
    createLink("setting_directories_filename_pattern", "[wiki] ").href = "https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Images-Filename-Name-and-Subdirectory"

    createLink("setting_quicksettings_list", "[info] ").addEventListener("click", function(event){
        requestGet("./internal/quicksettings-hint", {}, function(data){
            var table = document.createElement('table')
            table.className = 'settings-value-table'

            data.forEach(function(obj){
                var tr = document.createElement('tr')
                var td = document.createElement('td')
                td.textContent = obj.name
                tr.appendChild(td)

                var td = document.createElement('td')
                td.textContent = obj.label
                tr.appendChild(td)

                table.appendChild(tr)
            })

            popup(table);
        })
    });
})


