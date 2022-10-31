
function extensions_apply(_, _){
    disable = []
    update = []
    gradioApp().querySelectorAll('#extensions input[type="checkbox"]').forEach(function(x){
        if(x.name.startsWith("enable_") && ! x.checked)
            disable.push(x.name.substr(7))

        if(x.name.startsWith("update_") && x.checked)
            update.push(x.name.substr(7))
    })

    restart_reload()

    return [JSON.stringify(disable), JSON.stringify(update)]
}

function extensions_check(){
    gradioApp().querySelectorAll('#extensions .extension_status').forEach(function(x){
        x.innerHTML = "Loading..."
    })

    return []
}