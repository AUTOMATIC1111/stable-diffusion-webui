
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

function install_extension_from_index(button, url){
    button.disabled = "disabled"
    button.value = "Installing..."

    textarea = gradioApp().querySelector('#extension_to_install textarea')
    textarea.value = url
	textarea.dispatchEvent(new Event("input", { bubbles: true }))

    gradioApp().querySelector('#install_extension_button').click()
}
