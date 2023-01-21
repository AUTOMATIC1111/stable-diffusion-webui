
function setupExtraNetworksForTab(tabname){
    gradioApp().querySelector('#'+tabname+'_extra_tabs').classList.add('extra-networks')

    gradioApp().querySelector('#'+tabname+'_extra_tabs > div').appendChild(gradioApp().getElementById(tabname+'_extra_refresh'))
    gradioApp().querySelector('#'+tabname+'_extra_tabs > div').appendChild(gradioApp().getElementById(tabname+'_extra_close'))
}

var activePromptTextarea = null;
var activePositivePromptTextarea = null;

function setupExtraNetworks(){
    setupExtraNetworksForTab('txt2img')
    setupExtraNetworksForTab('img2img')

    function registerPrompt(id, isNegative){
        var textarea = gradioApp().querySelector("#" + id + " > label > textarea");

        if (activePromptTextarea == null){
            activePromptTextarea = textarea
        }
        if (activePositivePromptTextarea == null && ! isNegative){
            activePositivePromptTextarea = textarea
        }

		textarea.addEventListener("focus", function(){
            activePromptTextarea = textarea;
            if(! isNegative)  activePositivePromptTextarea = textarea;
		});
    }

    registerPrompt('txt2img_prompt')
    registerPrompt('txt2img_neg_prompt', true)
    registerPrompt('img2img_prompt')
    registerPrompt('img2img_neg_prompt', true)
}

onUiLoaded(setupExtraNetworks)

function cardClicked(textToAdd, allowNegativePrompt){
    textarea = allowNegativePrompt ? activePromptTextarea : activePositivePromptTextarea

    textarea.value = textarea.value + " " + textToAdd
    updateInput(textarea)

    return false
}

function saveCardPreview(event, tabname, filename){
    textarea = gradioApp().querySelector("#" + tabname + '_preview_filename  > label > textarea')
    button = gradioApp().getElementById(tabname + '_save_preview')

    textarea.value = filename
    updateInput(textarea)

    button.click()

    event.stopPropagation()
    event.preventDefault()
}
