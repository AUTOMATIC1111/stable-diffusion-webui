


function start_training_textual_inversion(){
    gradioApp().querySelector('#ti_error').innerHTML=''

    var id = randomId()
    requestProgress(id, gradioApp().getElementById('ti_output'), gradioApp().getElementById('ti_gallery'), function(){}, function(progress){
        gradioApp().getElementById('ti_progress').innerHTML = progress.textinfo
    })

    var res = args_to_array(arguments)

    res[0] = id

    return res
}
