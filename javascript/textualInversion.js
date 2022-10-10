

function start_training_textual_inversion(){
    requestProgress('ti')
    gradioApp().querySelector('#ti_error').innerHTML=''

    return args_to_array(arguments)
}
