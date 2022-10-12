

function start_training_dreambooth(){
    requestProgress('db');
    gradioApp().querySelector('#db_error').innerHTML='';
    return args_to_array(arguments);
}
