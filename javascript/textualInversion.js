function start_training_textual_inversion() {
  gradioApp().querySelector('#ti_error').innerHTML=''
  var id = randomId()
  const onProgress = (progress) => gradioApp().getElementById('ti_progress').innerHTML = progress.textinfo;
  // requestProgress(id_task, progressbarContainer, gallery, atEnd = null, onProgress = null, once = false) {
  requestProgress(id, gradioApp().getElementById('ti_output'), gradioApp().getElementById('ti_gallery'), null, onProgress, false)
  var res = args_to_array(arguments)
  res[0] = id
  return res
}
