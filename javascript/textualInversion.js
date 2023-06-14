function start_training_textual_inversion() {
  gradioApp().querySelector('#ti_error').innerHTML=''
  var id = randomId()
  const onProgress = (progress) => gradioApp().getElementById('ti_progress').innerHTML = progress.textinfo;
  requestProgress(id, gradioApp().getElementById('ti_gallery'), null, onProgress, false)
  var res = Array.from(arguments);
  res[0] = id
  return res
}
