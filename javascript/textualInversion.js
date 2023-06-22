function start_train_monitoring() {
  gradioApp().querySelector('#train_error').innerHTML=''
  var id = randomId()
  const onProgress = (progress) => gradioApp().getElementById('train_progress').innerHTML = progress.textinfo;
  requestProgress(id, gradioApp().getElementById('train_gallery'), null, onProgress, false)
  var res = Array.from(arguments);
  res[0] = id
  return res
}
