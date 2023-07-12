function startTrainMonitor() {
  gradioApp().querySelector('#train_error').innerHTML = '';
  const id = randomId();
  const onProgress = (progress) => gradioApp().getElementById('train_progress').innerHTML = progress.textinfo;
  requestProgress(id, gradioApp().getElementById('train_gallery'), null, onProgress, false);
  const res = Array.from(arguments);
  res[0] = id;
  return res;
}
