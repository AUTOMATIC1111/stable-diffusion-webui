function extensions_apply(extensions_disabled_list, extensions_update_list, disable_all) {
  console.log('Extensions apply:', extensions_disabled_list, extensions_update_list, disable_all);
  const disable = [];
  const update = [];
  gradioApp().querySelectorAll('#extensions input[type="checkbox"]').forEach((x) => {
    if (x.name.startsWith('enable_') && !x.checked) disable.push(x.name.substring(7));
    if (x.name.startsWith('update_') && x.checked) update.push(x.name.substring(7));
  });
  restart_reload();
  return [JSON.stringify(disable), JSON.stringify(update), disable_all];
}

function extensions_check(info, extensions_disabled_list, search_text, sort_column) {
  console.log('Extensions check:', info, extensions_disabled_list);
  const disable = [];
  gradioApp().querySelectorAll('#extensions input[type="checkbox"]').forEach((x) => {
    if (x.name.startsWith('enable_') && !x.checked) disable.push(x.name.substring(7));
  });
  // gradioApp().querySelectorAll('#extensions .extension_status').forEach((x) => {
  //    x.innerHTML = 'Loading...';
  // });
  const id = randomId();
  // requestProgress(id, gradioApp().getElementById('extensions_installed_top'), null, null, null, false);
  return [id, JSON.stringify(disable), search_text, sort_column];
}

function install_extension(button, url) {
  console.log('Extension install:', url);
  button.disabled = 'disabled';
  button.value = 'Installing...';
  button.innerHTML = 'installing';
  const textarea = gradioApp().querySelector('#extension_to_install textarea');
  textarea.value = url;
  updateInput(textarea);
  gradioApp().querySelector('#install_extension_button').click();
}

function uninstall_extension(button, url) {
  console.log('Extension uninstall:', url, JSON.stringify(url), decodeURIComponent(url), encodeURI(url));
  button.disabled = 'disabled';
  button.value = 'Uninstalling...';
  button.innerHTML = 'uninstalling';
  const textarea = gradioApp().querySelector('#extension_to_install textarea');
  textarea.value = url;
  updateInput(textarea);
  gradioApp().querySelector('#uninstall_extension_button').click();
}

function update_extension(button, url) {
  console.log('Extension update:', url);
  button.value = 'Updating...';
  button.innerHTML = 'updating';
  const textarea = gradioApp().querySelector('#extension_to_install textarea');
  textarea.value = url;
  updateInput(textarea);
  gradioApp().querySelector('#update_extension_button').click();
}
