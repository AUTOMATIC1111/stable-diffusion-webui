function extensions_apply(extensions_disabled_list, extensions_update_list, disable_all) {
  const disable = [];
  const update = [];
  gradioApp().querySelectorAll('#extensions input[type="checkbox"]').forEach((x) => {
    if (x.name.startsWith('enable_') && !x.checked) disable.push(x.name.substr(7));
    if (x.name.startsWith('update_') && x.checked) update.push(x.name.substr(7));
  });
  restart_reload();
  return [JSON.stringify(disable), JSON.stringify(update), disable_all];
}

function extensions_check(info, extensions_disabled_list, search_text, sort_column) {
  const disable = [];
  gradioApp().querySelectorAll('#extensions input[type="checkbox"]').forEach((x) => {
    if (x.name.startsWith('enable_') && !x.checked) disable.push(x.name.substr(7));
  });
  gradioApp().querySelectorAll('#extensions .extension_status').forEach((x) => {
    x.innerHTML = 'Loading...';
  });
  const id = randomId();
  // requestProgress(id, gradioApp().getElementById('extensions_installed_top'), null, null, null, false);
  return [id, JSON.stringify(disable), search_text, sort_column];
}

function install_extension(button, url) {
  button.disabled = 'disabled';
  button.value = 'Installing...';
  const textarea = gradioApp().querySelector('#extension_to_install textarea');
  textarea.value = url;
  updateInput(textarea);
  gradioApp().querySelector('#install_extension_button').click();
}

function uninstall_extension(button, url) {
  button.disabled = 'disabled';
  button.value = 'Uninstalling...';
  const textarea = gradioApp().querySelector('#extension_to_install textarea');
  textarea.value = url;
  updateInput(textarea);
  gradioApp().querySelector('#uninstall_extension_button').click();
}
