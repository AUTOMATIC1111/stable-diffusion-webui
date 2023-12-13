const getControlActiveTab = (...args) => {
  const selectedTab = gradioApp().querySelector('#control-tabs > .tab-nav > .selected');
  let activeTab = '';
  if (selectedTab) activeTab = selectedTab.innerText.toLowerCase();
  args.shift();
  return [activeTab, ...args];
};
