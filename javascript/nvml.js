let nvmlInterval = true; // eslint-disable-line prefer-const
let nvmlEl = null;
let nvmlTable = null;

async function updateNVML() {
  try {
    const res = await fetch('/sdapi/v1/nvml');
    if (!res.ok) {
      clearInterval(nvmlInterval);
      nvmlEl.style.display = 'none';
      return;
    }
    const data = await res.json();
    if (!data) {
      clearInterval(nvmlInterval);
      nvmlEl.style.display = 'none';
      return;
    }
    const nvmlTbody = nvmlTable.querySelector('tbody');
    for (const gpu of data) {
      const rows = `
        <tr><td>GPU</td><td>${gpu.name}</td></tr>
        <tr><td>Driver</td><td>${gpu.version.driver}</td></tr>
        <tr><td>VBIOS</td><td>${gpu.version.vbios}</td></tr>
        <tr><td>ROM</td><td>${gpu.version.rom}</td></tr>
        <tr><td>Driver</td><td>${gpu.version.driver}</td></tr>
        <tr><td>PCI</td><td>Gen.${gpu.pci.link} x${gpu.pci.width}</td></tr>
        <tr><td>Memory</td><td>${gpu.memory.used}Mb / ${gpu.memory.total}Mb</td></tr>
        <tr><td>Clock</td><td>${gpu.clock.gpu[0]}Mhz / ${gpu.clock.gpu[1]}Mhz</td></tr>
        <tr><td>Power</td><td>${gpu.power[0]}W / ${gpu.power[1]}W</td></tr>
        <tr><td>Load GPU</td><td>${gpu.load.gpu}%</td></tr>
        <tr><td>Load Memory</td><td>${gpu.load.memory}%</td></tr>
        <tr><td>Temperature</td><td>${gpu.load.temp}°C</td></tr>
        <tr><td>Fans</td><td>${gpu.load.fan}%</td></tr>
        <tr><td>State</td><td>${gpu.state}</td></tr>
      `;
      nvmlTbody.innerHTML = rows;
    }
    nvmlEl.style.display = 'block';
  } catch (e) {
    clearInterval(nvmlInterval);
    nvmlEl.style.display = 'none';
  }
}

async function initNVML() {
  nvmlEl = document.getElementById('nvml');
  if (!nvmlEl) {
    nvmlEl = document.createElement('div');
    nvmlEl.className = 'nvml';
    nvmlEl.id = 'nvml';
    nvmlTable = document.createElement('table');
    nvmlTable.className = 'nvml-table';
    nvmlTable.id = 'nvml-table';
    nvmlTable.innerHTML = `
      <thead><tr><th></th><th></th></tr></thead>
      <tbody></tbody>
    `;
    nvmlEl.appendChild(nvmlTable);
    gradioApp().appendChild(nvmlEl);
    log('initNVML');
  }
  nvmlInterval = setInterval(updateNVML, 1000);
}

async function disableNVML() {
  clearInterval(nvmlInterval);
  nvmlEl.style.display = 'none';
}
