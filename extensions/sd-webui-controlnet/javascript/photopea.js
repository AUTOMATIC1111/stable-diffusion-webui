(function () {
  /*
  MIT LICENSE
  Copyright 2011 Jon Leighton
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
  associated documentation files (the "Software"), to deal in the Software without restriction,
  including without limitation the rights to use, copy, modify, merge, publish, distribute, 
  sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial
  portions of the Software. 
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
  PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
  */
  // From: https://gist.github.com/jonleighton/958841
  function base64ArrayBuffer(arrayBuffer) {
    var base64 = ''
    var encodings = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

    var bytes = new Uint8Array(arrayBuffer)
    var byteLength = bytes.byteLength
    var byteRemainder = byteLength % 3
    var mainLength = byteLength - byteRemainder

    var a, b, c, d
    var chunk

    // Main loop deals with bytes in chunks of 3
    for (var i = 0; i < mainLength; i = i + 3) {
      // Combine the three bytes into a single integer
      chunk = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2]

      // Use bitmasks to extract 6-bit segments from the triplet
      a = (chunk & 16515072) >> 18 // 16515072 = (2^6 - 1) << 18
      b = (chunk & 258048) >> 12 // 258048   = (2^6 - 1) << 12
      c = (chunk & 4032) >> 6 // 4032     = (2^6 - 1) << 6
      d = chunk & 63               // 63       = 2^6 - 1

      // Convert the raw binary segments to the appropriate ASCII encoding
      base64 += encodings[a] + encodings[b] + encodings[c] + encodings[d]
    }

    // Deal with the remaining bytes and padding
    if (byteRemainder == 1) {
      chunk = bytes[mainLength]

      a = (chunk & 252) >> 2 // 252 = (2^6 - 1) << 2

      // Set the 4 least significant bits to zero
      b = (chunk & 3) << 4 // 3   = 2^2 - 1

      base64 += encodings[a] + encodings[b] + '=='
    } else if (byteRemainder == 2) {
      chunk = (bytes[mainLength] << 8) | bytes[mainLength + 1]

      a = (chunk & 64512) >> 10 // 64512 = (2^6 - 1) << 10
      b = (chunk & 1008) >> 4 // 1008  = (2^6 - 1) << 4

      // Set the 2 least significant bits to zero
      c = (chunk & 15) << 2 // 15    = 2^4 - 1

      base64 += encodings[a] + encodings[b] + encodings[c] + '='
    }

    return base64
  }

  // Turn a base64 string into a blob. 
  // From https://gist.github.com/gauravmehla/7a7dfd87dd7d1b13697b6e894426615f
  function b64toBlob(b64Data, contentType, sliceSize) {
    var contentType = contentType || '';
    var sliceSize = sliceSize || 512;
    var byteCharacters = atob(b64Data);
    var byteArrays = [];
    for (var offset = 0; offset < byteCharacters.length; offset += sliceSize) {
      var slice = byteCharacters.slice(offset, offset + sliceSize);
      var byteNumbers = new Array(slice.length);
      for (var i = 0; i < slice.length; i++) {
        byteNumbers[i] = slice.charCodeAt(i);
      }
      var byteArray = new Uint8Array(byteNumbers);
      byteArrays.push(byteArray);
    }
    return new Blob(byteArrays, { type: contentType });
  }

  function createBlackImageBase64(width, height) {
    // Create a canvas element
    var canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;

    // Get the context of the canvas
    var ctx = canvas.getContext('2d');

    // Fill the canvas with black color
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, width, height);

    // Get the base64 encoded string
    var base64Image = canvas.toDataURL('image/png');

    return base64Image;
  }

  // Functions to be called within photopea context.
  // Start of photopea functions
  function pasteImage(base64image) {
    app.open(base64image, null, /* asSmart */ true);
    app.echoToOE("success");
  }

  function setLayerNames(names) {
    const layers = app.activeDocument.layers;
    if (layers.length !== names.length) {
      console.error("layer length does not match names length");
      echoToOE("error");
      return;
    }

    for (let i = 0; i < names.length; i++) {
      const layer = layers[i];
      layer.name = names[i];
    }
    app.echoToOE("success");
  }

  function removeLayersWithNames(names) {
    const layers = app.activeDocument.layers;
    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      if (names.includes(layer.name)) {
        layer.remove();
      }
    }
    app.echoToOE("success");
  }

  function getAllLayerNames() {
    const layers = app.activeDocument.layers;
    const names = [];
    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      names.push(layer.name);
    }
    app.echoToOE(JSON.stringify(names));
  }

  // Hides all layers except the current one, outputs the whole image, then restores the previous
  // layers state.
  function exportSelectedLayerOnly(format, layerName) {
    // Gets all layers recursively, including the ones inside folders.
    function getAllArtLayers(document) {
      let allArtLayers = [];

      for (let i = 0; i < document.layers.length; i++) {
        const currentLayer = document.layers[i];
        allArtLayers.push(currentLayer);
        if (currentLayer.typename === "LayerSet") {
          allArtLayers = allArtLayers.concat(getAllArtLayers(currentLayer));
        }
      }
      return allArtLayers;
    }

    function makeLayerVisible(layer) {
      let currentLayer = layer;
      while (currentLayer != app.activeDocument) {
        currentLayer.visible = true;
        if (currentLayer.parent.typename != 'Document') {
          currentLayer = currentLayer.parent;
        } else {
          break;
        }
      }
    }


    const allLayers = getAllArtLayers(app.activeDocument);
    // Make all layers except the currently selected one invisible, and store
    // their initial state.
    const layerStates = [];
    for (let i = 0; i < allLayers.length; i++) {
      const layer = allLayers[i];
      layerStates.push(layer.visible);
    }
    // Hide all layers to begin with
    for (let i = 0; i < allLayers.length; i++) {
      const layer = allLayers[i];
      layer.visible = false;
    }
    for (let i = 0; i < allLayers.length; i++) {
      const layer = allLayers[i];
      const selected = layer.name === layerName;
      if (selected) {
        makeLayerVisible(layer);
      }
    }
    app.activeDocument.saveToOE(format);

    for (let i = 0; i < allLayers.length; i++) {
      const layer = allLayers[i];
      layer.visible = layerStates[i];
    }
  }

  function hasActiveDocument() {
    app.echoToOE(app.documents.length > 0 ? "true" : "false");
  }
  // End of photopea functions

  const MESSAGE_END_ACK = "done";
  const MESSAGE_ERROR = "error";
  const PHOTOPEA_URL = "https://www.photopea.com/";
  class PhotopeaContext {
    constructor(photopeaIframe) {
      this.photopeaIframe = photopeaIframe;
      this.timeout = 1000;
    }

    navigateIframe() {
      const iframe = this.photopeaIframe;
      const editorURL = PHOTOPEA_URL;

      return new Promise(async (resolve) => {
        if (iframe.src !== editorURL) {
          iframe.src = editorURL;
          // Stop waiting after 10s.
          setTimeout(resolve, 10000);

          // Testing whether photopea is able to accept message.
          while (true) {
            try {
              await this.invoke(hasActiveDocument);
              break;
            } catch (e) {
              console.log("Keep waiting for photopea to accept message.");
            }
          }
          this.timeout = 5000; // Restore to a longer timeout in normal messaging.
        }
        resolve();
      });
    }

    // From https://github.com/huchenlei/stable-diffusion-ps-pea/blob/main/src/Photopea.ts
    postMessageToPhotopea(message) {
      return new Promise((resolve, reject) => {
        const responseDataPieces = [];
        let hasError = false;
        const photopeaMessageHandle = (event) => {
          if (event.source !== this.photopeaIframe.contentWindow) {
            return;
          }
          // Filter out the ping messages
          if (typeof event.data === 'string' && event.data.includes('MSFAPI#')) {
            return;
          }
          // Ignore "done" when no data has been received. The "done" can come from
          // MSFAPI ping.
          if (event.data === MESSAGE_END_ACK && responseDataPieces.length === 0) {
            return;
          }
          if (event.data === MESSAGE_END_ACK) {
            window.removeEventListener("message", photopeaMessageHandle);
            if (hasError) {
              reject('Photopea Error.');
            } else {
              resolve(responseDataPieces.length === 1 ? responseDataPieces[0] : responseDataPieces);
            }
          } else if (event.data === MESSAGE_ERROR) {
            responseDataPieces.push(event.data);
            hasError = true;
          } else {
            responseDataPieces.push(event.data);
          }
        };

        window.addEventListener("message", photopeaMessageHandle);
        setTimeout(() => reject("Photopea message timeout"), this.timeout);
        this.photopeaIframe.contentWindow.postMessage(message, "*");
      });
    }

    // From https://github.com/huchenlei/stable-diffusion-ps-pea/blob/main/src/Photopea.ts
    async invoke(func, ...args) {
      await this.navigateIframe();
      const message = `${func.toString()} ${func.name}(${args.map(arg => JSON.stringify(arg)).join(',')});`;
      try {
        return await this.postMessageToPhotopea(message);
      } catch (e) {
        throw `Failed to invoke ${func.name}. ${e}.`;
      }
    }

    /**
     * Fetch detected maps from each ControlNet units. 
     * Create a new photopea document.
     * Add those detected maps to the created document.
     */
    async fetchFromControlNet(tabs) {
      if (tabs.length === 0) return;
      const isImg2Img = tabs[0].querySelector('.cnet-unit-enabled').id.includes('img2img');
      const generationType = isImg2Img ? 'img2img' : 'txt2img';
      const width = gradioApp().querySelector(`#${generationType}_width input[type=number]`).value;
      const height = gradioApp().querySelector(`#${generationType}_height input[type=number]`).value;

      const layerNames = ["background"];
      await this.invoke(pasteImage, createBlackImageBase64(width, height));
      await new Promise(r => setTimeout(r, 200));
      for (const [i, tab] of tabs.entries()) {
        const generatedImage = tab.querySelector('.cnet-generated-image-group .cnet-image img');
        if (!generatedImage) continue;
        await this.invoke(pasteImage, generatedImage.src);
        // Wait 200ms for pasting to fully complete so that we do not ended up with 2 separate
        // documents.
        await new Promise(r => setTimeout(r, 200));
        layerNames.push(`unit-${i}`);
      }
      await this.invoke(removeLayersWithNames, layerNames);
      await this.invoke(setLayerNames, layerNames.reverse());
    }

    /**
     * Send the images in the active photopea document back to each ControlNet units.
     */
    async sendToControlNet(tabs) {
      // Gradio's image widgets are inputs. To set the image in one, we set the image on the input and
      // force it to refresh.
      function setImageOnInput(imageInput, file) {
        // Createa a data transfer element to set as the data in the input.
        const dt = new DataTransfer();
        dt.items.add(file);
        const list = dt.files;

        // Actually set the image in the image widget.
        imageInput.files = list;

        // Foce the image widget to update with the new image, after setting its source files.
        const event = new Event('change', {
          'bubbles': true,
          "composed": true
        });
        imageInput.dispatchEvent(event);
      }

      function sendToControlNetUnit(b64Image, index) {
        const tab = tabs[index];
        // Upload image to output image element.
        const outputImage = tab.querySelector('.cnet-photopea-output');
        const outputImageUpload = outputImage.querySelector('input[type="file"]');
        setImageOnInput(outputImageUpload, new File([b64toBlob(b64Image, "image/png")], "photopea_output.png"));

        // Make sure `UsePreviewAsInput` checkbox is checked.
        const checkbox = tab.querySelector('.cnet-preview-as-input input[type="checkbox"]');
        if (!checkbox.checked) {
          checkbox.click();
        }
      }

      const layerNames =
        JSON.parse(await this.invoke(getAllLayerNames))
          .filter(name => /unit-\d+/.test(name));

      for (const layerName of layerNames) {
        const arrayBuffer = await this.invoke(exportSelectedLayerOnly, 'PNG', layerName);
        const b64Image = base64ArrayBuffer(arrayBuffer);
        const layerIndex = Number.parseInt(layerName.split('-')[1]);
        sendToControlNetUnit(b64Image, layerIndex);
      }
    }
  }

  let photopeaWarningShown = false;

  function firstTimeUserPrompt() {
    if (opts.controlnet_photopea_warning){
      const photopeaPopupMsg = "you are about to connect to https://photopea.com\n" +
        "- Click OK: proceed.\n" +
        "- Click Cancel: abort.\n" +
        "Photopea integration can be disabled in Settings > ControlNet > Disable photopea edit.\n" +
        "This popup can be disabled in Settings > ControlNet > Photopea popup warning.";
      if (photopeaWarningShown || confirm(photopeaPopupMsg)) photopeaWarningShown = true;
      else return false;
    }
    return true;
  }

  const cnetRegisteredAccordions = new Set();
  function loadPhotopea() {
    function registerCallbacks(accordion) {
      const photopeaMainTrigger = accordion.querySelector('.cnet-photopea-main-trigger');
      // Photopea edit feature disabled.
      if (!photopeaMainTrigger) {
        console.log("ControlNet photopea edit disabled.");
        return;
      }

      const closeModalButton = accordion.querySelector('.cnet-photopea-edit .cnet-modal-close');
      const tabs = accordion.querySelectorAll('.cnet-unit-tab');
      const photopeaIframe = accordion.querySelector('.photopea-iframe');
      const photopeaContext = new PhotopeaContext(photopeaIframe, tabs);

      tabs.forEach(tab => {
        const photopeaChildTrigger = tab.querySelector('.cnet-photopea-child-trigger');
        photopeaChildTrigger.addEventListener('click', async () => {
          if (!firstTimeUserPrompt()) return;

          photopeaMainTrigger.click();
          if (await photopeaContext.invoke(hasActiveDocument) === "false") {
            await photopeaContext.fetchFromControlNet(tabs);
          }
        });
      });
      accordion.querySelector('.photopea-fetch').addEventListener('click', () => photopeaContext.fetchFromControlNet(tabs));
      accordion.querySelector('.photopea-send').addEventListener('click', () => {
        photopeaContext.sendToControlNet(tabs)
        closeModalButton.click();
      });
    }

    const accordions = gradioApp().querySelectorAll('#controlnet');
    accordions.forEach(accordion => {
      if (cnetRegisteredAccordions.has(accordion)) return;
      registerCallbacks(accordion);
      cnetRegisteredAccordions.add(accordion);
    });
  }

  onUiUpdate(loadPhotopea);
})();