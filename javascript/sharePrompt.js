(function (root, factory) {
  if (typeof module === "object" && typeof module.exports === "object") {
    module.exports = factory();
  } else if (typeof define === "function" && define.amd) {
    define(factory());
  } else if (typeof define === "function" && define.cmd) {
    define(function (require, exports, module) {
      module.exports = factory();
    });
  } else {
    root["sharePrompt"] = factory();
  }
})(this, function () {
  var floatDashboardStyleHTML = `
  .float-dashboard {
    position: fixed;
    left: 8vw;
    top: 0.5em;
    width: 80vw;
    min-height: 6em;
    padding: 0.5em 1vw;
    box-shadow: 0 0 3px 0 #000;
    border-radius: 0.5em;
    background: #ffffff;
    z-index: 300; /* >= 'Stable Diffusion checkpoint' label */
    color: #333;
  }
  .float-dashboard a {
    text-decoration: none;
    color: #333;
  }
  .float-dashboard a:hover {
    color: #000;
    box-shadow: 0 0 1px 0 #666;
  }
  .float-dashboard a:active {
    color: #000;
    box-shadow: 0 0 0 1px #ccc;
    background: #f9f9f9;
  }

  .float-dashboard > #close {
    position: absolute;
    right: -0.2em;
    top: -0.2em;
    border-radius: 0.8em;
  }

  .float-dashboard > #mask {
    display: none;
  }
  .float-dashboard.mask > #mask {
    display: block;
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background: #eee;
    opacity: 0.8;
    color: #666;
    text-align: center;
  }

  .float-dashboard > #head {
    padding-bottom: 0.35em;
    border-bottom: 1px solid #eee;
    margin-bottom: 0.5em;
  }
  .float-dashboard > #head>#label { 
    margin-right: 0.5em;
    text-decoration: underline;
  }

  .float-dashboard .tiny-button {
    padding: 0 0.5em;
    border: 1px solid #666;
    border-radius: 0.8em;
    font-size: 0.6em;
    white-space: nowrap;
  }

  .float-dashboard input.error {
    color: #f00;
  }

  .float-dashboard .flex-h {
    display: flex;
    flex-wrap: wrap;
  }
  `;

  /**
   * create float dashboard
   * @param {string} id
   * @param {string} label
   * @param {boolean} hideIfBlur
   */
  function craeteFloatDashboard(id, label, hideIfBlur) {
    // create and append stylesheet
    var styleId = "floatDashboardStylesheet";
    if (document.querySelector("#" + styleId) == null) {
      var $style = document.createElement("style");
      $style.id = styleId;
      $style.innerHTML = floatDashboardStyleHTML;
      document.head.appendChild($style);
    }

    // create dashboard
    var $board = document.createElement("div");
    $board.id = id;
    $board.classList.add("float-dashboard");

    var showing = false;
    function show(event) {
      hideIfBlur && event.stopPropagation();
      if (showing) return;
      showing = true;
      document.body.appendChild($board);
    }
    function hide() {
      if (!showing) return;
      $board.remove();
      showing = false;
    }
    // append close button
    $board.appendChild(
      (function () {
        var $close = document.createElement("a");
        $close.id = "close";
        $close.href = "javascript:void(0);";
        $close.innerHTML = "×";
        $close.style.fontSize = "1.5em";
        $close.style.padding = "0 0.3em";
        $close.addEventListener("click", hide);
        return $close;
      })()
    );
    if (hideIfBlur) {
      // use the `mouseup` instead of `click` to avoid the custom trigger event
      document.addEventListener("mouseup", hide);
      $board.addEventListener("mouseup", function (event) {
        event.stopPropagation();
      });
    }

    // create and append mask
    var $mask = document.createElement("div");
    $mask.id = "mask";
    $board.appendChild($mask);
    function wait(inform) {
      if ($board.classList.contains("mask")) return;
      $mask.innerHTML = "<h3>" + inform + "</h3>";
      $board.classList.add("mask");
      return function () {
        $board.classList.remove("mask");
      };
    }

    var $head = document.createElement("div");
    $head.id = "head";
    $head.innerHTML = '<span id="label">' + label + "</span>";
    $board.appendChild($head);

    var $body = document.createElement("div");
    $body.id = "body";
    $board.appendChild($body);

    return {
      show: show,
      hide: hide,
      wait: wait,
      $head: $head,
      $body: $body,
    };
  }

  function createSimpleTextInput(title, placeholder, onChange) {
    var $input = document.createElement("input");
    $input.title = title;
    $input.placeholder = placeholder;
    $input.type = "text";
    $input.style.margin = "0 0.5em";
    $input.style.flexGrow = "1";
    onChange && $input.addEventListener("change", onChange);
    return $input;
  }

  function createTinyButton(title, innerHTML, onClick) {
    var $load = document.createElement("a");
    $load.classList.add("tiny-button");
    $load.title = title;
    $load.href = "javascript:void(0);";
    $load.innerHTML = innerHTML;
    $load.addEventListener("click", onClick);
    return $load;
  }

  function createDividerV() {
    var $i = document.createElement("i");
    $i.style.borderRight = "1px solid #666";
    $i.style.margin = "0 0.5em";
    return $i;
  }

  var state = (function () {
    /** @typedef {{ attr: "+" | "-", keys: string[], desc: string, note: string }} Tag */
    /** @typedef {{ style: string, comment: string }} Preset */
    var storeKey = "sharePromptState";
    var initState = '{"localTags":[],"prompt-tags":{},"preset-styles":{}}';
    /** @type {{ localTags: Tag[], "prompt-tags": { [group: string]: Tag[] }, "preset-styles": { [group: string]: Preset[] } }} */
    var state = JSON.parse(localStorage.getItem(storeKey) || initState);

    function empty() {
      return localStorage.getItem(storeKey) == null;
    }

    var activedMethods;
    function active(methods) {
      activedMethods = methods;
    }

    function parseTag(/** @type {string} */ line) {
      var matched = line.match(/^([\+-])\s([^:]+)(.*)/);
      if (matched == null) return;
      var attr = matched[1];
      var keys = matched[2]
        .toLowerCase()
        .split(" ")
        .flatMap(function (key) {
          return key.split("-");
        })
        .filter(function (key) {
          return key !== "";
        });
      var supplement = matched[3].match(/:([^#]+)(.*)/);
      if (!supplement) return { attr, keys, desc: "", note: "" };
      var desc = supplement[1].trim();
      var note = supplement[2].substring(1).trim();
      return { attr, keys, desc, note };
    }

    function stringifyTag(/** @type {Tag} */ tag) {
      var key = tag.attr + " " + tag.keys.join(" ");
      if (tag.desc) key += " : " + tag.desc;
      if (tag.note) key += " # " + tag.note;
      return key;
    }

    var renderDom = (function () {
      var $container = document.createElement("div");
      $container.id = "sharePrompt";
      var $style = document.createElement("style");
      $style.id = "sharePromptStyleSheet";
      $style.innerHTML = `
      #sharePrompt .title {
        color: #666;
        margin: 0 0.5em;
        border-bottom: 1px solid #ccc;
      }
      #sharePrompt .panel {
        min-height: 1em;
        margin-bottom: 0.5em;
      }
      #sharePrompt .tiny-button {
        margin: 0 0.35em;
      }

      #sharePrompt #prompts>*,
      #sharePrompt #presets>*,
      #sharePrompt #local>* {
        margin: 0 0.5em;
        white-space: nowrap;
      }
      #sharePrompt #prompts>*::before,
      #sharePrompt #presets>*::before,
      #sharePrompt #local>*::before {
        content: "●";
        margin: 0 0.3em;
        color: #666;
      }
      `;
      $container.appendChild($style);

      function appendGroupTitle(title) {
        var $bar = document.createElement("div");
        $bar.classList.add("flex-h");
        $bar.innerHTML = '<span class="title">' + title + ": </span>";
        $container.appendChild($bar);
        var $slot = document.createElement("span");
        $slot.classList.add("flex-h");
        $slot.style.flexGrow = 1;
        $bar.appendChild($slot);
        return $slot;
      }

      function appendGroupPanel(id) {
        var $panel = document.createElement("div");
        $panel.id = id;
        $panel.classList.add("panel");
        $container.appendChild($panel);
        return $panel;
      }

      var $promptsTitle = appendGroupTitle("prompt tags");
      var $promptsPanel = appendGroupPanel("prompts");
      var $presetsTitle = appendGroupTitle("preset styles");
      var $presetsPanel = appendGroupPanel("presets");
      var $localTitle = appendGroupTitle("local tags");
      var $localPanel = appendGroupPanel("local");

      function renderGroup(group, $title, $panel, eachHtml, eachClick) {
        $title.innerHTML = "";
        $panel.innerHTML = "";
        var selectedGroup = null;
        Object.keys(group).forEach(function (key) {
          $title.appendChild(
            createTinyButton(key, key, function () {
              $panel.innerHTML = "";
              if (selectedGroup === key) {
                selectedGroup = null;
                return;
              }
              selectedGroup = key;
              var values = group[key];
              values.forEach(function (value) {
                var $item = document.createElement("a");
                $item.href = "javascript:void(0);";
                $item.innerHTML = eachHtml(value);
                $item.addEventListener("click", function () {
                  eachClick(value);
                });
                $panel.appendChild($item);
              });
            })
          );
        });
      }

      function renderPromptTag(/** @type {Tag} */ tag) {
        var key = tag.keys.join(" ");
        return (
          '<span title="' +
          (tag.note || tag.desc || key) +
          '">' +
          "<small>" +
          (tag.desc ? key + ":" : key) +
          "</small>" +
          tag.desc +
          "</span>"
        );
      }
      function onPromptTagClick(/** @type {Tag} */ tag) {
        var key = tag.keys.join(" ");
        navigator.clipboard.writeText(key);
        tag.attr === "+"
          ? activedMethods.addPositive(key)
          : activedMethods.addNegative(key);
      }

      function renderPresetStyle(/** @type {Preset} */ preset) {
        return (
          '<span title="' + preset.style + '">' + preset.comment + "</span>"
        );
      }
      function onPresetStyleClick(/** @type {Preset} */ preset) {
        navigator.clipboard.writeText(preset.style);
        activedMethods.parsePrompt(preset.style);
      }

      function appendLocalTag(/** @type {Tag} */ tag) {
        var $item = document.createElement("a");
        $item.href = "javascript:void(0);";
        $item.innerHTML = renderPromptTag(tag);
        $item.addEventListener("click", function () {
          onPromptTagClick(tag);
        });
        $localPanel.appendChild($item);
      }

      $localTitle.appendChild(
        (function () {
          var $input = createSimpleTextInput(
            "custom prompt tag",
            "+/- custom prompt tag : desc (# note)"
          );
          $input.addEventListener("change", function () {
            var value = $input.value.trim();
            if (!value) {
              $input.classList.remove("error");
              return;
            }
            var tag = parseTag(value);
            if (tag == null) {
              $input.classList.add("error");
              return;
            }
            $input.classList.remove("error");
            $input.value = stringifyTag(tag);

            appendLocalTag(tag);
            state.localTags.push(tag);
            localStorage.setItem(storeKey, JSON.stringify(state));
          });
          return $input;
        })()
      );

      function render() {
        renderGroup(
          state["prompt-tags"],
          $promptsTitle,
          $promptsPanel,
          renderPromptTag,
          onPromptTagClick
        );
        renderGroup(
          state["preset-styles"],
          $presetsTitle,
          $presetsPanel,
          renderPresetStyle,
          onPresetStyleClick
        );
        $localPanel.innerHTML = "";
        state.localTags.forEach(appendLocalTag);
        return $container;
      }
      return render;
    })();

    var sectionParser = {
      "prompt-tags": function (/** @type {string[]} */ lines, offset) {
        var i = offset + 1;
        /** @type {Tag[]}*/ var values = [];
        var emptyLineCount = 0;
        for (; i < lines.length; i++) {
          var line = lines[i].trim();
          if (!line) emptyLineCount++;
          if (emptyLineCount >= 2) break;
          var tag = parseTag(line);
          if (tag == null) break;
          values.push(tag);
        }
        return { offset: i - 1, values: values };
      },
      "preset-styles": function (/** @type {string[]} */ lines, offset) {
        var i = offset + 1;
        /** @type {Preset[]} */ var values = [];
        for (; i < lines.length; i++) {
          var line = lines[i].trim();
          if (line !== "``` style") continue;
          var style = [];
          i++;
          for (; i < lines.length; i++) {
            var line = lines[i];
            if (line === "```") {
              break;
            } else {
              style.push(line);
            }
          }
          i++;
          var comment = [];
          for (; i < lines.length; i++) {
            var line = lines[i].trim();
            if (line.startsWith("> ")) comment.push(line.slice(2));
            else if (line != "") comment.push(line);
            else break;
          }
          values.push({ style: style.join("\n"), comment: comment.join("\n") });
          i--;
        }
        return { offset: i - 1, values: values };
      },
    };

    function parseMarkdown(/** @type {string} */ text) {
      var lines = text.split("\n");
      for (var i = 0; i < lines.length; i++) {
        var line = lines[i];
        var matched = line.match(/^### \[([\w-]+)\]\s(\S+)/);
        if (matched == null) continue;

        var sectionType = matched[1];
        var sectionLabel = matched[2];
        var parser = sectionParser[sectionType];
        if (!parser) continue;

        var section = parser(lines, i);
        console.log(
          "load " + sectionType + " group: " + sectionLabel,
          section.values
        );
        i = section.offset;
        state[sectionType][sectionLabel] = section.values;
      }
      return state;
    }

    function load(/** @type {string} */ text) {
      state = JSON.parse(initState); // clear state
      if (text.startsWith("{")) {
        var json = JSON.parse(text);
        if ("files" in json) {
          // https://api.github.com/gists/...
          Object.values(json.files).forEach(function (file) {
            if (file.language === "Markdown") parseMarkdown(file.content);
          });
        } else {
          // clipboard
          state.localTags = json.localTags || [];
          state["prompt-tags"] = json["prompt-tags"] || {};
          state["preset-styles"] = json["preset-styles"] || {};
        }
      } else {
        parseMarkdown(text);
      }
      localStorage.setItem(storeKey, JSON.stringify(state));
      renderDom();
    }

    function dumpMarkdown() {
      var text = "> " + location.href + "\n";
      var prompts = state["prompt-tags"];
      Object.keys(prompts).forEach(function (group) {
        text += "\n### [prompt-tags] " + group + "\n";
        prompts[group].forEach(function (tag) {
          text += stringifyTag(tag) + "\n";
        });
      });
      var presets = state["preset-styles"];
      Object.keys(presets).forEach(function (group) {
        text += "\n### [preset-styles] " + group + "\n";
        presets[group].forEach(function (preset) {
          text +=
            "``` style" + preset.style + "\n```\n> " + preset.comment + "\n";
        });
      });
      return text;
    }

    function dump() {
      var $win = window.open();
      $win.navigator.clipboard.writeText(JSON.stringify(state, undefined, 2));

      $win.document.writeln(
        '<textarea id="dump" style="width: 100%; height: 100%;">' +
          dumpMarkdown() +
          "</textarea>"
      );

      var $title = document.createElement("title");
      $title.innerHTML = "shrea prompt - dump";
      $win.document.head.appendChild($title);
    }
    return {
      empty: empty,
      active: active,
      load: load,
      dump: dump,
      $dom: renderDom(),
    };
  })();

  document.addEventListener("DOMContentLoaded", function () {
    //#region dashboard
    var dashboard = craeteFloatDashboard(
      "sharePromptDashboard",
      "Share prompt",
      true
    );

    dashboard.$head.classList.add("flex-h");
    dashboard.$head.appendChild(
      createSimpleTextInput(
        "search prompts(tags) from danbooru",
        "Search prompt(tags) from danbooru",
        function (event) {
          var value = event.target.value.trim();
          if (value) {
            var $win = window.open(
              "https://danbooru.donmai.us/autocomplete?search[query]=" +
                value +
                "&search[type]=tag&version=1&limit=20",
              "danbooru",
              "height=480,width=360,menubar=no,toolbar=no,status=no"
            );
          }
        }
      )
    );
    dashboard.$head.appendChild(
      (function () {
        function load(url) {
          var done = dashboard.wait("L O A D I N G . . .");
          if (done == null) return;
          var loadFrom = url ? url : "clipboard";
          return (
            url
              ? fetch(url).then(function (r) {
                  localStorage.setItem(urlStoreKey, url);
                  return r.text();
                })
              : navigator.clipboard.readText()
          )
            .then(state.load)
            .then(function () {
              console.info("loaded prompts from " + loadFrom);
            })
            .catch(function (error) {
              console.error("failed to load prompts from " + loadFrom, error);
            })
            .then(done);
        }
        var urlStoreKey = "sharePromptUrl";
        var $input = createSimpleTextInput(
          "input load prompts url, or load from clipboard if no value present",
          "Load prompts url (https://path/to/prompts/json)",
          function (event) {
            load(event.target.value.trim());
          }
        );
        var url = localStorage.getItem(urlStoreKey);
        if (url == null) {
          // find in configuration
          var url = ""; 
          if (!url) {
            // find in location search, for example: ?share=https://api.github.com/gists/ccf3fc8a51b1bcce9709b237fa860e01
            var matched = location.search.match(/[\?&]share=([\S^\?^&]+)/);
            matched && matched[1] && (url = matched[1]);
          }
          // initial local storage
          url &&
            state.empty() &&
            load(url).then(function () {
              localStorage.setItem(urlStoreKey, "");
            });
        } else {
          $input.defaultValue = url;
        }
        return $input;
      })()
    );
    dashboard.$head.appendChild(createDividerV());
    dashboard.$head.appendChild(
      (function () {
        var $dump = createTinyButton(
          "dump prompts to json",
          "Dump",
          state.dump
        );
        $dump.style.marginRight = "1.5em";
        return $dump;
      })()
    );

    dashboard.$body.appendChild(state.$dom);
    //#endregion

    var entrypointButtonId = "sharePromptEntrypoint";
    // create entrypoint button
    function createEntrypointButton(activedMethods) {
      var $button = document.createElement("button");
      $button.id = entrypointButtonId;
      $button.classList.add("gr-button", "gr-button-lg", "gr-button-secondary");
      $button.style.maxWidth = "2.5em";
      $button.style.minWidth = "2.5em";
      $button.style.height = "2.4em";
      $button.innerHTML = "\u{1f4da}"; // '📚' // '\u2601';  // '☁'
      $button.title = "browse share-prompt";
      $button.addEventListener("click", function (event) {
        state.active(activedMethods);
        dashboard.show(event);
      });
      return $button;
    }

    function createPromptActiveMethods(/** @type {HTMLDivElement} */ $app) {
      var findVisibilityDom = (function () {
        var $cols = $app.querySelectorAll("#roll_col");
        var $doms = Array.prototype.map.call($cols, function ($col) {
          var $paste = $col.querySelector("#paste");
          var $textareas =
            $col.parentNode.firstElementChild.querySelectorAll("textarea");
          return {
            $paste: $paste,
            $positive: $textareas[0],
            $negative: $textareas[1],
          };
        });
        function findVisibilityDom() {
          for (let index = 0; index < $cols.length; index++) {
            const $col = $cols[index];
            if ($col.offsetWidth > 0 || $col.offsetHeight > 0) {
              return $doms[index];
            }
          }
        }
        return findVisibilityDom;
      })();

      function changeTextareaValue($target, value) {
        $target.value = value;
        var event = new InputEvent("input", { bubbles: true });
        Object.defineProperty(event, "target", {
          writable: false,
          value: $target,
        });
        $target.dispatchEvent(event);
      }
      function addPositive(tag) {
        var $dom = findVisibilityDom();
        var $textarea = $dom.$positive;
        changeTextareaValue($textarea, tag + ", " + $textarea.value);
      }
      function addNegative(tag) {
        var $dom = findVisibilityDom();
        var $textarea = $dom.$negative;
        changeTextareaValue($textarea, tag + ", " + $textarea.value);
      }
      function parsePrompt(prompt) {
        var $dom = findVisibilityDom();
        var $textarea = $dom.$positive;
        changeTextareaValue($textarea, prompt);
        $dom.$paste.click();
      }
      return { addPositive, addNegative, parsePrompt };
    }

    // append entrypoint button into gradio app
    onUiUpdate(function () {
      /** @type {ShadowRoot} */
      var $app = gradioApp();
      if ($app.querySelector("#" + entrypointButtonId) != null) return;
      var $quicksettings = $app.querySelector("#quicksettings");
      if ($quicksettings == null) return;
      var methods = createPromptActiveMethods($app);
      if (methods == null) {
        console.error("dom structure has changed.");
        return;
      }
      var $button = createEntrypointButton(methods);
      $quicksettings.appendChild($button);
    });
  });

  return {
    dom: {
      craeteFloatDashboard,
      createSimpleTextInput,
      createTinyButton,
      createDividerV,
    },
  };
});
