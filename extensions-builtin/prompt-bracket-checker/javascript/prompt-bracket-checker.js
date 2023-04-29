// Stable Diffusion WebUI - Bracket checker
// By Hingashi no Florin/Bwin4L & @akx
// Counts open and closed brackets (round, square, curly) in the prompt and negative prompt text boxes in the txt2img and img2img tabs.
// If there's a mismatch, the keyword counter turns red and if you hover on it, a tooltip tells you what's wrong.

function checkBrackets(textArea, counterElt) {
  var counts = {};
  (textArea.value.match(/[(){}\[\]]/g) || []).forEach(bracket => {
    counts[bracket] = (counts[bracket] || 0) + 1;
  });
  var errors = [];

  function checkPair(open, close, kind) {
    if (counts[open] !== counts[close]) {
      errors.push(
        `${open}...${close} - Detected ${counts[open] || 0} opening and ${counts[close] || 0} closing ${kind}.`
      );
    }
  }

  checkPair('(', ')', 'round brackets');
  checkPair('[', ']', 'square brackets');
  checkPair('{', '}', 'curly brackets');
  counterElt.title = errors.join('\n');
  counterElt.classList.toggle('error', errors.length !== 0);
}

function setupBracketChecking(id_prompt, id_counter) {
  var textarea = gradioApp().querySelector("#" + id_prompt + " > label > textarea");
  var counter = gradioApp().getElementById(id_counter)

  if (textarea && counter) {
    textarea.addEventListener("input", () => checkBrackets(textarea, counter));
  }
}

onUiLoaded(function () {
  setupBracketChecking('txt2img_prompt', 'txt2img_token_counter');
  setupBracketChecking('txt2img_neg_prompt', 'txt2img_negative_token_counter');
  setupBracketChecking('img2img_prompt', 'img2img_token_counter');
  setupBracketChecking('img2img_neg_prompt', 'img2img_negative_token_counter');
});
