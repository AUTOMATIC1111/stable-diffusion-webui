// Stable Diffusion WebUI - Bracket checker
// Version 1.0
// By Hingashi no Florin/Bwin4L
// Counts open and closed brackets (round, square, curly) in the prompt and negative prompt text boxes in the txt2img and img2img tabs.
// If there's a mismatch, the keyword counter turns red and if you hover on it, a tooltip tells you what's wrong.

function checkBrackets(evt) {
  textArea = evt.target;
  tabName = evt.target.parentElement.parentElement.id.split("_")[0];
  counterElt = document.querySelector('gradio-app').shadowRoot.querySelector('#' + tabName + '_token_counter');

  promptName = evt.target.parentElement.parentElement.id.includes('neg') ? ' negative' : '';

  errorStringParen  = '(' + tabName + promptName + ' prompt) - Different number of opening and closing parentheses detected.\n';
  errorStringSquare = '[' + tabName + promptName + ' prompt] - Different number of opening and closing square brackets detected.\n';
  errorStringCurly  = '{' + tabName + promptName + ' prompt} - Different number of opening and closing curly brackets detected.\n';

  openBracketRegExp = /\(/g;
  closeBracketRegExp = /\)/g;

  openSquareBracketRegExp = /\[/g;
  closeSquareBracketRegExp = /\]/g;

  openCurlyBracketRegExp = /\{/g;
  closeCurlyBracketRegExp = /\}/g;

  totalOpenBracketMatches = 0;
  totalCloseBracketMatches = 0;
  totalOpenSquareBracketMatches = 0;
  totalCloseSquareBracketMatches = 0;
  totalOpenCurlyBracketMatches = 0;
  totalCloseCurlyBracketMatches = 0;

  openBracketMatches = textArea.value.match(openBracketRegExp);
  if(openBracketMatches) {
    totalOpenBracketMatches = openBracketMatches.length;
  }

  closeBracketMatches = textArea.value.match(closeBracketRegExp);
  if(closeBracketMatches) {
    totalCloseBracketMatches = closeBracketMatches.length;
  }

  openSquareBracketMatches = textArea.value.match(openSquareBracketRegExp);
  if(openSquareBracketMatches) {
    totalOpenSquareBracketMatches = openSquareBracketMatches.length;
  }

  closeSquareBracketMatches = textArea.value.match(closeSquareBracketRegExp);
  if(closeSquareBracketMatches) {
    totalCloseSquareBracketMatches = closeSquareBracketMatches.length;
  }

  openCurlyBracketMatches = textArea.value.match(openCurlyBracketRegExp);
  if(openCurlyBracketMatches) {
    totalOpenCurlyBracketMatches = openCurlyBracketMatches.length;
  }

  closeCurlyBracketMatches = textArea.value.match(closeCurlyBracketRegExp);
  if(closeCurlyBracketMatches) {
    totalCloseCurlyBracketMatches = closeCurlyBracketMatches.length;
  }

  if(totalOpenBracketMatches != totalCloseBracketMatches) {
    if(!counterElt.title.includes(errorStringParen)) {
      counterElt.title += errorStringParen;
    }
  } else {
    counterElt.title = counterElt.title.replace(errorStringParen, '');
  }

  if(totalOpenSquareBracketMatches != totalCloseSquareBracketMatches) {
    if(!counterElt.title.includes(errorStringSquare)) {
      counterElt.title += errorStringSquare;
    }
  } else {
    counterElt.title = counterElt.title.replace(errorStringSquare, '');
  }

  if(totalOpenCurlyBracketMatches != totalCloseCurlyBracketMatches) {
    if(!counterElt.title.includes(errorStringCurly)) {
      counterElt.title += errorStringCurly;
    }
  } else {
    counterElt.title = counterElt.title.replace(errorStringCurly, '');
  }

  if(counterElt.title != '') {
    counterElt.style = 'color: #FF5555;';
  } else {
    counterElt.style = '';
  }
}

var shadowRootLoaded = setInterval(function() {
  var shadowTextArea = document.querySelector('gradio-app').shadowRoot.querySelectorAll('#txt2img_prompt > label > textarea');
  if(shadowTextArea.length < 1) {
      return false;
  }

  clearInterval(shadowRootLoaded);

  document.querySelector('gradio-app').shadowRoot.querySelector('#txt2img_prompt').onkeyup = checkBrackets;
  document.querySelector('gradio-app').shadowRoot.querySelector('#txt2img_neg_prompt').onkeyup = checkBrackets;
  document.querySelector('gradio-app').shadowRoot.querySelector('#img2img_prompt').onkeyup = checkBrackets;
  document.querySelector('gradio-app').shadowRoot.querySelector('#img2img_neg_prompt').onkeyup = checkBrackets;
}, 1000);
