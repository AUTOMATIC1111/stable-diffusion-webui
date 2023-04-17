// Stable Diffusion WebUI - Bracket checker
// Version 1.0
// By Hingashi no Florin/Bwin4L
// Counts open and closed brackets (round, square, curly) in the prompt and negative prompt text boxes in the txt2img and img2img tabs.
// If there's a mismatch, the keyword counter turns red and if you hover on it, a tooltip tells you what's wrong.

function checkBrackets(evt, textArea, counterElt) {
  errorStringParen  = '(...) - Different number of opening and closing parentheses detected.\n';
  errorStringSquare = '[...] - Different number of opening and closing square brackets detected.\n';
  errorStringCurly  = '{...} - Different number of opening and closing curly brackets detected.\n';

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
    counterElt.classList.add('error');
  } else {
    counterElt.classList.remove('error');
  }
}

function setupBracketChecking(id_prompt, id_counter){
    var textarea = gradioApp().querySelector("#" + id_prompt + " > label > textarea");
    var counter = gradioApp().getElementById(id_counter)

    textarea.addEventListener("input", function(evt){
        checkBrackets(evt, textarea, counter)
    });
}

onUiLoaded(function(){
    setupBracketChecking('txt2img_prompt', 'txt2img_token_counter')
    setupBracketChecking('txt2img_neg_prompt', 'txt2img_negative_token_counter')
    setupBracketChecking('img2img_prompt', 'img2img_token_counter')
    setupBracketChecking('img2img_neg_prompt', 'img2img_negative_token_counter')
})