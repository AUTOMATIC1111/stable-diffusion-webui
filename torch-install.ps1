$scriptDirectory = $PSScriptRoot
Set-Location $scriptDirectory

## modify webui-user.bat
$filePath = $pwd.Path + "\webui-user.bat"

$newContent = @"

@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--skip-torch-cuda-test --precision full --no-half --skip-prepare-environment
set PYTORCH_TRACING_MODE=TORCHFX

call webui.bat

"@
$newContent | Set-Content -Path $filePath







### modify eval_frame

$eval_filePath = $pwd.Path + "\venv\Lib\site-packages\torch\_dynamo\eval_frame.py"



#comment out the two lines to test torch.compile on windows
$replacements = @{
    "    if sys.platform == `"win32`":" = "#    if sys.platform == `"win32`":"
    "        raise RuntimeError(`"Windows not yet supported for torch.compile`")" = "#        raise RuntimeError(`"Windows not yet supported for torch.compile`")"
}


$lines = Get-Content -Path $eval_filePath

foreach ($search_Text in $replacements.Keys){
    $replaceText = $replacements[$search_text]
    $lines = $lines.Replace($search_Text , $replaceText)
}


#write content back to file
$lines | Set-Content -Path $eval_filePath

