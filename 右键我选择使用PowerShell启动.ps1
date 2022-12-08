function first {
clear
if ("$pwd" -match "\s") {
"检测到路径中含有空格，请移动至无空格路径"
pause
exit
} else {
if ("$pwd" -match "[\u4e00-\u9fa5]") {
"检测到路径中含有中文，请移动至无中文路径"
pause
exit
}
"当前页面提示均为正常内容"
$null = Remove-Item .\python3.10\scripts -recurse
$null = .\python3.10\python.exe get-pip.py
$null = .\git\mingw64\bin\git.exe config--global--add safe.directory "*"
}
clear
"========================================"
"温馨提示："
"遭遇报错请优先查看视频顶置/群内公告/频道报错询问区"
"若询问常见问题或已发布至公告问题将无人回答"
"建议Win10及以上将Windows Terminal设为默认终端"
"部署中的任何选项切勿输入中文与空格！！！"
"========================================”
$Pause1 = Read-Host '请输入"我已知晓上述内容"继续操作'
if ($Pause1 -eq "我已知晓上述内容") {deploying}
"请输入正确内容!"
first
}
function updated {
"========================================"
"检测最新版本中..."
.\Git\mingw64\libexec\git-core\git.exe -C .\deploy pull
if ((Get-Content ".\deploy\.git\refs\heads\main") -ne "c7e1c049dcdbdf1beb71b8d51ddb0a1988429f9e" ) {
"检测到新版本，建议更新！"
Get-Content ".\deploy\README" -encoding UTF8
pause
check
} else {
"当前为最新版"
Start-Sleep -Seconds 2
check
}
}
function deployed {
clear
"========================================"
$Pause2 = Read-Host "检测到已完成部署，请选择运行模式[1.直接启动 2.再次部署 3.模型安装]"
if ($Pause2 -eq "1") {loading}
if ($Pause2 -eq "2") {
$null = Remove-Item .\venv\Scripts -recurse
$null = Remove-Item .\venv\pyvenv.cfg -recurse
deploying
}
if ($Pause2 -eq "3") {clear;"更新模型列表中...";.\Git\mingw64\libexec\git-core\git.exe -C .\tbaimodels pull;pt}
"请正确输入选项!"
deployed
}
function deploying {
clear
"========================================"
"正在启动一键部署程序中"
"========================================"
"开始部署Stable Diffusion WebUI"
"请保证C盘预留5G以上空间"
"========================================"
if (Test-Path "$HOME\.cache") {
$null = Remove-Item $HOME\.cache -recurse
$null = robocopy ".\deploy\.cache" $HOME\.cache /e
} else {
$null = robocopy ".\deploy\.cache" $HOME\.cache /e
}
"已部署完成!"
Start-Sleep -Seconds 2
nvidia-smi
if ($?) {
clear
"========================================"
"检测为N卡，将以GPU模式启动..."
$ARG = "--autolaunch "
Start-Sleep -Seconds 2
options1
} else {
clear
"========================================"
"检测为非N卡，将以CPU模式启动..."
Start-Sleep -Seconds 2
options
}
}
function options {
clear
$Pause3 = Read-Host "为避免检测失误，请确认是否为N卡 [1.是 2.否]"
if ($Pause3 -eq "1") {
$ARG = "--autolaunch "
options1
}
if ($Pause3 -eq "2") {
$ARG = "--skip-torch-cuda-test --lowvram --always-batch-cond-uncond --no-half --autolaunch "
options4
}
"请正确输入选项!"
Start-Sleep -Seconds 1
options
}
function options1 {
clear
"========================================"
$Pause4 = Read-Host "请根据实际情况选择显存[1.3G以下 2.4G显存 3.6G显存 4.8G及以上(训练模型必选)]"
if ($Pause4 -eq "1") {
$ARG = "$ARG--lowvram --always-batch-cond-uncond "
options2
}
if ($Pause4 -eq "2") {
$ARG = "$ARG--lowvram "
options2
}
if ($Pause4 -eq "3") {
$ARG = "$ARG--medvram "
options2
}
if ($Pause4 -eq "4") {
$ARG = "$ARG"
options2
}
"请正确输入选项!"
Start-Sleep -Seconds 1
options1
}
function options2 {
clear
"========================================"
"仅用于支持半精度的显卡，若使用后报错请再次部署关闭此选项！"
"16XX系列显卡请务必关闭！"
$Pause5 = Read-Host "是否开启半精度[Y/N]"
if ($Pause5 -eq "Y") {options3}
if ($Pause5 -eq "N") {
$ARG = "$ARG--no-half "
options3
}
"请正确输入选项!"
Start-Sleep -Seconds 1
options2
}
function options3 {
clear
"========================================"
$Pause6 = Read-Host "是否开启xformers,本选项可提升30%性能[Y/N]"
if ($Pause6 -eq "Y") {
$ARG = "$ARG--xformers "
options4
}
if ($Pause6 -eq "N") {options4}
"请正确输入选项!"
Start-Sleep -Seconds 1
options3
}
function options4 {
clear
"========================================"
$Pause7 = Read-Host "是否开启DeepDanBooru，可用于查询Tag [Y/N]"
if ($Pause7 -eq "Y") {
$ARG = "$ARG--deepdanbooru "
options5
}
if ($Pause7 -eq "N") {options5}
"请正确输入选项!"
Start-Sleep -Seconds 1
options4
}
function options5 {
clear
"========================================"
"本地运行:仅使用本机器可访问"
"线上运行:局域网内访问，可端口映射"
"公网分享:借用Gradio服务器，可由任意用户访问，若网络不好将无法启动"
"注意:公网分享务必设置密码，否则易遭受其他用户入侵电脑！"
$Pause8 = Read-Host "请选择运行模式 [1.本地运行  2.线上运行  3.公网分享]"
if ($Pause8 -eq "1") {
options8
}
if ($Pause8 -eq "2") {
$ARG = "$ARG--listen "
options6
}
if ($Pause8 -eq "3") {
$ARG = "$ARG--share "
options6
}
"请正确输入选项!"
Start-Sleep -Seconds 1
options5
}
function options6 {
clear
"========================================"
$Pause9 = Read-Host "请输入端口[留空默认为7860]"
if ($Pause9 -ne "") {
$ARG = "$ARG--port $Pause9 "
options7
}
options7
}
function options7 {
clear
"========================================"
$Pause10 = Read-Host "请以‘用户名:密码’格式输入账号`n[留空默认不设置,不要输入中文或特殊符号，使用‘,’可创建多个用户]"
if ($Pause10 -ne "") {
if ($Pause10 -match "[\u4e00-\u9fa5]") {
"请勿输入中文！"
Start-Sleep -Seconds 2
options7
} else {
$ARG = "$ARG--gradio-auth $Pause10 "
options8
}
options8
}
}
function options8 {
$PSDefaultParameterValues['Out-File:Encoding'] = 'ASCII'
echo "@echo off"> webui-user.bat
echo "set PYTHON=$PWD\Python3.10\python.exe">> webui-user.bat
echo "set GIT=$PWD\Git\mingw64\libexec\git-core\git.exe">> webui-user.bat
echo "set VENV_DIR=">> webui-user.bat
echo "set COMMANDLINE_ARGS=$ARG">> webui-user.bat
echo "set GIT_PYTHON_REFRESH=quiet">> webui-user.bat
echo "call webui.bat">> webui-user.bat
$null = Remove-item ".\first"
clear
"========================================"
"部署已完成..."
Start-Sleep -Seconds 2
deployed
}
function loading {
clear
"========================================"
"当前整合包版本为:V1.3"
"友情链接:"
"QQ(NovelAI贴吧)频道:https://pd.qq.com/s/88sv2u19b"
"NovelAI并联计划导航站:https://a2a.top/"
"NovelAI自助资源站:https://g.h5gd.com/p/yuvwitqh"
"Curry资源站:http://xart.top:44803/#/up"
"Tag魔导书(B站UID:6537379):https://thereisnospon.github.io/NovelAiTag/"
"若使用Windows Terminal按住Ctrl并点击任意链接即可访问"
"若使用CMD,请手动复制并访问"
"========================================"
cmd /c "webui-user.bat"
exit
}
function update {
clear
Get-Content(".\deploy\README.md")
pause
deployed
}
function check {
if (Test-Path "$PWD\first") {
"检测到未完成部署，正在启动部署程序..."
Start-Sleep -Seconds 2
first
} else {
deployed
}
}
function pt {
clear
$Pause11 = Read-Host "模型列表 [1.embeddings模型 2.hypernetwokrd模型 3.ckpt模型]"
if ($Pause11 -eq "1") {
clear
"欢迎提供更多模型"
"提供请携带作者名与来源"
"并@Ms.Rin 或 QQ:1337515813"
"embeddings模型:"
Get-Content ".\tbaimodel\embeddings"
$Pause12 = Read-Host "输入模型编号自动下载,输入c返回上一步"
if ($Pause12 -match "[0-9]") {
powershell (Get-Content .\tbaimodel\embeddings-download)[$Pause12 - 1]
if ($?) {
"下载完成..."
Start-Sleep -Seconds 2
pt
}
if ($Pause12 -eq "c") {
pt
}
"请正确输入选项!"
Start-Sleep -Seconds 2
pt
}
}
if ($Pause11 -eq "2") {
"欢迎提供更多模型"
"提供请携带作者名与来源"
"并@Ms.Rin 或 QQ:1337515813"
"hypernetwokr模型:"
Get-Content ".\tbaimodel\hypernetworks"
$Pause12 = Read-Host "请手动下载后放入$pwd\models\hypernetworks文件夹中`n输入任意内容返回上一步"
pt
}
if ($Pause11 -eq "3") {
"欢迎提供更多模型"
"提供请携带作者名与来源"
"并@Ms.Rin 或 QQ:1337515813"
"ckpt模型:"
Get-Content ".\tbaimodel\ckpt"
$Pause12 = Read-Host "请手动下载后放入$pwd\models\Stable-diffusion文件夹中`n输入任意内容返回上一步"
pt
}
"请正确输入选项!"
Start-Sleep -Seconds 2
pt
}

Set-location $PSScriptRoot
updated
cmd /c "pause"