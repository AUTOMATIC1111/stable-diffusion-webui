function first {
clear
if ("$pwd" -match "\s") {
"��⵽·���к��пո����ƶ����޿ո�·��"
pause
exit
} else {
if ("$pwd" -match "[\u4e00-\u9fa5]") {
"��⵽·���к������ģ����ƶ���������·��"
pause
exit
}
"��ǰҳ����ʾ��Ϊ��������"
$null = Remove-Item .\python3.10\scripts -recurse
$null = .\python3.10\python.exe get-pip.py
$null = .\git\mingw64\bin\git.exe config--global--add safe.directory "*"
}
clear
"========================================"
"��ܰ��ʾ��"
"�������������Ȳ鿴��Ƶ����/Ⱥ�ڹ���/Ƶ������ѯ����"
"��ѯ�ʳ���������ѷ������������⽫���˻ش�"
"����Win10�����Ͻ�Windows Terminal��ΪĬ���ն�"
"�����е��κ�ѡ����������������ո񣡣���"
"========================================��
$Pause1 = Read-Host '������"����֪����������"��������'
if ($Pause1 -eq "����֪����������") {deploying}
"��������ȷ����!"
first
}
function updated {
"========================================"
"������°汾��..."
.\Git\mingw64\libexec\git-core\git.exe -C .\deploy pull
if ((Get-Content ".\deploy\.git\refs\heads\main") -ne "c7e1c049dcdbdf1beb71b8d51ddb0a1988429f9e" ) {
"��⵽�°汾��������£�"
Get-Content ".\deploy\README" -encoding UTF8
pause
check
} else {
"��ǰΪ���°�"
Start-Sleep -Seconds 2
check
}
}
function deployed {
clear
"========================================"
$Pause2 = Read-Host "��⵽����ɲ�����ѡ������ģʽ[1.ֱ������ 2.�ٴβ��� 3.ģ�Ͱ�װ]"
if ($Pause2 -eq "1") {loading}
if ($Pause2 -eq "2") {
$null = Remove-Item .\venv\Scripts -recurse
$null = Remove-Item .\venv\pyvenv.cfg -recurse
deploying
}
if ($Pause2 -eq "3") {clear;"����ģ���б���...";.\Git\mingw64\libexec\git-core\git.exe -C .\tbaimodels pull;pt}
"����ȷ����ѡ��!"
deployed
}
function deploying {
clear
"========================================"
"��������һ�����������"
"========================================"
"��ʼ����Stable Diffusion WebUI"
"�뱣֤C��Ԥ��5G���Ͽռ�"
"========================================"
if (Test-Path "$HOME\.cache") {
$null = Remove-Item $HOME\.cache -recurse
$null = robocopy ".\deploy\.cache" $HOME\.cache /e
} else {
$null = robocopy ".\deploy\.cache" $HOME\.cache /e
}
"�Ѳ������!"
Start-Sleep -Seconds 2
nvidia-smi
if ($?) {
clear
"========================================"
"���ΪN��������GPUģʽ����..."
$ARG = "--autolaunch "
Start-Sleep -Seconds 2
options1
} else {
clear
"========================================"
"���Ϊ��N��������CPUģʽ����..."
Start-Sleep -Seconds 2
options
}
}
function options {
clear
$Pause3 = Read-Host "Ϊ������ʧ����ȷ���Ƿ�ΪN�� [1.�� 2.��]"
if ($Pause3 -eq "1") {
$ARG = "--autolaunch "
options1
}
if ($Pause3 -eq "2") {
$ARG = "--skip-torch-cuda-test --lowvram --always-batch-cond-uncond --no-half --autolaunch "
options4
}
"����ȷ����ѡ��!"
Start-Sleep -Seconds 1
options
}
function options1 {
clear
"========================================"
$Pause4 = Read-Host "�����ʵ�����ѡ���Դ�[1.3G���� 2.4G�Դ� 3.6G�Դ� 4.8G������(ѵ��ģ�ͱ�ѡ)]"
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
"����ȷ����ѡ��!"
Start-Sleep -Seconds 1
options1
}
function options2 {
clear
"========================================"
"������֧�ְ뾫�ȵ��Կ�����ʹ�ú󱨴����ٴβ���رմ�ѡ�"
"16XXϵ���Կ�����عرգ�"
$Pause5 = Read-Host "�Ƿ����뾫��[Y/N]"
if ($Pause5 -eq "Y") {options3}
if ($Pause5 -eq "N") {
$ARG = "$ARG--no-half "
options3
}
"����ȷ����ѡ��!"
Start-Sleep -Seconds 1
options2
}
function options3 {
clear
"========================================"
$Pause6 = Read-Host "�Ƿ���xformers,��ѡ�������30%����[Y/N]"
if ($Pause6 -eq "Y") {
$ARG = "$ARG--xformers "
options4
}
if ($Pause6 -eq "N") {options4}
"����ȷ����ѡ��!"
Start-Sleep -Seconds 1
options3
}
function options4 {
clear
"========================================"
$Pause7 = Read-Host "�Ƿ���DeepDanBooru�������ڲ�ѯTag [Y/N]"
if ($Pause7 -eq "Y") {
$ARG = "$ARG--deepdanbooru "
options5
}
if ($Pause7 -eq "N") {options5}
"����ȷ����ѡ��!"
Start-Sleep -Seconds 1
options4
}
function options5 {
clear
"========================================"
"��������:��ʹ�ñ������ɷ���"
"��������:�������ڷ��ʣ��ɶ˿�ӳ��"
"��������:����Gradio�����������������û����ʣ������粻�ý��޷�����"
"ע��:������������������룬���������������û����ֵ��ԣ�"
$Pause8 = Read-Host "��ѡ������ģʽ [1.��������  2.��������  3.��������]"
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
"����ȷ����ѡ��!"
Start-Sleep -Seconds 1
options5
}
function options6 {
clear
"========================================"
$Pause9 = Read-Host "������˿�[����Ĭ��Ϊ7860]"
if ($Pause9 -ne "") {
$ARG = "$ARG--port $Pause9 "
options7
}
options7
}
function options7 {
clear
"========================================"
$Pause10 = Read-Host "���ԡ��û���:���롯��ʽ�����˺�`n[����Ĭ�ϲ�����,��Ҫ�������Ļ�������ţ�ʹ�á�,���ɴ�������û�]"
if ($Pause10 -ne "") {
if ($Pause10 -match "[\u4e00-\u9fa5]") {
"�����������ģ�"
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
"���������..."
Start-Sleep -Seconds 2
deployed
}
function loading {
clear
"========================================"
"��ǰ���ϰ��汾Ϊ:V1.3"
"��������:"
"QQ(NovelAI����)Ƶ��:https://pd.qq.com/s/88sv2u19b"
"NovelAI�����ƻ�����վ:https://a2a.top/"
"NovelAI������Դվ:https://g.h5gd.com/p/yuvwitqh"
"Curry��Դվ:http://xart.top:44803/#/up"
"Tagħ����(BվUID:6537379):https://thereisnospon.github.io/NovelAiTag/"
"��ʹ��Windows Terminal��סCtrl������������Ӽ��ɷ���"
"��ʹ��CMD,���ֶ����Ʋ�����"
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
"��⵽δ��ɲ������������������..."
Start-Sleep -Seconds 2
first
} else {
deployed
}
}
function pt {
clear
$Pause11 = Read-Host "ģ���б� [1.embeddingsģ�� 2.hypernetwokrdģ�� 3.ckptģ��]"
if ($Pause11 -eq "1") {
clear
"��ӭ�ṩ����ģ��"
"�ṩ��Я������������Դ"
"��@Ms.Rin �� QQ:1337515813"
"embeddingsģ��:"
Get-Content ".\tbaimodel\embeddings"
$Pause12 = Read-Host "����ģ�ͱ���Զ�����,����c������һ��"
if ($Pause12 -match "[0-9]") {
powershell (Get-Content .\tbaimodel\embeddings-download)[$Pause12 - 1]
if ($?) {
"�������..."
Start-Sleep -Seconds 2
pt
}
if ($Pause12 -eq "c") {
pt
}
"����ȷ����ѡ��!"
Start-Sleep -Seconds 2
pt
}
}
if ($Pause11 -eq "2") {
"��ӭ�ṩ����ģ��"
"�ṩ��Я������������Դ"
"��@Ms.Rin �� QQ:1337515813"
"hypernetwokrģ��:"
Get-Content ".\tbaimodel\hypernetworks"
$Pause12 = Read-Host "���ֶ����غ����$pwd\models\hypernetworks�ļ�����`n�����������ݷ�����һ��"
pt
}
if ($Pause11 -eq "3") {
"��ӭ�ṩ����ģ��"
"�ṩ��Я������������Դ"
"��@Ms.Rin �� QQ:1337515813"
"ckptģ��:"
Get-Content ".\tbaimodel\ckpt"
$Pause12 = Read-Host "���ֶ����غ����$pwd\models\Stable-diffusion�ļ�����`n�����������ݷ�����һ��"
pt
}
"����ȷ����ѡ��!"
Start-Sleep -Seconds 2
pt
}

Set-location $PSScriptRoot
updated
cmd /c "pause"