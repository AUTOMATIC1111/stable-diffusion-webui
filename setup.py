import paramiko
import os
import re
from dotenv import load_dotenv
load_dotenv()

# change git origin
# git remote set-url origin https://github.com/afterglowstudios/stable-diffusion-webui.git


def replaceInFile(filein, fileout, searchExp, replaceExp):
    with open(filein, 'r') as file :
        filedata = file.read()
    filedata = filedata.replace(searchExp, replaceExp)
    with open(fileout, 'w') as file:
        file.write(filedata)

def setup(ipaddr, port):
    # create SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # connect to remote host
    ssh.connect(hostname=ipaddr, username='root',port=port)

    scp = ssh.open_sftp()

    local_path = 'api.py'
    remote_path = '/workspace/stable-diffusion-webui/modules/api/api.py'
    scp.put(local_path, remote_path)

    replaceInFile('webui-user.sh', 'webui-user.sh.temp', 'AUTOMATIC1111_AUTH', os.getenv('AUTOMATIC1111_AUTH'))
    local_path = 'webui-user.sh.temp'
    remote_path = '/workspace/stable-diffusion-webui/webui-user.sh'
    scp.put(local_path, remote_path)
    os.remove('webui-user.sh.temp')

    local_path = 'download.py'
    remote_path = '/workspace/stable-diffusion-webui/models/Stable-diffusion/download.py'
    scp.put(local_path, remote_path)

    # close SSH connection
    ssh.close()

servers = [
    'ssh -p 41000 root@184.67.78.114 -L 8080:localhost:8080',
    'ssh -p 11100 root@107.222.215.224 -L 8080:localhost:8080',
#    'ssh -p 40082 root@96.20.36.104 -L 8080:localhost:8080',
#    'ssh -p 50300 root@5.12.40.18 -L 8080:localhost:8080',
#    'ssh -p 11100 root@107.222.215.224 -L 8080:localhost:8080'
]

for srv in servers:
    parts = srv.split(' ')
    ipaddr = parts[3].split('@')[1]
    port = int(parts[2])
    print(f"pushing to {ipaddr}")
    setup(ipaddr, port)
