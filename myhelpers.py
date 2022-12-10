import os
import subprocess
import sys
import re
from configparser import ConfigParser
from typing import List
import shutil 
import asyncio
import json
from modules.shared import opts

orig_stdout = sys.stdout
servertag = ''

# paste_fields=[]
# paste_fields_outputs=[]
class _Any:
    pass

any = _Any()

def getDictKeyByReverse(dic:dict):
    keys = [k for k in dic]
    keys.reverse()
    return keys

def loadJson(filename:str)->dict:
    text = readFileAllText(filename)
    if text == '': return {}
    return json.loads(text) 

def readJson(fn:str):return loadJson(fn)

def readJsonWithTag(fn:str):
    return readJson(f'{servertag}{fn}')

def saveJson(dict:dict,fn:str):
    # print(f'saveJson fn:{fn} dic:{dict}')
    saveFileAllText(fn,json.dumps(dict))

def saveJsonWithTag(dict:dict,fn:str):
    saveJson(dict,f'{servertag}{fn}')

def getFilePath(f=__file__):
    """获取当前脚本的路径"""
    return os.path.abspath(f)

def getFileDir(f=__file__):
    """获取当前脚本的路径目录"""
    filepath = os.path.abspath(f)
    return os.path.dirname(filepath)

def foreachFile(path,ation,filter:List=None):
    g = os.walk(path) 
    for dirpath,_,file_list in g:  
        for file_name in file_list:  
            ext = os.path.splitext(file_name)[-1].lower()
            filepath = os.path.join(dirpath, file_name) 
            r = None
            if filter is None:
               r = ation(dirpath,filepath,file_name)
            elif ext in filter:
               r = ation(dirpath,filepath,file_name)
            if r:
                return

def foreachDir(path,ation):
    g = os.walk(path)
    for info in g:  
        ation(info[0])




def openfileToPrint(filename,Utf8=True):
    if Utf8:
        sys.stdout = open(filename, 'w', encoding="utf-8")
    else:
        sys.stdout = open(filename, 'w')
    return sys.stdout,orig_stdout


def copyDir(src,dst):
    return shutil.copytree(src, dst)  

def copyFile(src,dst):
    return shutil.copyfile(src, dst)  

def moveFile(src,dst):
    return shutil.move(src, dst)
	
def readFileAllLines(filename, encoding="utf-8"):
    with open(filename, 'r', encoding=encoding) as file:
        return file.readlines()

def readFileAllText(filename, encoding="utf-8"):
    try:
        with open(filename, 'r', encoding=encoding) as file:
            return file.read()
    except:
        return ''

def readFileAllTextWithTag(filename, encoding="utf-8"):
    return readFileAllText(f'{servertag}{filename}',
    encoding)

def readAllBytes(filename):
    with open(filename, "rb") as fh:
      return fh.read()

def saveFileAllText(filename:str, text:str, encoding="utf-8"):
    with open(filename, 'w', encoding=encoding) as file:
        file.write(text)
        file.flush()

def saveFileAllTextWithTag(filename:str, text:str, encoding="utf-8"):    
    saveFileAllText(f'{servertag}{filename}',text,encoding)


def getfilenameWithoutExt(fn):
    fn = os.path.basename(fn)
    ext = os.path.splitext(fn)[-1]
    return fn.replace(ext, '')

def getConfig(filename='config.ini'):
    config = ConfigParser()
    config.read(filename)
    return config


def saveConfig(config,filename='config.ini'):
    with open(filename, "w") as f:
        config.write(f)

def saveToConfig(config,key,value,section='1'):
    if not config.has_section(section):
        config.add_section(section)
    config.set(section,key,value)

def getFromConfig(config,key,section='1',defaultValue = '')->str:
    return config.get(section, key, fallback=defaultValue)

def getIntFromConfig(config,key,section='1',defaultValue = ''):
    return config.getint(section, key, fallback=defaultValue)

def getFloatFromConfig(config,key,section='1',defaultValue = ''):
    return config.getfloat(section, key, fallback=defaultValue)

def fastGetStringConfig(key,defaultValue,section='1',filename='config.ini'):
    return getConfig(filename).get(section, key, fallback=defaultValue)

def fastGetIntConfig(key,defaultValue,section='1',filename='config.ini'):
    return getConfig(filename).getint(section, key, fallback=defaultValue)

def fastGetFloatConfig(key,defaultValue,section='1',filename='config.ini'):
    return getConfig(filename).getfloat(section, key, fallback=defaultValue)

def fastSaveConfig(key,value,section='1',filename='config.ini'):
    saveToConfig(getConfig(filename),key,value,section)

def is_contain_chinese(check_str):
    """
    判断字符串中是否包含中文
    :param check_str: {str} 需要检测的字符串
    :return: {bool} 包含返回True， 不包含返回False
    """
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def run_subprocess(subprocessName,encoding = 'gbk',shell=False):
    cmd = subprocess.Popen(subprocessName, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                           stdout=subprocess.PIPE, universal_newlines=True, shell=shell, bufsize=1, encoding=encoding)
    # 实时输出
    while True:
        line = cmd.stdout.readline()
        print(line, end='')
        if subprocess.Popen.poll(cmd) == 0:  # 判断子进程是否结束
            break

    return cmd.returncode

def truncateString(data,len_=100,omitStr='...'):
    return (data[:len_] + omitStr) if len(data) > len_ else data

def truncateStringFromBack(data,len_=100,omitStr='...'):
    datalen = len(data)
    if datalen>len_:
        return omitStr + data[datalen-len_:]
    return data

def findFirstDigit(s):
    firstindex = -1
    endindex = -1
    counter = 0
    firstmark = False
    for c in s:
        if c.isdigit():
            if not firstmark:
                firstindex = counter
                firstmark = True
        else:
            if firstmark:
                endindex = counter
                break
        counter+=1
    if firstindex>=0:
        digitstr = s[firstindex:endindex]
        return digitstr
    return None

def truncateString(data,len_=100,omitStr='...'):
    return (data[:len_] + omitStr) if len(data) > len_ else data

def truncateStringFromBack(data,len_=100,omitStr='...'):
    datalen = len(data)
    if datalen>len_:
        return omitStr + data[datalen-len_:]
    return data

async def _read_stream(stream, cb):  
    while True:
        line = await stream.readline()
        if line and cb:
            cb(line)
        else:
            break

async def stream_subprocess(cmd, stdout_cb, stderr_cb):  
    process = await asyncio.create_subprocess_exec(*cmd,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

    await asyncio.wait([
        _read_stream(process.stdout, stdout_cb),
        _read_stream(process.stderr, stderr_cb)
    ])
    return await process.wait()

class ConfigHelper:
    def __init__(self, filename, autosave = True):
        self.filename=filename
        self.config = ConfigParser()
        self.config.read(filename)
        self.autosave=autosave

    def save(self):
        with open(self.filename, "w") as f:
            self.config.write(f)

    def saveToConfig(self,key,value,section='1'):
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section,key,value)
        if self.autosave:
            self.save()

    def getFromConfig(self,key,section='1',defaultValue = '')->str:
        return self.config.get(section, key, fallback=defaultValue)

    def getIntFromConfig(self,key,section='1',defaultValue = ''):
        return self.config.getint(section, key, fallback=defaultValue)

    def getFloatFromConfig(self,key,section='1',defaultValue = ''):
        return self.config.getfloat(section, key, fallback=defaultValue)