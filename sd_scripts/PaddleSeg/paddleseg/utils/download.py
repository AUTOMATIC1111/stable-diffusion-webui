# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import os
import shutil
import sys
import tarfile
import time
import zipfile

import requests

lasttime = time.time()
FLUSH_INTERVAL = 0.1


def progress(str, end=False):
    global lasttime
    if end:
        str += "\n"
        lasttime = 0
    if time.time() - lasttime >= FLUSH_INTERVAL:
        sys.stdout.write("\r%s" % str)
        lasttime = time.time()
        sys.stdout.flush()


def _download_file(url, savepath, print_progress):
    if print_progress:
        print("Connecting to {}".format(url))
    r = requests.get(url, stream=True, timeout=15)
    total_length = r.headers.get('content-length')

    if total_length is None:
        with open(savepath, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    else:
        with open(savepath, 'wb') as f:
            dl = 0
            total_length = int(total_length)
            starttime = time.time()
            if print_progress:
                print("Downloading %s" % os.path.basename(savepath))
            for data in r.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                if print_progress:
                    done = int(50 * dl / total_length)
                    progress("[%-50s] %.2f%%" %
                             ('=' * done, float(100 * dl) / total_length))
        if print_progress:
            progress("[%-50s] %.2f%%" % ('=' * 50, 100), end=True)


def _uncompress_file_zip(filepath, extrapath):
    files = zipfile.ZipFile(filepath, 'r')
    filelist = files.namelist()
    rootpath = filelist[0]
    total_num = len(filelist)
    for index, file in enumerate(filelist):
        files.extract(file, extrapath)
        yield total_num, index, rootpath
    files.close()
    yield total_num, index, rootpath


def _uncompress_file_tar(filepath, extrapath, mode="r:gz"):
    files = tarfile.open(filepath, mode)
    filelist = files.getnames()
    total_num = len(filelist)
    rootpath = filelist[0]
    for index, file in enumerate(filelist):
        files.extract(file, extrapath)
        yield total_num, index, rootpath
    files.close()
    yield total_num, index, rootpath


def _uncompress_file(filepath, extrapath, delete_file, print_progress):
    if print_progress:
        print("Uncompress %s" % os.path.basename(filepath))

    if filepath.endswith("zip"):
        handler = _uncompress_file_zip
    elif filepath.endswith("tgz"):
        handler = functools.partial(_uncompress_file_tar, mode="r:*")
    else:
        handler = functools.partial(_uncompress_file_tar, mode="r")

    for total_num, index, rootpath in handler(filepath, extrapath):
        if print_progress:
            done = int(50 * float(index) / total_num)
            progress("[%-50s] %.2f%%" %
                     ('=' * done, float(100 * index) / total_num))
    if print_progress:
        progress("[%-50s] %.2f%%" % ('=' * 50, 100), end=True)

    if delete_file:
        os.remove(filepath)

    return rootpath


def download_file_and_uncompress(url,
                                 savepath=None,
                                 extrapath=None,
                                 extraname=None,
                                 print_progress=True,
                                 cover=False,
                                 delete_file=True,
                                 filename=None):
    if savepath is None:
        savepath = "."

    if extrapath is None:
        extrapath = "."

    savename = url.split("/")[-1]
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    savepath = os.path.join(savepath, savename)
    savename = ".".join(savename.split(".")[:-1])
    savename = os.path.join(extrapath, savename)
    extraname = savename if extraname is None else os.path.join(extrapath,
                                                                extraname)

    if cover:
        if os.path.exists(savepath):
            shutil.rmtree(savepath)
        if os.path.exists(savename):
            shutil.rmtree(savename)
        if os.path.exists(extraname):
            shutil.rmtree(extraname)
    full_path = os.path.join(extraname,
                             filename) if filename is not None else extraname
    rank_id_curr_node = int(os.environ.get("PADDLE_RANK_IN_NODE", 0))
    if not os.path.exists(
            full_path):  # If pretrained model exists, skip download process.
        lock_path = extraname + '.download.lock'
        with open(lock_path, 'w'):  # touch    
            os.utime(lock_path, None)
        if rank_id_curr_node == 0:
            if not os.path.exists(savename):
                if not os.path.exists(savepath):
                    _download_file(url, savepath, print_progress)

                if (not tarfile.is_tarfile(savepath)) and (
                        not zipfile.is_zipfile(savepath)):
                    if not os.path.exists(extraname):
                        os.makedirs(extraname)
                    shutil.move(savepath, extraname)

                else:
                    savename = _uncompress_file(savepath, extrapath,
                                                delete_file, print_progress)
                    savename = os.path.join(extrapath, savename)
                    shutil.move(savename, extraname)

            os.remove(lock_path)

        else:
            while os.path.exists(lock_path):
                time.sleep(0.5)

    return extraname
