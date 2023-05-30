#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/3 10:07 AM
# @Author  : wangdongming
# @Site    : 
# @File    : upload_model_ui.py
# @Software: Hifive

import json
import re
import types
import typing
import traceback
import requests
import shutil
import os
import random
import gradio as gr
from typing import Tuple
from tools.file import find_files_from_dir, zip_uncompress

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1"
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
    "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
    "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
    "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",
    # 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.2 Safari/605.1.15',
    "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36"
]


def upload_asset(file_obj, model_type):
    model = 'user-models'
    target_dir = os.path.join(model, model_type)
    os.makedirs(target_dir, exist_ok=True)
    upload_path = os.path.join(target_dir, os.path.basename(file_obj.orig_name))
    shutil.move(file_obj.name, upload_path)
    if os.path.exists(upload_path):
        return "ok"
    return "upload falied"


def http_request(url, method='GET', headers=None, cookies=None, data=None, timeout=30, proxy=None, stream=False):
    from urllib.parse import urlsplit
    _headers = {
        'Accept-Language': 'en-US, en; q=0.8, zh-Hans-CN; q=0.5, zh-Hans; q=0.3',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept': 'text/html, application/xhtml+xml, image/jxr, */*',
        'Connection': 'Keep-Alive',
    }
    if headers and isinstance(headers, dict):
        _headers.update(headers)
    method = method.upper()
    kwargs = {
        'headers': _headers,
        'cookies': cookies,
        'timeout': timeout,
        'verify': False,
    }
    print(f"[{method}]request url:{url}")
    if method == 'GET':
        kwargs['stream'] = stream
    if proxy:
        kwargs['proxies'] = proxy
    if data and method != "GET" and 'json' in _headers.get('Content-Type', ''):
        data = json.dumps(data)
    scheme, netloc, path, query, fragment = urlsplit(url)
    if query:
        data = data or {}
        data.update((item.split('=', maxsplit=1) for item in query.split("&") if item))
        url = url.split("?")[0]

    res = None
    for i in range(3):
        try:
            if 'User-Agent' not in _headers:
                _headers['User-Agent'] = random.choice(USER_AGENTS)
            kwargs['headers'] = _headers
            if method == 'GET':
                res = requests.get(url, data, **kwargs)
            elif method == 'PUT':
                res = requests.put(url, data, **kwargs)
            elif method == 'DELETE':
                res = requests.delete(url, **kwargs)
            elif method == 'OPTIONS':
                data = data if isinstance(data, dict) else {}
                res = requests.options(url, **data)
            else:
                res = requests.post(url, data, **kwargs)
            if res.ok:
                break
        except:
            if i >= 2:
                raise
    return res


def request_model_url(url, model_type, model_name, cover_url, progress=gr.Progress()):
    model = 'user-models'
    if not os.path.exists(model):
        os.mkdir(model)
    target_dir = os.path.join(model, model_type)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    if not url:
        return "cannot found url"
    try:
        url, cover = parse_download_url(url, cover_url)
        resp = http_request(url, timeout=30, stream=True)
        if not model_name:
            if 'Content-Disposition' in resp.headers:
                cd = resp.headers.get('Content-Disposition')
                map = dict(item.strip().split('=')[:2] for item in (item for item in cd.split(';') if '=' in item))
                if 'filename' in map:
                    model_name = map['filename'].strip('"')
            if not model_name:
                import urllib.parse
                (scheme, netloc, path, params, query, fragment) = urllib.parse.urlparse(url)
                model_name = os.path.basename(path)
        filepath = os.path.join(target_dir, model_name)
        if resp.ok:
            chunk_size = 512
            current = 0
            progress(0, desc="starting...")
            file_size = int(resp.headers["content-length"])
            progress.tqdm(range(file_size // chunk_size), desc='download progress')
            with open(filepath, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        current += chunk_size
                    # 迭代一次
                    for i in progress:
                        break

        else:
            raise Exception("response error:" + resp.text)
        if cover:
            without_ex, ex = os.path.splitext(model_name)
            resp = http_request(cover, timeout=30)
            if resp:
                ex = '.png'
                # ct = resp.headers.get('Content-Type')
                # if ct and 'image/' in ct:
                #     ex = ct.replace('image/', '.')
                cover_path = os.path.join(target_dir, without_ex + ex)
                with open(cover_path, 'wb') as f:
                    f.write(resp.content)
            else:
                raise Exception("response error:" + resp.text)
    except Exception as err:
        traceback.print_exc()
        return f"download failed:{err}"
    return "ok"


def parse_download_url(url: str, cover: str) -> Tuple[str, str]:
    if 'civitai' in url:
        ms = re.match('https://civitai.com/api/download/models/\d+', url)
        if ms:
            return url, cover
        if 'civitai.com/models/' in url:
            resp = http_request(url)
            if not resp.ok:
                raise Exception(f'response err code:{resp.status_code} url')
            text = resp.content.decode("utf-8")
            ms = re.search('(api/download/models/\d+).+?Download\sLatest', text)
            if ms:
                ms2 = re.search('https://imagecache.civitai.com.+?"', text)
                cover = ms2.group(0) if ms2 else cover
                return 'https://civitai.com/' + ms.group(1), cover
    elif 'samba' in url:
        if not cover:
            without_ex, _ = os.path.splitext(url)
            return url, without_ex + '.png'

    return url, cover


def create_upload_model_ui():
    gr.Label("你可以提供下载链接(本地文件需先上传服务器并提供外网访问URL)，选择模型类型并上传提示OK后完成", label=None)
    # gr.Label("You can upload the model via a local file or a specified network URL", label=None)
    radio_ctl = gr.Radio(["Lora", "Stable-diffusion"],
                         value="Lora",
                         label="选择模型类型:")
    with gr.Tabs(elem_id="tabs") as tabs:
        with gr.TabItem('模型文件上传', elem_id='tab_upload_file'):
            gr.Label(None, label="通过模型文件上传，模型文件较大，请耐心等待:")
            upload_ctl = gr.File(label="本地上传模型文件:")
            upload_img_ctl = gr.File(label="本地上传模型文件封面（可选,需要与模型文件名称一致的png格式图片）:")
        with gr.TabItem('通过URL上传', elem_id='tab_upload_file'):
            url_txt_ctl = gr.Textbox(label="从URL下载:", placeholder="输入下载链接，支持civitai,samba页面地址直接解析")
            model_name_ctl = gr.Textbox(label="自定义文件名:", placeholder="自定义模型命名（含后缀），默认使用平台命名")
            url_img_ctl = gr.Textbox(label="从URL下载封面:",
                                     placeholder="输入封面下载链接，civitai自动解析无需手动添加,samba默认自动拉取同名PNG资源")
            btn = gr.Button(value="开始下载")

    result = gr.Label(label="上传结果:")
    upload_ctl.upload(fn=upload_asset,
                      inputs=[upload_ctl, radio_ctl],
                      outputs=[result],
                      show_progress=True)
    # upload_ctl.change(
    #     inputs=[upload_ctl, radio_ctl],
    #     outputs=[result],
    #     show_progress=True)

    upload_img_ctl.upload(fn=upload_asset,
                          inputs=[upload_img_ctl, radio_ctl],
                          outputs=[result],
                          show_progress=True)
    btn.click(request_model_url,
              inputs=[url_txt_ctl, radio_ctl, model_name_ctl, url_img_ctl],
              outputs=[result],
              show_progress=True)


def create_rm_model_ui():
    gr.Label("你可以对自己空间模型进行管理", label=None)
    # gr.Label("You can upload the model via a local file or a specified network URL", label=None)
    radio_ctl = gr.Radio(["Lora", "Stable-diffusion"],
                         label="选择模型类型:")
    # model_set = gr.Dataset(
    #     label='选择要删除的文件',
    #     headers=["文件名", "路径"],
    #     components=[gr.Textbox(show_label=False, visible=False), gr.Textbox(show_label=False, visible=False)]
    # )
    state = gr.State()
    model_frame = gr.Dataframe(
        headers=["路径"],
        datatype=["str"],
        col_count=1,
        type='array'
    )
    confirm = gr.Button('删除选中文件')

    def list_models(relative):
        models = []
        relative = os.path.join('user-models', relative)
        for fn in find_files_from_dir(relative, 'safetensors', 'ckpt', 'pt'):
            index = fn.index(relative)

            models.append([fn[index:]])
        if not models:
            models = None

        return models

    def set_ds(relative):
        models = list_models(relative)
        return gr.Dataframe.update(
            value=models
        )

    def remove(index, models):
        filename = models[index][0]
        if os.path.isfile(filename):
            os.remove(filename)
        base, _ = os.path.splitext(filename)
        cover = base + '.png'
        if os.path.isfile(cover):
            os.remove(cover)

        values = [x for x in models if x != models[index]]
        if not values:
            return None

        return values

    def selected(ind: gr.SelectData):
        return ind.index[0]

    radio_ctl.change(list_models, inputs=radio_ctl, outputs=[model_frame])
    model_frame.select(fn=selected, outputs=state)
    confirm.click(remove, inputs=[state, model_frame], outputs=model_frame)


def create_upload_others():
    gr.Label("你可以上传资料到指定路径", label=None)
    # gr.Label("You can upload the model via a local file or a specified network URL", label=None)
    radio_ctl = gr.Radio(["用户空间", "插件物料"],
                         value="用户空间",
                         label="选择文件类型:")
    extension_input = gr.Textbox(
        placeholder='如:tagcomplete/tags/zh_cn.csv',
        label='输入插件物料路径:extensions/',
        interactive=True,
        visible=False
    )

    def upload(file_obj, asset_type, relative: str):
        folder = 'extensions'
        if asset_type == '插件物料':
            relative = relative.lstrip('/')
            if relative.startswith(folder):
                relative = relative[len(folder):].lstrip('/')
            file_path = os.path.join(folder, relative)
        else:
            folder = 'user-asset'
            file_path = os.path.join(folder, os.path.basename(file_obj.orig_name))
        os.makedirs(folder, exist_ok=True)
        shutil.copy(file_obj.name, file_path)

        base, ex = os.path.splitext(file_path)
        if ex.lower() == '.zip':
            dst = os.path.join(folder, base+'_zip')
            os.makedirs()
            zip_uncompress(file_path, )
        return os.path.join(os.getcwd(), file_path)

    def update_visible(asset_type):
        visible = asset_type == '插件物料'
        return gr.Textbox.update(
            visible=visible
        )

    upload_ctl = gr.File(label="本地上传文件:")
    result = gr.Label(label="服务器文件路径:")

    radio_ctl.change(
        update_visible,
        inputs=[radio_ctl],
        outputs=[extension_input])
    upload_ctl.upload(fn=upload,
                      inputs=[upload_ctl, radio_ctl, extension_input],
                      outputs=[result],
                      show_progress=True)


def download_user_assets():
    gr.Label("你可以下载个人物料空间下的文件", label=None)
    refresh_btn = gr.Button('刷新文件列表')

    def list_files():
        files = []
        for fn in os.listdir('user-asset'):
            files.append(os.path.join('user-asset', fn))
        if not files:
            files = None
        return files

    f = gr.Files(label='选择文件下载:', interactive=False)

    refresh_btn.click(list_files, outputs=f)


def append_upload_model_ui(interfaces: typing.List):
    '''
    将模型页添加到UI的TAB，需要在ui.py下找到interfaces列表下加入：
        from .upload_model_ui import append_upload_model_ui
        append_upload_model_ui(interfaces)
    并设置launch时的queue限制
    :param interfaces:
    :return:
    '''
    with gr.Blocks(analytics_enabled=False) as upload_model_interface:
        with gr.Tabs(elem_id="upload_tab_elem") as tabs:
            with gr.TabItem('模型上传', elem_id='tab_upload_model'):
                create_upload_model_ui()
            with gr.TabItem('模型删除', elem_id='tab_rm_model'):
                create_rm_model_ui()
            with gr.TabItem('物料上传', elem_id='tab_upload_others'):
                create_upload_others()
            with gr.TabItem('资料下载'):
                download_user_assets()
    interfaces.append((upload_model_interface, "文件管理", "FileManager"))
