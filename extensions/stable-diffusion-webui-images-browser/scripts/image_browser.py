import gradio as gr
import csv
import importlib
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import stat
import subprocess as sp
import sys
import tempfile
import time
import torch
import traceback
import hashlib
import modules.extras
import modules.images
import modules.ui
from datetime import datetime
from modules import paths, shared, script_callbacks, scripts, images
from modules.shared import opts, cmd_opts
from modules.ui_common import plaintext_to_html
from modules.ui_components import ToolButton, DropdownMulti
from PIL import Image, UnidentifiedImageError
from packaging import version
from pathlib import Path
from typing import List, Tuple
from itertools import chain
from io import StringIO

try:
    from scripts.wib import wib_db
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
    from wib import wib_db

try:
    from send2trash import send2trash
    send2trash_installed = True
except ImportError:
    print("Image Browser: send2trash is not installed. recycle bin cannot be used.")
    send2trash_installed = False

# Force reload wib_db, as it doesn't get reloaded otherwise, if an extension update is started from webui
importlib.reload(wib_db)

yappi_do = False

components_list = ["Sort by", "Filename keyword search", "EXIF keyword search", "Ranking Filter", "Aesthestic Score", "Generation Info", "File Name", "File Time", "Open Folder", "Send to buttons", "Copy to directory", "Gallery Controls Bar", "Ranking Bar", "Delete Bar", "Additional Generation Info"]

num_of_imgs_per_page = 0
loads_files_num = 0
image_ext_list = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".svg"]
exif_cache = {}
aes_cache = {}
none_select = "Nothing selected"
refresh_symbol = '\U0001f504'  # 🔄
up_symbol = '\U000025b2'  # ▲
down_symbol = '\U000025bc'  # ▼
caution_symbol = '\U000026a0'  # ⚠
folder_symbol = '\U0001f4c2'  # 📂
current_depth = 0
init = True
copy_move = ["Move", "Copy"]
copied_moved = ["Moved", "Copied"]
np = "negative_prompt: "
openoutpaint = False
controlnet = False
js_dummy_return = None
log_file = os.path.join(scripts.basedir(), "image_browser.log")

db_version = wib_db.check()

favorite_tab_name = "Favorites"
default_tab_options = ["txt2img", "img2img", "txt2img-grids", "img2img-grids", "Extras", favorite_tab_name, "Others", "All", "Maintenance"]

def check_image_browser_active_tabs():
    last_default_tab = wib_db.get_last_default_tab()
    if last_default_tab[0] == "Others":
        # New tabs don't exist yet in image_browser_active_tabs, add them
        conn, cursor = wib_db.transaction_begin()
        wib_db.update_db_data(cursor, "last_default_tab", "Maintenance")
        wib_db.transaction_end(conn, cursor)
        if hasattr(opts, "image_browser_active_tabs"):
            active_and_new_tabs = f"{opts.image_browser_active_tabs}, All, Maintenance"
            shared.opts.__setattr__("image_browser_active_tabs", active_and_new_tabs)
            shared.opts.save(shared.config_filename)

check_image_browser_active_tabs()
tabs_list = [tab.strip() for tab in chain.from_iterable(csv.reader(StringIO(opts.image_browser_active_tabs))) if tab] if hasattr(opts, "image_browser_active_tabs") and opts.image_browser_active_tabs != ""  else default_tab_options


def callPseudonym(req : gr.Request):
    """
    Sets a global pseudo variable which contains the current user
    (the one that sent the gradio request by interacting with the UI)
    """
    global pseudo
    pseudo = req.username

# Defining paths for image browser
path_maps = {
    "txt2img": opts.outdir_samples or opts.outdir_txt2img_samples,
    "img2img": opts.outdir_samples or opts.outdir_img2img_samples,
    "txt2img-grids": opts.outdir_grids or opts.outdir_txt2img_grids,
    "img2img-grids": opts.outdir_grids or opts.outdir_img2img_grids,
    "Extras": opts.outdir_samples or opts.outdir_extras_samples,
    favorite_tab_name: opts.outdir_save
}

# Force create the paths at the extension's launch
for path in path_maps.values():
    if not os.path.exists(path):
        os.makedirs(path)

class ImageBrowserTab():

    seen_base_tags = set()

    def __init__(self, name: str):
        self.name: str = os.path.basename(name) if os.path.isdir(name) else name
        self.path: str = os.path.realpath(path_maps.get(name, name))
        self.base_tag: str = f"image_browser_tab_{self.get_unique_base_tag(self.remove_invalid_html_tag_chars(self.name).lower())}"

    def remove_invalid_html_tag_chars(self, tag: str) -> str:
        # Removes any character that is not a letter, a digit, a hyphen, or an underscore
        removed = re.sub(r'[^a-zA-Z0-9\-_]', '', tag)
        return removed

    def get_unique_base_tag(self, base_tag: str) -> str:
        counter = 1
        while base_tag in self.seen_base_tags:
            match = re.search(r'_(\d+)$', base_tag)
            if match:
                counter = int(match.group(1)) + 1
                base_tag = re.sub(r'_(\d+)$', f"_{counter}", base_tag)
            else:
                base_tag = f"{base_tag}_{counter}"
            counter += 1
        self.seen_base_tags.add(base_tag)
        return base_tag

    def __str__(self):
        return f"Name: {self.name} / Path: {self.path} / Base tag: {self.base_tag} / Seen base tags: {self.seen_base_tags}"

tabs_list = [ImageBrowserTab(tab) for tab in tabs_list]

debug_level_types = ["none", "warning log", "debug log", "javascript log", "capture logs to file"]

debug_levels_list = []
for i in range(len(debug_level_types)):
    level = debug_level_types[i].split(" ")[0]
    text = str(i) + " - " + debug_level_types[i]
    debug_levels_list.append((level, text))

def debug_levels(arg_value=None, arg_level=None, arg_text=None):
    if arg_value is not None:
        return arg_value, debug_levels_list[arg_value]
    elif arg_level is not None:
        for i, (level, text) in enumerate(debug_levels_list):
            if level == arg_level:
                return i, debug_levels_list[i]
    elif arg_text is not None:
        for i, (level, text) in enumerate(debug_levels_list):
            if text == arg_text:
                return i, debug_levels_list[i]

# Logging
logger = None
def restart_debug(parameter):
    global logger
    logger = logging.getLogger(__name__)
    logger.disabled = False
    logger_mode = logging.ERROR
    level_value = 0
    capture_level_value = 99
    if hasattr(opts, "image_browser_debug_level"):
        warning_level_value, (warning_level, warning_level_text) = debug_levels(arg_level="warning")
        debug_level_value, (debug_level, debug_level_text) = debug_levels(arg_level="debug")
        capture_level_value, (capture_level, capture_level_text) = debug_levels(arg_level="capture")
        level_value, (level, level_text) = debug_levels(arg_text=opts.image_browser_debug_level)
        if level_value >= debug_level_value:
            logger_mode = logging.DEBUG
        elif level_value >= warning_level_value:
            logger_mode = logging.WARNING
    logger.setLevel(logger_mode)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logger_mode)
    formatter = logging.Formatter(f'%(asctime)s image_browser.py: %(message)s', datefmt='%Y-%m-%d-%H:%M:%S')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if level_value >= capture_level_value:
        try:
            os.unlink(log_file)
        except FileNotFoundError:
            pass
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logger_mode)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.warning(f"debug_level: {level_value}")
    # Debug logging
    if logger.getEffectiveLevel() == logging.DEBUG:
        if parameter != "startup":
            logging.disable(logging.NOTSET)

        logger.debug(f"{sys.executable} {sys.version}")
        logger.debug(f"{platform.system()} {platform.version()}")
        try:
            git = os.environ.get('GIT', "git")
            webui_commit_hash = os.popen(f"{git} rev-parse HEAD").read().strip()
            sm_hashes = os.popen(f"{git} submodule").read()
            sm_hashes_lines = sm_hashes.splitlines()
            image_browser_commit_hash = f"image_browser_commit_hash not found: {sm_hashes}"
            for sm_hashes_line in sm_hashes_lines:
                if "images-browser" in sm_hashes_line.lower():
                    image_browser_commit_hash = sm_hashes_line[1:41]
                    break            
        except Exception as e:
            webui_commit_hash = e
            image_browser_commit_hash = e
        logger.debug(f"Webui {webui_commit_hash}")
        logger.debug(f"Image Browser {image_browser_commit_hash}")
        logger.debug(f"Gradio {gr.__version__}")
        logger.debug(f"{paths.script_path}")
        # Don't spam config-files to console
        logger.removeHandler(console_handler)
        with open(cmd_opts.ui_config_file, "r") as f:
            logger.debug(f.read())
        with open(cmd_opts.ui_settings_file, "r") as f:
            logger.debug(f.read())
        logger.addHandler(console_handler)
        logger.debug(os.path.realpath(__file__))
        logger.debug([str(tab) for tab in tabs_list])
        logger.debug(f"db_version: {db_version}")
    maint_last_msg = "Debug restarted"

    return parameter, maint_last_msg

restart_debug("startup")

def delete_recycle(filename):
    if opts.image_browser_delete_recycle and send2trash_installed:
        send2trash(filename)
    else:
        file = Path(filename)
        file.unlink()
    return

def img_path_subdirs_get(img_path, req:gr.Request):
    callPseudonym(req)
    subdirs = []
    subdirs.append(none_select)
    for item in os.listdir(img_path):
        item_path = os.path.join(img_path, item)
        if os.path.isdir(item_path):
            subdirs.append(item_path)
    return gr.update(choices=subdirs)

def img_path_add_remove(img_dir, path_recorder, add_remove, img_path_depth, req:gr.Request):
    callPseudonym(req)
    img_dir = os.path.realpath(img_dir)    
    if add_remove == "add" or (add_remove == "remove" and img_dir in path_recorder):
        if add_remove == "add":
            path_recorder[img_dir] = {
                "depth": int(img_path_depth),
                "path_display": f"{img_dir} [{int(img_path_depth)}]"
            }
            wib_db.update_path_recorder(img_dir, path_recorder[img_dir]["depth"], path_recorder[img_dir]["path_display"])
        else:
            del path_recorder[img_dir]
            wib_db.delete_path_recorder(img_dir)
    path_recorder_formatted = [value.get("path_display") for key, value in path_recorder.items()]
    path_recorder_formatted = sorted(path_recorder_formatted, key=lambda x: natural_keys(x.lower()))

    if add_remove == "remove":
        selected = path_recorder[list(path_recorder.keys())[0]]["path_display"]
    else:
        selected = path_recorder[img_dir]["path_display"]
    return path_recorder, gr.update(choices=path_recorder_formatted, value=selected)

def sort_order_flip(turn_page_switch, sort_order, req:gr.Request):
    callPseudonym(req)
    if sort_order == up_symbol:
        sort_order = down_symbol
    else:
        sort_order = up_symbol
    return 1, -turn_page_switch, sort_order

def read_path_recorder():
    path_recorder = wib_db.load_path_recorder()
    path_recorder_formatted = [value.get("path_display") for key, value in path_recorder.items()]
    path_recorder_formatted = sorted(path_recorder_formatted, key=lambda x: natural_keys(x.lower()))
    path_recorder_unformatted = list(path_recorder.keys())
    path_recorder_unformatted = sorted(path_recorder_unformatted, key=lambda x: natural_keys(x.lower()))

    return path_recorder, path_recorder_formatted, path_recorder_unformatted

def pure_path(path):
    if path == []:
        return path, 0
    match = re.search(r" \[(\d+)\]$", path)
    if match:
        path = path[:match.start()]
        depth = int(match.group(1))
    else:
        depth = 0
    path = os.path.realpath(path)
    return path, depth

def browser2path(img_path_browser):
    img_path, _ = pure_path(img_path_browser)
    return img_path

def totxt(file):
    base, _ = os.path.splitext(file)
    file_txt = base + '.txt'

    return file_txt

def tab_select(request : gr.Request):
    callPseudonym(request)

    path_recorder, path_recorder_formatted, path_recorder_unformatted = read_path_recorder()
    return path_recorder, gr.update(choices=path_recorder_unformatted)

def js_logs_output(js_log, req:gr.Request):
    callPseudonym(req)
    logger.debug(f"js_log: {js_log}")
    return js_log

def ranking_filter_settings(page_index, turn_page_switch, ranking_filter, req:gr.Request):
    callPseudonym(req)
    if ranking_filter == "Min-max":
        interactive = True
    else:
        interactive = False
    page_index = 1
    turn_page_switch = -turn_page_switch
    return page_index, turn_page_switch, gr.update(interactive=interactive), gr.update(interactive=interactive)

def reduplicative_file_move(src, dst):
    def same_name_file(basename, path):
        name, ext = os.path.splitext(basename)
        f_list = os.listdir(path)
        max_num = 0
        for f in f_list:
            if len(f) <= len(basename):
                continue
            f_ext = f[-len(ext):] if len(ext) > 0 else ""
            if f[:len(name)] == name and f_ext == ext:                
                if f[len(name)] == "(" and f[-len(ext)-1] == ")":
                    number = f[len(name)+1:-len(ext)-1]
                    if number.isdigit():
                        if int(number) > max_num:
                            max_num = int(number)
        return f"{name}({max_num + 1}){ext}"
    name = os.path.basename(src)
    save_name = os.path.join(dst, name)
    src_txt_exists = False
    if opts.image_browser_txt_files:
        src_txt = totxt(src)
        if os.path.exists(src_txt):
            src_txt_exists = True
    if not os.path.exists(save_name):
        if opts.image_browser_copy_image:
            shutil.copy2(src, dst)
            if opts.image_browser_txt_files and src_txt_exists:
                shutil.copy2(src_txt, dst)
        else:
            shutil.move(src, dst)
            if opts.image_browser_txt_files and src_txt_exists:
                shutil.move(src_txt, dst)
    else:
        name = same_name_file(name, dst)
        if opts.image_browser_copy_image:
            shutil.copy2(src, os.path.join(dst, name))
            if opts.image_browser_txt_files and src_txt_exists:
                shutil.copy2(src_txt, totxt(os.path.join(dst, name)))
        else:
            shutil.move(src, os.path.join(dst, name))
            if opts.image_browser_txt_files and src_txt_exists:
                shutil.move(src_txt, totxt(os.path.join(dst, name)))

def save_image(file_name, filenames, page_index, turn_page_switch, dest_path, req:gr.Request):
    callPseudonym(req)
    if dest_path.__contains__("favorites"):
        opts.outdir_save = "\\".join(shared.opts.outdir_samples.rsplit("\\", 1)[:-1]) # Cutting "favorites"
        opts.outdir_save = "\\".join(shared.opts.outdir_samples.rsplit("\\", 1)[:-1]) # Cutting "[user]"
        opts.outdir_save += f"\\{pseudo}\\favorites" # Adding new username + favorites        dest_path =  opts.outdir_save
        dest_path = opts.outdir_save

    if file_name is not None and os.path.exists(file_name):
        reduplicative_file_move(file_name, dest_path)
        message = f"<div style='color:#999'>{copied_moved[opts.image_browser_copy_image]} to {dest_path}</div>"
        if not opts.image_browser_copy_image:
            # Force page refresh with checking filenames
            filenames = []
            turn_page_switch = -turn_page_switch
    else:
        message = "<div style='color:#999'>Image not found (may have been already moved)</div>"

    return message, filenames, page_index, turn_page_switch

def delete_image(tab_base_tag_box, delete_num, name, filenames, image_index, visible_num, delete_confirm, turn_page_switch, image_page_list):
    logger.debug("delete_image")
    refresh = False
    delete_num = int(delete_num)
    image_index = int(image_index)
    visible_num = int(visible_num)
    image_page_list = json.loads(image_page_list)
    new_file_list = []
    new_image_page_list = []
    if name == "":
        refresh = True
    else:
        try:
            index_files = list(filenames).index(name)
            
            index_on_page = image_page_list.index(name)
        except ValueError as e:
            print(traceback.format_exc(), file=sys.stderr)
            # Something went wrong, force a page refresh
            refresh = True
    if not refresh:
        if not delete_confirm:
            delete_num = min(visible_num - index_on_page, delete_num)
        new_file_list = filenames[:index_files] + filenames[index_files + delete_num:]
        new_image_page_list = image_page_list[:index_on_page] + image_page_list[index_on_page + delete_num:]

        for i in range(index_files, index_files + delete_num):
            if os.path.exists(filenames[i]):
                if opts.image_browser_delete_message:
                    print(f"Deleting file {filenames[i]}")
                delete_recycle(filenames[i])
                visible_num -= 1
                if opts.image_browser_txt_files:
                    txt_file = totxt(filenames[i])
                    if os.path.exists(txt_file):
                        delete_recycle(txt_file)
            else:
                print(f"File does not exist {filenames[i]}")
                # If we reach this point (which we shouldn't), things are messed up, better force a page refresh
                refresh = True

    if refresh:
        turn_page_switch = -turn_page_switch
        select_image = False
    else:
        select_image = True

    return new_file_list, 1, turn_page_switch, visible_num, new_image_page_list, select_image, json.dumps(new_image_page_list)

def traverse_all_files(curr_path, image_list, tab_base_tag_box, img_path_depth) -> List[Tuple[str, os.stat_result, str, int]]:
    global current_depth
    logger.debug(f"curr_path: {curr_path}")
    if curr_path == "":
        return image_list
    f_list = [(os.path.join(curr_path, entry.name), entry.stat()) for entry in os.scandir(curr_path)]
    for f_info in f_list:
        fname, fstat = f_info
        if os.path.splitext(fname)[1] in image_ext_list:
            image_list.append(f_info)
        elif stat.S_ISDIR(fstat.st_mode):
            if (opts.image_browser_with_subdirs and tab_base_tag_box != "image_browser_tab_others") or (tab_base_tag_box == "image_browser_tab_all") or (tab_base_tag_box == "image_browser_tab_others" and img_path_depth != 0 and (current_depth < img_path_depth or img_path_depth < 0)):
                current_depth = current_depth + 1
                image_list = traverse_all_files(fname, image_list, tab_base_tag_box, img_path_depth)
                current_depth = current_depth - 1
    return image_list

def cache_exif(fileinfos):
    global exif_cache, aes_cache

    if yappi_do:
        import yappi
        import pandas as pd
        yappi.set_clock_type("wall")
        yappi.start()
    
    cache_exif_start = time.time()
    new_exif = 0
    new_aes = 0
    conn, cursor = wib_db.transaction_begin()
    for fi_info in fileinfos:
        if any(fi_info[0].endswith(ext) for ext in image_ext_list):
            found_exif = False
            found_aes = False
            if fi_info[0] in exif_cache:
                found_exif = True
            if fi_info[0] in aes_cache:
                found_aes = True
            if not found_exif or not found_aes:
                exif_cache[fi_info[0]] = "0"
                aes_cache[fi_info[0]] = "0"
                try:
                    image = Image.open(fi_info[0])
                    (_, allExif, allExif_html) = modules.extras.run_pnginfo(image)
                    image.close()
                except SyntaxError:
                    allExif = False
                    logger.warning(f"Extension and content don't match: {fi_info[0]}")
                except UnidentifiedImageError as e:
                    allExif = False
                    logger.warning(f"UnidentifiedImageError: {e}")
                except Image.DecompressionBombError as e:
                    allExif = False
                    logger.warning(f"DecompressionBombError: {e}: {fi_info[0]}")
                except PermissionError as e:
                    allExif = False
                    logger.warning(f"PermissionError: {e}: {fi_info[0]}")
                except FileNotFoundError as e:
                    allExif = False
                    logger.warning(f"FileNotFoundError: {e}: {fi_info[0]}")
                except OSError as e:
                    if e.errno == 22:
                        logger.warning(f"Caught OSError with error code 22: {fi_info[0]}")
                    else:
                        raise
                if allExif:
                    exif_cache[fi_info[0]] = allExif
                    wib_db.update_exif_data(conn, fi_info[0], allExif)
                    new_exif = new_exif + 1

                    m = re.search("(?:aesthetic_score:|Score:) (\d+.\d+)", allExif, flags=re.IGNORECASE)
                    if m:
                        aes_value = m.group(1)
                    else:
                        aes_value = "0"
                    aes_cache[fi_info[0]] = aes_value
                    wib_db.update_exif_data_by_key(conn, fi_info[0], "aesthetic_score", aes_value)
                    new_aes = new_aes + 1
                else:
                    try:
                        filename = os.path.splitext(fi_info[0])[0] + ".txt"
                        geninfo = ""
                        with open(filename) as f:
                            for line in f:
                                geninfo += line
                        exif_cache[fi_info[0]] = geninfo
                        wib_db.update_exif_data_by_key(conn, fi_info[0], geninfo)
                        new_exif = new_exif + 1

                        m = re.search("(?:aesthetic_score:|Score:) (\d+.\d+)", geninfo, flags=re.IGNORECASE)
                        if m:
                            aes_value = m.group(1)
                        else:
                            aes_value = "0"
                        aes_cache[fi_info[0]] = aes_value
                        wib_db.update_exif_data_by_key(conn, fi_info[0], "aesthetic_score", aes_value)
                        new_aes = new_aes + 1
                    except Exception:
                        logger.warning(f"cache_exif: No EXIF in image or txt file for {fi_info[0]}")
                        # Saved with defaults to not scan it again next time
                        exif_cache[fi_info[0]] = "0"
                        allExif = "0"
                        wib_db.update_exif_data(conn, fi_info[0], allExif)
                        new_exif = new_exif + 1

                        aes_value = "0"
                        aes_cache[fi_info[0]] = aes_value
                        wib_db.update_exif_data_by_key(conn, fi_info[0], "aesthetic_score", aes_value)
                        new_aes = new_aes + 1
    wib_db.transaction_end(conn, cursor)

    if yappi_do:
        yappi.stop()
        pd.set_option('display.float_format', lambda x: '%.6f' % x)
        yappi_stats = yappi.get_func_stats().strip_dirs()
        data = [(s.name, s.ncall, s.tsub, s.ttot, s.ttot/s.ncall) for s in yappi_stats]
        df = pd.DataFrame(data, columns=['name', 'ncall', 'tsub', 'ttot', 'tavg'])
        print(df.to_string(index=False))
        yappi.get_thread_stats().print_all()

    cache_exif_end = time.time()
    logger.debug(f"cache_exif: {new_exif}/{len(fileinfos)} cache_aes: {new_aes}/{len(fileinfos)} {round(cache_exif_end - cache_exif_start, 1)} seconds")

def exif_rebuild(maint_wait, req:gr.Request):
    callPseudonym(req)
    global exif_cache, aes_cache
    if opts.image_browser_scan_exif:
        logger.debug("Rebuild start")
        exif_dirs = wib_db.get_exif_dirs()
        exif_cache = {}
        aes_cache = {}
        for key, value in exif_dirs.items():
            if os.path.exists(key):
                print(f"Rebuilding {key}")
                fileinfos = traverse_all_files(key, [], "", 0)
                cache_exif(fileinfos)
        logger.debug("Rebuild end")
        maint_last_msg = "Rebuild finished"
    else:
        maint_last_msg = "Exif cache not enabled in settings"

    return maint_wait, maint_last_msg

def exif_delete_0(maint_wait, req:gr.Request):
    callPseudonym(req)
    global exif_cache, aes_cache
    if opts.image_browser_scan_exif:
        conn, cursor = wib_db.transaction_begin()
        wib_db.delete_exif_0(cursor)
        wib_db.transaction_end(conn, cursor)
        exif_cache = wib_db.load_exif_data(exif_cache)
        aes_cache = wib_db.load_aes_data(aes_cache)
        maint_last_msg = "Delete finished"
    else:
        maint_last_msg = "Exif cache not enabled in settings"

    return maint_wait, maint_last_msg

def exif_update_dirs(maint_update_dirs_path_recorder, maint_update_dirs_exif_data, maint_update_dirs_ranking, maint_update_dirs_from, maint_update_dirs_to, maint_wait, req:gr.Request):
    callPseudonym(req)
    global exif_cache, aes_cache
    if maint_update_dirs_from == "":
        maint_last_msg = "From is empty"
    elif maint_update_dirs_to == "":
        maint_last_msg = "To is empty"
    else:
        maint_update_dirs_from = os.path.realpath(maint_update_dirs_from)
        maint_update_dirs_to = os.path.realpath(maint_update_dirs_to)
        rows = 0
        conn, cursor = wib_db.transaction_begin()
        if maint_update_dirs_path_recorder:
            wib_db.update_path_recorder_mult(cursor, maint_update_dirs_from, maint_update_dirs_to)
            rows = rows + cursor.rowcount
        if maint_update_dirs_exif_data:
            wib_db.update_exif_data_mult(cursor, maint_update_dirs_from, maint_update_dirs_to)
            rows = rows + cursor.rowcount
        if maint_update_dirs_ranking:
            wib_db.update_ranking_mult(cursor, maint_update_dirs_from, maint_update_dirs_to)
            rows = rows + cursor.rowcount
        wib_db.transaction_end(conn, cursor)
        if rows == 0:
            maint_last_msg = "No rows updated"
        else:
            maint_last_msg = f"{rows} rows updated. Please reload UI!"

    return maint_wait, maint_last_msg

def recreate_hash(maint_wait, req:gr.Request):
    callPseudonym(req)
    version = str(db_version)
    conn, cursor = wib_db.transaction_begin()
    wib_db.migrate_filehash(cursor, version)
    wib_db.transaction_end(conn, cursor)
    maint_last_msg = "Hashes recreated"

    return maint_wait, maint_last_msg

def reapply_ranking(path_recorder, maint_wait, req:gr.Request):
    callPseudonym(req)
    dirs = {}

    for tab in tabs_list:
        if os.path.exists(tab.path):
            dirs[tab.path] = tab.path

    for key in path_recorder:
        if os.path.exists(key):
            dirs[key] = key

    conn, cursor = wib_db.transaction_begin()

    # Traverse all known dirs, check if missing rankings are due to moved files
    for key in dirs.keys():
        fileinfos = traverse_all_files(key, [], "", 0)
        for (file, _) in fileinfos:
            # Is there a ranking for this full filepath
            ranking_by_file = wib_db.get_ranking_by_file(cursor, file)
            if ranking_by_file is None:
                name = os.path.basename(file)
                (ranking_by_name, alternate_hash) = wib_db.get_ranking_by_name(cursor, name)
                # Is there a ranking only for the filename
                if ranking_by_name is not None:
                    hash = wib_db.get_hash(file)
                    (alternate_file, alternate_ranking) = ranking_by_name
                    if alternate_ranking is not None and alternate_hash is not None:
                        (alternate_hash,) = alternate_hash
                    # Does the found filename's file have no hash or the same hash?
                    if alternate_hash is None or hash == alternate_hash:
                        if os.path.exists(alternate_file):
                            # Insert ranking as a copy of the found filename's ranking
                            wib_db.insert_ranking(cursor, file, alternate_ranking, hash)
                        else:
                            # Replace ranking of the found filename
                            wib_db.replace_ranking(cursor, file, alternate_file, hash)

    wib_db.transaction_end(conn, cursor)
    maint_last_msg = "Rankings reapplied"

    return maint_wait, maint_last_msg

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

def open_folder(path):
    if os.path.exists(path):
        # Code from ui_common.py
        if not shared.cmd_opts.hide_ui_dir_config:
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":
                sp.Popen(["open", path])
            elif "microsoft-standard-WSL2" in platform.uname().release:
                sp.Popen(["wsl-open", path])
            else:
                sp.Popen(["xdg-open", path])

def check_ext(ext):
    found = False
    scripts_list = scripts.list_scripts("scripts", ".py")
    for scriptfile in scripts_list:
            if ext in scriptfile.basedir.lower():
                found = True
                break
    return found

def exif_search(needle, haystack, use_regex, case_sensitive):
    found = False
    if use_regex:
        if case_sensitive:
            pattern = re.compile(needle, re.DOTALL)
        else:
            pattern = re.compile(needle, re.DOTALL | re.IGNORECASE)
        if pattern.search(haystack) is not None:
            found = True
    else:
        if not case_sensitive:
            haystack = haystack.lower()
            needle = needle.lower()
        if needle in haystack:
            found = True
    return found

def get_all_images(dir_name, sort_by, sort_order, keyword, tab_base_tag_box, img_path_depth, ranking_filter, ranking_filter_min, ranking_filter_max, aes_filter_min, aes_filter_max, exif_keyword, negative_prompt_search, use_regex, case_sensitive):
    global current_depth
    logger.debug("get_all_images")
    current_depth = 0

    # ------------------------------------------------------------------------

    # GRIDS & IMAGES
    opts.outdir_samples =  "\\".join(opts.outdir_samples.rsplit("\\", 1)[:-1]) # Cutting [user]
    opts.outdir_samples += f"\\{pseudo}" # Adding new username

    opts.outdir_grids = "\\".join(opts.outdir_grids.rsplit("\\", 1)[:-1])
    opts.outdir_grids += f"\\{pseudo}"

    # SAVE
    opts.outdir_save = "\\".join(opts.outdir_samples.rsplit("\\", 1)[:-1]) # Cutting "favorites"
    opts.outdir_save = "\\".join(opts.outdir_samples.rsplit("\\", 1)[:-1]) # Cutting "[user]"
    opts.outdir_save += f"\\{pseudo}\\favorites" # Adding new username + favorites

    # Redefining paths for image browser in the dictionnary
    path_maps["txt2img"] =  opts.outdir_samples or opts.outdir_txt2img_samples 
    path_maps["img2img"] =  opts.outdir_samples or opts.outdir_img2img_samples 
    path_maps["txt2img-grids"] = opts.outdir_grids or opts.outdir_txt2img_grids 
    path_maps["img2img-grids"] = opts.outdir_grids or opts.outdir_img2img_grids 
    path_maps["Extras"] =  opts.outdir_samples or opts.outdir_extras_samples 
    path_maps[favorite_tab_name] = opts.outdir_save

    # Creating folders if they don't exist
    for path in path_maps.values():
        if not os.path.exists(path):
            os.makedirs(path)


    # Editing dir_name to take the username into account

    # First case : we want to browse in a path that contains "[user]" without favorites
    if dir_name.__contains__("[user]") and not (dir_name.__contains__("favorites")) :
        debut = dir_name.split("[user]")[0] # Cutting user
        dir_name = debut + pseudo # Adding username

    # # Second case : browsing inside a path that contains "favorites"
    elif dir_name.__contains__("[user]") and (dir_name.__contains__("favorites")) :
        debut = dir_name.split("[user]")[0] # Cutting user
        dir_name = debut + pseudo + "\\favorites" # Adding username and favorites at the end of the path


    # ------------------------------------------------------------------------

    fileinfos = []
    if tab_base_tag_box == "image_browser_tab_all":
        for path in path_maps.values():
            list1 = fileinfos
            list2 = traverse_all_files(os.path.realpath(path), [], tab_base_tag_box, img_path_depth)
            tmp = dict(list1)
            tmp.update(dict(list2))
            fileinfos = list(tmp.items())
    else:
        try:
            fileinfos = traverse_all_files(dir_name, [], tab_base_tag_box, img_path_depth)
        except FileNotFoundError:
            print('\033[31mError : Specified path is unknown : \033[0m' + dir_name)
            print("\033[31mMaybe you didn't generate anything yet ?\033[0m")
    
    keyword = keyword.strip(" ")
    
    if opts.image_browser_scan_exif:
        cache_exif(fileinfos)
    
    if len(keyword) != 0:
        fileinfos = [x for x in fileinfos if keyword.lower() in x[0].lower()]
        filenames = [finfo[0] for finfo in fileinfos]
    
    if opts.image_browser_scan_exif:
        conn, cursor = wib_db.transaction_begin()
        if len(exif_keyword) != 0:
            if use_regex:
                regex_error = False
                try:
                    test_re = re.compile(exif_keyword, re.DOTALL)
                except re.error as e:
                    regex_error = True
                    print(f"Regex error: {e}")
            if (use_regex and not regex_error) or not use_regex:
                if negative_prompt_search == "Yes":
                    fileinfos = [x for x in fileinfos if exif_search(exif_keyword, exif_cache[x[0]], use_regex, case_sensitive)]
                else:
                    result = []
                    for file_info in fileinfos:
                        file_name = file_info[0]
                        file_exif = exif_cache[file_name]
                        file_exif_lc = file_exif.lower()
                        start_index = file_exif_lc.find(np)
                        end_index = file_exif.find("\n", start_index)
                        if negative_prompt_search == "Only":
                            start_index = start_index + len(np)
                            sub_string = file_exif[start_index:end_index].strip()
                            if exif_search(exif_keyword, sub_string, use_regex, case_sensitive):
                                result.append(file_info)
                        else:
                            sub_string = file_exif[start_index:end_index].strip()
                            file_exif = file_exif.replace(sub_string, "")
                            
                            if exif_search(exif_keyword, file_exif, use_regex, case_sensitive):
                                result.append(file_info)
                    fileinfos = result
                filenames = [finfo[0] for finfo in fileinfos]
        wib_db.fill_work_files(cursor, fileinfos)
        if len(aes_filter_min) != 0 or len(aes_filter_max) != 0:
            try:
                aes_filter_min_num = float(aes_filter_min)
            except ValueError:
                aes_filter_min_num = sys.float_info.min
            try:
                aes_filter_max_num = float(aes_filter_max)
            except ValueError:
                aes_filter_max_num = sys.float_info.max

            fileinfos = wib_db.filter_aes(cursor, fileinfos, aes_filter_min_num, aes_filter_max_num)
            filenames = [finfo[0] for finfo in fileinfos]   
        if ranking_filter != "All":
            ranking_filter_min_num = 1
            ranking_filter_max_num = 5
            if ranking_filter == "Min-max":
                try:
                    ranking_filter_min_num = int(ranking_filter_min)
                except ValueError:
                    ranking_filter_min_num = 0
                try:
                    ranking_filter_max_num = int(ranking_filter_max)
                except ValueError:
                    ranking_filter_max_num = 0
                if ranking_filter_min_num < 1:
                    ranking_filter_min_num = 1
                if ranking_filter_max_num < 1 or ranking_filter_max_num > 5:
                    ranking_filter_max_num = 5

            fileinfos = wib_db.filter_ranking(cursor, fileinfos, ranking_filter, ranking_filter_min_num, ranking_filter_max_num)
            filenames = [finfo[0] for finfo in fileinfos]
        
        wib_db.transaction_end(conn, cursor)
    
    if sort_by == "date":
        if sort_order == up_symbol:
            fileinfos = sorted(fileinfos, key=lambda x: x[1].st_mtime)
        else:
            fileinfos = sorted(fileinfos, key=lambda x: -x[1].st_mtime)
        filenames = [finfo[0] for finfo in fileinfos]
    elif sort_by == "path name":
        if sort_order == up_symbol:
            fileinfos = sorted(fileinfos)
        else:
            fileinfos = sorted(fileinfos, reverse=True)
        filenames = [finfo[0] for finfo in fileinfos]
    elif sort_by == "random":
        random.shuffle(fileinfos)
        filenames = [finfo[0] for finfo in fileinfos]
    elif sort_by == "ranking":
        finfo_ranked = {}
        for fi_info in fileinfos:
            finfo_ranked[fi_info[0]], _ = get_ranking(fi_info[0])
        if sort_order == up_symbol:
            fileinfos = dict(sorted(finfo_ranked.items(), key=lambda x: (x[1], x[0])))
        else:
            fileinfos = dict(reversed(sorted(finfo_ranked.items(), key=lambda x: (x[1], x[0]))))
        filenames = [finfo for finfo in fileinfos]
    else:
        sort_values = {}
        exif_info = dict(exif_cache)
        if exif_info:
            for k, v in exif_info.items():
                match = re.search(r'(?<='+ sort_by + ":" ').*?(?=(,|$))', v, flags=re.DOTALL|re.IGNORECASE)
                if match:
                    sort_values[k] = match.group().strip()
                else:
                    sort_values[k] = "0"
            if sort_by == "aesthetic_score" or sort_by == "cfg scale":
                sort_float = True
            else:
                sort_float = False

            if sort_order == down_symbol:
                if sort_float:
                    fileinfos = [x for x in fileinfos if sort_values[x[0]] != "0"]
                    fileinfos.sort(key=lambda x: float(sort_values[x[0]]), reverse=True)
                    fileinfos = dict(fileinfos)
                else:
                    fileinfos = dict(reversed(sorted(fileinfos, key=lambda x: natural_keys(sort_values[x[0]]))))
            else:
                if sort_float:
                    fileinfos = [x for x in fileinfos if sort_values[x[0]] != "0"]
                    fileinfos.sort(key=lambda x: float(sort_values[x[0]]))
                    fileinfos = dict(fileinfos)
                else:
                    fileinfos = dict(sorted(fileinfos, key=lambda x: natural_keys(sort_values[x[0]])))
            filenames = [finfo for finfo in fileinfos]
        else:
            filenames = [finfo for finfo in fileinfos]
    return filenames

def get_image_thumbnail(image_list):
    logger.debug("get_image_thumbnail")
    optimized_cache = os.path.join(tempfile.gettempdir(),"optimized")
    os.makedirs(optimized_cache,exist_ok=True)
    thumbnail_list = []
    for image_path in image_list:
        image_path_hash = hashlib.md5(image_path.encode("utf-8")).hexdigest()
        cache_image_path = os.path.join(optimized_cache, image_path_hash + ".jpg")
        if os.path.isfile(cache_image_path):
            thumbnail_list.append(cache_image_path)
        else:
            try:
                image = Image.open(image_path)
            except OSError:
                # If PIL cannot open the image, use the original path
                thumbnail_list.append(image_path)
                continue
            width, height = image.size
            left = (width - min(width, height)) / 2
            top = (height - min(width, height)) / 2
            right = (width + min(width, height)) / 2
            bottom = (height + min(width, height)) / 2
            thumbnail = image.crop((left, top, right, bottom))
            thumbnail.thumbnail((opts.image_browser_thumbnail_size, opts.image_browser_thumbnail_size))
            if thumbnail.mode != "RGB":
                thumbnail = thumbnail.convert("RGB")
            try:
                thumbnail.save(cache_image_path, "JPEG")
                thumbnail_list.append(cache_image_path)
            except FileNotFoundError:
                # Cannot save cache, use PIL object
                thumbnail_list.append(thumbnail)
    return thumbnail_list

def set_tooltip_info(image_list):
    image_browser_img_info = {}
    conn, cursor = wib_db.transaction_begin()
    for filename in image_list:
        x, y = wib_db.select_x_y(cursor, filename)
        image_browser_img_info[filename] = {"x": x, "y": y}
    wib_db.transaction_end(conn, cursor)
    image_browser_img_info_json = json.dumps(image_browser_img_info)
    return image_browser_img_info_json

def get_image_page(img_path, page_index, filenames, keyword, sort_by, sort_order, tab_base_tag_box, img_path_depth, ranking_filter, ranking_filter_min, ranking_filter_max, aes_filter_min, aes_filter_max, exif_keyword, negative_prompt_search, use_regex, case_sensitive):
    logger.debug("get_image_page")
    if img_path == "":
        return [], page_index, [],  "", "",  "", 0, "", None, "", "[]"

    # Set temp_dir from webui settings, so gradio uses it
    if shared.opts.temp_dir != "":
        tempfile.tempdir = shared.opts.temp_dir
        
    img_path, _ = pure_path(img_path)
    filenames = get_all_images(img_path, sort_by, sort_order, keyword, tab_base_tag_box, img_path_depth, ranking_filter, ranking_filter_min, ranking_filter_max, aes_filter_min, aes_filter_max, exif_keyword, negative_prompt_search, use_regex, case_sensitive)
    page_index = int(page_index)
    length = len(filenames)
    max_page_index = math.ceil(length / num_of_imgs_per_page)
    page_index = max_page_index if page_index == -1 else page_index
    page_index = 1 if page_index < 1 else page_index
    page_index = max_page_index if page_index > max_page_index else page_index
    idx_frm = (page_index - 1) * num_of_imgs_per_page
    image_list = filenames[idx_frm:idx_frm + num_of_imgs_per_page]

    if opts.image_browser_scan_exif and opts.image_browser_img_tooltips:
        image_browser_img_info = set_tooltip_info(image_list)
    else:
        image_browser_img_info = "[]"

    if opts.image_browser_use_thumbnail:
        thumbnail_list = get_image_thumbnail(image_list)
    else:
        thumbnail_list = image_list

    visible_num = num_of_imgs_per_page if  idx_frm + num_of_imgs_per_page < length else length % num_of_imgs_per_page 
    visible_num = num_of_imgs_per_page if visible_num == 0 else visible_num

    load_info = "<div style='color:#999' align='center'>"
    load_info += f"{length} images in this directory, divided into {int((length + 1) // num_of_imgs_per_page  + 1)} pages"
    load_info += "</div>"

    return filenames, gr.update(value=page_index, label=f"Page Index ({page_index}/{max_page_index})"), thumbnail_list,  "", "",  "", visible_num, load_info, None, json.dumps(image_list), image_browser_img_info, gr.update(visible=True)

def get_current_file(tab_base_tag_box, num, page_index, filenames):
    file = filenames[int(num) + int((page_index - 1) * num_of_imgs_per_page)]
    return file

def pnginfo2html(pnginfo, items):
    items = {**{'parameters': pnginfo}, **items}

    info = ''
    for key, text in items.items():
        info += f"""
            <div>
            <p><b>{plaintext_to_html(str(key))}</b></p>
            <p>{plaintext_to_html(str(text))}</p>
            </div>
            """.strip()+"\n"

    return info

def show_image_info(tab_base_tag_box, num, page_index, filenames, turn_page_switch, image_gallery, req:gr.Request):
    callPseudonym(req)
    logger.debug(f"show_image_info: tab_base_tag_box, num, page_index, len(filenames), num_of_imgs_per_page: {tab_base_tag_box}, {num}, {page_index}, {len(filenames)}, {num_of_imgs_per_page}")
    if len(filenames) == 0:
        # This should only happen if webui was stopped and started again and the user clicks on one of the still displayed images.
        # The state with the filenames will be empty then. In that case we return None to prevent further errors and force a page refresh.
        turn_page_switch = -turn_page_switch
        file = None
        tm =  None
        info = ""
    else:
        file_num = int(num) + int(
            (page_index - 1) * num_of_imgs_per_page)
        if file_num >= len(filenames):
            # Last image to the right is deleted, page refresh
            turn_page_switch = -turn_page_switch
            file = None
            tm =  None
            info = ""
        else:
            file = filenames[file_num]
            tm =   "<div style='color:#999' align='right'>" + time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(os.path.getmtime(file))) + "</div>"
            pnginfo = exif_cache.get(file)
            if pnginfo and not opts.image_browser_info_add:
                items = {}
                info = pnginfo2html(pnginfo, items)
            else:
                try:
                    with Image.open(file) as image:
                        _, geninfo, info = modules.extras.run_pnginfo(image)
                except UnidentifiedImageError as e:
                    info = ""
                    logger.warning(f"UnidentifiedImageError: {e}")
            if opts.image_browser_use_thumbnail:
                image_gallery = [image['name'] for image in image_gallery]
                image_gallery[int(num)] = filenames[file_num]
                return file, tm, num, file, turn_page_switch, info, image_gallery
            else:
                return file, tm, num, file, turn_page_switch, info

def change_dir(img_dir, path_recorder, load_switch, img_path_browser, img_path_depth, img_path, req:gr.Request):
    callPseudonym(req)
    warning = None
    img_path, _ = pure_path(img_path)
    img_path_depth_org = img_path_depth
    if img_dir == none_select:
        return warning, gr.update(visible=False), img_path_browser, path_recorder, load_switch, img_path, img_path_depth
    else:
        img_dir, img_path_depth = pure_path(img_dir)
        if warning is None:
            try:
                if os.path.exists(img_dir):
                    try:
                        f = os.listdir(img_dir)                
                    except:
                        warning = f"'{img_dir} is not a directory"
                else:
                    warning = "The directory does not exist"
            except:
                warning = "The format of the directory is incorrect"   
        if warning is None: 
            return "", gr.update(visible=True), img_path_browser, path_recorder, img_dir, img_dir, img_path_depth
        else:
            return warning, gr.update(visible=False), img_path_browser, path_recorder, load_switch, img_path, img_path_depth_org

def update_move_text_one(btn):
    btn_text = " ".join(btn.split()[1:])
    return f"{copy_move[opts.image_browser_copy_image]} {btn_text}"

def update_move_text(favorites_btn, to_dir_btn):
    return update_move_text_one(favorites_btn), update_move_text_one(to_dir_btn)

def get_ranking(filename):
    ranking_value = wib_db.select_ranking(filename)
    return ranking_value, None

def img_file_name_changed(img_file_name, favorites_btn, to_dir_btn):
    ranking_current, ranking = get_ranking(img_file_name)
    favorites_btn, to_dir_btn = update_move_text(favorites_btn, to_dir_btn)

    return ranking_current, ranking, "", favorites_btn, to_dir_btn

def update_exif(img_file_name, key, value):
    image = Image.open(img_file_name)
    geninfo, items = images.read_info_from_image(image)  
    if geninfo is not None:
        if f"{key}: " in geninfo:
            if value == "None":
                geninfo = re.sub(f', {key}: \d+(\.\d+)*', '', geninfo)
            else:
                geninfo = re.sub(f'{key}: \d+(\.\d+)*', f'{key}: {value}', geninfo)
        else:
            geninfo = f'{geninfo}, {key}: {value}'
    
    original_time = os.path.getmtime(img_file_name)
    images.save_image(image, os.path.dirname(img_file_name), "", extension=os.path.splitext(img_file_name)[1][1:], info=geninfo, forced_filename=os.path.splitext(os.path.basename(img_file_name))[0], save_to_dirs=False)
    os.utime(img_file_name, (original_time, original_time))
    return geninfo

def update_ranking(img_file_name, ranking_current, ranking, img_file_info):
    # ranking = None is different than ranking = "None"! None means no radio button selected. "None" means radio button called "None" selected.
    if ranking is None:
        return ranking_current, None, img_file_info

    saved_ranking, _ = get_ranking(img_file_name)
    if saved_ranking != ranking:
        wib_db.update_ranking(img_file_name, ranking)
        if opts.image_browser_ranking_pnginfo and any(img_file_name.endswith(ext) for ext in image_ext_list):
            img_file_info = update_exif(img_file_name, "Ranking", ranking)
    return ranking, None, img_file_info
            
def create_tab(tab: ImageBrowserTab, current_gr_tab: gr.Tab):
    global init, exif_cache, aes_cache, openoutpaint, controlnet, js_dummy_return
    dir_name = None
    others_dir = False
    maint = False
    standard_ui = True
    path_recorder = {}
    path_recorder_formatted = []
    path_recorder_unformatted = []
    
    if init:
        exif_cache = wib_db.load_exif_data(exif_cache)
        aes_cache = wib_db.load_exif_data_by_key(aes_cache, "aesthetic_score", "Score")
        init = False
    
    path_recorder, path_recorder_formatted, path_recorder_unformatted = read_path_recorder()
    openoutpaint = check_ext("openoutpaint")
    controlnet = check_ext("controlnet")

    if tab.name == "Others":
        others_dir = True
        standard_ui = False
    elif tab.name == "Maintenance":
        maint = True
        standard_ui = False
    else:
        dir_name = tab.path

    if standard_ui:
        dir_name = str(Path(dir_name))
        if not os.path.exists(dir_name) and tab.name != "All":
            os.makedirs(dir_name)

    with gr.Row():                 
        path_recorder = gr.State(path_recorder)
        with gr.Column(scale=10):
            warning_box = gr.HTML("<p>&nbsp", elem_id=f"{tab.base_tag}_image_browser_warning_box")
        with gr.Column(scale=5, visible=(tab.name==favorite_tab_name)):
            gr.HTML(f"<p>Favorites path from settings: {opts.outdir_save}")

    with gr.Row(visible=others_dir):
        with gr.Column(scale=10):
            img_path = gr.Textbox(dir_name, label="Images directory", placeholder="Input images directory", interactive=others_dir)
        with gr.Column(scale=1):
            img_path_depth = gr.Number(value="0", label="Sub directory depth")
        with gr.Column(scale=1):
            img_path_save_button = gr.Button(value="Add to / replace in saved directories")

    with gr.Row(visible=others_dir):
        with gr.Column(scale=10):
            img_path_browser = gr.Dropdown(choices=path_recorder_formatted, label="Saved directories")
        with gr.Column(scale=1):
            img_path_remove_button = gr.Button(value="Remove from saved directories")

    with gr.Row(visible=others_dir): 
        with gr.Column(scale=10):
            img_path_subdirs = gr.Dropdown(choices=[none_select], value=none_select, label="Sub directories", interactive=True, elem_id=f"{tab.base_tag}_img_path_subdirs")
        with gr.Column(scale=1):
            img_path_subdirs_button = gr.Button(value="Get sub directories")
    
    with gr.Row(visible=standard_ui, elem_id=f"{tab.base_tag}_image_browser") as main_panel:
        with gr.Column():  
            with gr.Row():    
                with gr.Column(scale=2):    
                    with gr.Row(elem_id=f"{tab.base_tag}_image_browser_gallery_controls") as gallery_controls_panel:
                        with gr.Column(scale=2, min_width=20):
                            first_page = gr.Button("First Page", elem_id=f"{tab.base_tag}_control_image_browser_first_page")
                        with gr.Column(scale=2, min_width=20):
                            prev_page = gr.Button("Prev Page", elem_id=f"{tab.base_tag}_control_image_browser_prev_page")
                        with gr.Column(scale=2, min_width=20):
                            page_index = gr.Number(value=1, label="Page Index", elem_id=f"{tab.base_tag}_control_image_browser_page_index")
                        with gr.Column(scale=1, min_width=20):
                            refresh_index_button = ToolButton(value=refresh_symbol, elem_id=f"{tab.base_tag}_control_image_browser_refresh_index")
                        with gr.Column(scale=2, min_width=20):
                            next_page = gr.Button("Next Page", elem_id=f"{tab.base_tag}_control_image_browser_next_page")
                        with gr.Column(scale=2, min_width=20):
                            end_page = gr.Button("End Page", elem_id=f"{tab.base_tag}_control_image_browser_end_page") 
                    with gr.Row(visible=False) as ranking_panel:
                        with gr.Column(scale=1, min_width=20):
                            ranking_current = gr.Textbox(value="None", label="Current ranking", interactive=False)
                        with gr.Column(scale=4, min_width=20):
                            ranking = gr.Radio(choices=["1", "2", "3", "4", "5", "None"], label="Set ranking to", elem_id=f"{tab.base_tag}_control_image_browser_ranking", interactive=True)
                    with gr.Row():
                        image_gallery = gr.Gallery(show_label=False, elem_id=f"{tab.base_tag}_image_browser_gallery").style(columns=opts.image_browser_page_columns, height=("max-content" if opts.image_browser_height_auto else None))
                    with gr.Row() as delete_panel:
                        with gr.Column(scale=1):
                            delete_num = gr.Number(value=1, interactive=True, label="delete next", elem_id=f"{tab.base_tag}_image_browser_del_num")
                            delete_confirm = gr.Checkbox(value=False, label="also delete off-screen images")
                        with gr.Column(scale=3):
                            delete = gr.Button('Delete', elem_id=f"{tab.base_tag}_image_browser_del_img_btn")
                    with gr.Row() as info_add_panel:
                        with gr.Box(visible=opts.image_browser_info_add):
                            gr.HTML("<h3>Additional Generation Info</h3>")
                            img_file_info_add = gr.HTML()

                with gr.Column(scale=1): 
                    with gr.Row() as sort_panel:
                        sort_by = gr.Dropdown(value="date", choices=["path name", "date", "aesthetic_score", "random", "cfg scale", "steps", "seed", "sampler", "size", "model", "model hash", "ranking"], label="Sort by")
                        sort_order = ToolButton(value=down_symbol)
                    with gr.Row() as filename_search_panel:
                        filename_keyword_search = gr.Textbox(value="", label="Filename keyword search")
                    with gr.Box() as exif_search_panel:
                        with gr.Row():
                                exif_keyword_search = gr.Textbox(value="", label="EXIF keyword search")
                                negative_prompt_search = gr.Radio(value="No", choices=["No", "Yes", "Only"], label="Search negative prompt", interactive=True)
                        with gr.Row():
                                case_sensitive = gr.Checkbox(value=False, label="case sensitive")
                                use_regex = gr.Checkbox(value=False, label=r"regex - e.g. ^(?!.*Hires).*$")
                    with gr.Box() as ranking_filter_panel:
                        with gr.Row():
                            ranking_filter = gr.Radio(value="All", choices=["All", "1", "2", "3", "4", "5", "None", "Min-max"], label="Ranking filter", interactive=True)
                        with gr.Row():
                            with gr.Column(scale=2, min_width=20):
                                ranking_filter_min = gr.Textbox(value="1", label="Minimum ranking", interactive=False)
                            with gr.Column(scale=2, min_width=20):
                                ranking_filter_max = gr.Textbox(value="5", label="Maximum ranking", interactive=False)
                            with gr.Column(scale=4, min_width=20):
                                gr.Textbox(value="Choose Min-max to activate these controls", label="", interactive=False)
                    with gr.Box() as aesthetic_score_filter_panel:
                        with gr.Row():
                            aes_filter_min = gr.Textbox(value="", label="Minimum score")
                            aes_filter_max = gr.Textbox(value="", label="Maximum score")
                    with gr.Row() as generation_info_panel:
                        img_file_info = gr.Textbox(label="Generation Info", interactive=False, lines=6,elem_id=f"{tab.base_tag}_image_browser_file_info")
                    with gr.Row() as filename_panel:
                        img_file_name = gr.Textbox(value="", label="File Name", interactive=False)
                    with gr.Row() as filetime_panel:
                        img_file_time= gr.HTML()
                    with gr.Row() as open_folder_panel:
                        if tab.name == "All":
                            open_folder_button = gr.Button(folder_symbol, visible=False)
                        else:
                            open_folder_button = gr.Button(folder_symbol, visible=standard_ui or others_dir)
                            gr.HTML("&nbsp")
                            gr.HTML("&nbsp")
                            gr.HTML("&nbsp")
                    with gr.Row(elem_id=f"{tab.base_tag}_image_browser_button_panel", visible=False) as button_panel:
                        with gr.Column():
                            with gr.Row():
                                if tab.name == favorite_tab_name:
                                    favorites_btn_show = False
                                else:
                                    favorites_btn_show = True
                                favorites_btn = gr.Button(f'{copy_move[opts.image_browser_copy_image]} to favorites', elem_id=f"{tab.base_tag}_image_browser_favorites_btn", visible=favorites_btn_show)
                                try:
                                    send_to_buttons = modules.generation_parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint", "extras"])
                                except:
                                    pass
                                sendto_openoutpaint = gr.Button("Send to openOutpaint", elem_id=f"{tab.base_tag}_image_browser_openoutpaint_btn", visible=openoutpaint)
                            with gr.Row(visible=controlnet):
                                sendto_controlnet_txt2img = gr.Button("Send to txt2img ControlNet", visible=controlnet)
                                sendto_controlnet_img2img = gr.Button("Send to img2img ControlNet", visible=controlnet)
                                controlnet_max = opts.data.get("control_net_max_models_num", 1)
                                sendto_controlnet_num = gr.Dropdown([str(i) for i in range(controlnet_max)], label="ControlNet number", value="0", interactive=True, visible=(controlnet and controlnet_max > 1))
                                if controlnet_max is None:
                                    sendto_controlnet_type = gr.Textbox(value="none", visible=False)
                                elif controlnet_max == 1:
                                    sendto_controlnet_type = gr.Textbox(value="single", visible=False)
                                else:
                                    sendto_controlnet_type = gr.Textbox(value="multiple", visible=False)
                    with gr.Row(elem_id=f"{tab.base_tag}_image_browser_to_dir_panel", visible=False) as to_dir_panel:
                        with gr.Box():
                            with gr.Row():
                                to_dir_path = gr.Textbox(label="Directory path")
                            with gr.Row():
                                to_dir_saved = gr.Dropdown(choices=path_recorder_unformatted, label="Saved directories")
                            with gr.Row():
                                to_dir_btn = gr.Button(f'{copy_move[opts.image_browser_copy_image]} to directory', elem_id=f"{tab.base_tag}_image_browser_to_dir_btn")

                    with gr.Row():
                        collected_warning = gr.HTML()

                    with gr.Row(visible=False):
                        renew_page = gr.Button("Renew Page", elem_id=f"{tab.base_tag}_image_browser_renew_page")
                        visible_img_num = gr.Number()                     
                        tab_base_tag_box = gr.Textbox(tab.base_tag)
                        image_index = gr.Textbox(value=-1, elem_id=f"{tab.base_tag}_image_browser_image_index")
                        set_index = gr.Button('set_index', elem_id=f"{tab.base_tag}_image_browser_set_index")
                        filenames = gr.State([])
                        hidden = gr.Image(type="pil", elem_id=f"{tab.base_tag}_image_browser_hidden_image")
                        image_page_list = gr.Textbox(elem_id=f"{tab.base_tag}_image_browser_image_page_list")
                        info1 = gr.Textbox()
                        info2 = gr.Textbox()
                        load_switch = gr.Textbox(value="load_switch", label="load_switch")
                        to_dir_load_switch = gr.Textbox(value="to dir load_switch", label="to_dir_load_switch")
                        turn_page_switch = gr.Number(value=1, label="turn_page_switch")
                        select_image = gr.Number(value=1)
                        img_path_add = gr.Textbox(value="add")
                        img_path_remove = gr.Textbox(value="remove")
                        favorites_path = gr.Textbox(value=opts.outdir_save)
                        mod_keys = ""
                        if opts.image_browser_mod_ctrl_shift:
                            mod_keys = f"{mod_keys}CS"
                        elif opts.image_browser_mod_shift:
                            mod_keys = f"{mod_keys}S"
                        image_browser_mod_keys = gr.Textbox(value=mod_keys, elem_id=f"{tab.base_tag}_image_browser_mod_keys")
                        image_browser_prompt = gr.Textbox(elem_id=f"{tab.base_tag}_image_browser_prompt")
                        image_browser_neg_prompt = gr.Textbox(elem_id=f"{tab.base_tag}_image_browser_neg_prompt")
                        js_logs = gr.Textbox()
                        image_browser_img_info = gr.Textbox(value="[]", elem_id=f"{tab.base_tag}_image_browser_img_info")

    # Maintenance tab
    with gr.Row(visible=maint): 
        with gr.Column(scale=4):
            gr.HTML(f"{caution_symbol} Caution: You should only use these options if you know what you are doing. {caution_symbol}")
        with gr.Column(scale=3):
            maint_wait = gr.HTML("Status:")
        with gr.Column(scale=7):
            gr.HTML("&nbsp")
    with gr.Row(visible=maint): 
        maint_last_msg = gr.Textbox(label="Last message", interactive=False)
    with gr.Row(visible=maint): 
        with gr.Column(scale=1):
            maint_exif_rebuild = gr.Button(value="Rebuild exif cache")
        with gr.Column(scale=1):
            maint_exif_delete_0 = gr.Button(value="Delete 0-entries from exif cache")
        with gr.Column(scale=10):
            gr.HTML(visible=False)
    with gr.Row(visible=maint): 
        with gr.Column(scale=2):
            maint_update_dirs = gr.Button(value="Update directory names in database")
        with gr.Column(scale=2, min_width=40):
            maint_update_dirs_path_recorder = gr.Checkbox(value=True, label="path_recorder")
        with gr.Column(scale=2, min_width=40):
            maint_update_dirs_exif_data = gr.Checkbox(value=True, label="exif_data")
        with gr.Column(scale=2, min_width=40):
            maint_update_dirs_ranking = gr.Checkbox(value=True, label="ranking")
        with gr.Column(scale=10):
            maint_update_dirs_from = gr.Textbox(label="From (full path)")
        with gr.Column(scale=10):
            maint_update_dirs_to = gr.Textbox(label="to (full path)")
    with gr.Row(visible=maint): 
        with gr.Column(scale=1):
            maint_recreate_hash = gr.Button(value="Recreate hash for existing files")
        with gr.Column(scale=10):
            gr.HTML(visible=False)
    with gr.Row(visible=maint): 
        with gr.Column(scale=1):
            maint_reapply_ranking = gr.Button(value="Reapply ranking after moving files")
        with gr.Column(scale=10):
            gr.HTML(visible=False)
    with gr.Row(visible=maint): 
        with gr.Column(scale=1):
            maint_restart_debug = gr.Button(value="Restart debug")
        with gr.Column(scale=10):
            gr.HTML(visible=False)
    with gr.Row(visible=maint): 
        with gr.Column(scale=1):
            maint_get_js_logs = gr.Button(value="Get javascript logs")
        with gr.Column(scale=10):
            maint_show_logs = gr.Textbox(label="Javascript logs", lines=10, interactive=False)
    with gr.Row(visible=False): 
        with gr.Column(scale=1):
            maint_rebuild_ranking = gr.Button(value="Rebuild ranking from exif info")
        with gr.Column(scale=10):
            gr.HTML(visible=False)

    # Hide components based on opts.image_browser_hidden_components
    hidden_component_map = {
        "Sort by": sort_panel,
        "Filename keyword search": filename_search_panel,
        "EXIF keyword search": exif_search_panel,
        "Ranking Filter": ranking_filter_panel,
        "Aesthestic Score": aesthetic_score_filter_panel,
        "Generation Info": generation_info_panel,
        "File Name": filename_panel,
        "File Time": filetime_panel,
        "Open Folder": open_folder_panel,
        "Send to buttons": button_panel,
        "Copy to directory": to_dir_panel,
        "Gallery Controls Bar": gallery_controls_panel,
        "Ranking Bar": ranking_panel,
        "Delete Bar": delete_panel,
        "Additional Generation Info": info_add_panel
    }

    if set(hidden_component_map.keys()) != set(components_list):
        logger.warning(f"Invalid items present in either hidden_component_map or components_list. Make sure when adding new components they are added to both.")

    override_hidden = set()
    if hasattr(opts, "image_browser_hidden_components"):
        for item in opts.image_browser_hidden_components:
            hidden_component_map[item].visible = False
            override_hidden.add(hidden_component_map[item])

    change_dir_outputs = [warning_box, main_panel, img_path_browser, path_recorder, load_switch, img_path, img_path_depth]
    img_path.submit(change_dir, inputs=[img_path, path_recorder, load_switch, img_path_browser, img_path_depth, img_path], outputs=change_dir_outputs, show_progress=opts.image_browser_show_progress)
    img_path_browser.change(change_dir, inputs=[img_path_browser, path_recorder, load_switch, img_path_browser, img_path_depth, img_path], outputs=change_dir_outputs, show_progress=opts.image_browser_show_progress)
    # img_path_browser.change(browser2path, inputs=[img_path_browser], outputs=[img_path])
    to_dir_saved.change(change_dir, inputs=[to_dir_saved, path_recorder, to_dir_load_switch, to_dir_saved, img_path_depth, to_dir_path], outputs=[warning_box, main_panel, to_dir_saved, path_recorder, to_dir_load_switch, to_dir_path, img_path_depth], show_progress=opts.image_browser_show_progress)

    #delete
    delete.click(
        fn=delete_image,
        inputs=[tab_base_tag_box, delete_num, img_file_name, filenames, image_index, visible_img_num, delete_confirm, turn_page_switch, image_page_list],
        outputs=[filenames, delete_num, turn_page_switch, visible_img_num, image_gallery, select_image, image_page_list],
        show_progress=opts.image_browser_show_progress
    ).then(
        fn=None,
        _js="image_browser_select_image",
        inputs=[tab_base_tag_box, image_index, select_image],
        outputs=[js_dummy_return],
        show_progress=opts.image_browser_show_progress
    )

    to_dir_btn.click(save_image, inputs=[img_file_name, filenames, page_index, turn_page_switch, to_dir_path], outputs=[collected_warning, filenames, page_index, turn_page_switch], show_progress=opts.image_browser_show_progress)

    #turn page

    # Using callPseudonym which is defined at the very beginning

    def firstPage(s, req: gr.Request):
        callPseudonym(req)
        return (1,-s)
    
    def nextPage(s, p, req: gr.Request):
        callPseudonym(req)
        return (p + 1, -s)
    
    def prevPage(s, p, req: gr.Request):
        callPseudonym(req)
        return (p - 1, -s)
    
    def endPage(s, req: gr.Request):
        callPseudonym(req)
        return (-1, -s)
    
    def oneMinusS(s, req: gr.Request):
        """Called by the following gradio elements :
        - load_switch
        - filename_keyword_search
        - exif_keyword_search
        - ranking_filter_min / max
        - aes_filter_min / max
        - sort_by"""
        callPseudonym(req)
        return (1, -s)
    
    def minusS(s, req: gr.Request):
        """Called by the following gradio elements :
        - page_index
        - renew_page
        - img_path_depth"""
        callPseudonym(req)
        return -s

    def refreshIndexButton(s,p, req: gr.Request):
        callPseudonym(req)
        return (p, -s)

    first_page.click(fn=firstPage , inputs=[turn_page_switch], outputs=[page_index, turn_page_switch], show_progress=opts.image_browser_show_progress)
    next_page.click(fn=nextPage, inputs=[page_index, turn_page_switch], outputs=[page_index, turn_page_switch], show_progress=opts.image_browser_show_progress)
    prev_page.click(fn=prevPage, inputs=[page_index, turn_page_switch], outputs=[page_index, turn_page_switch], show_progress=opts.image_browser_show_progress)
    end_page.click(fn=endPage, inputs=[turn_page_switch], outputs=[page_index, turn_page_switch], show_progress=opts.image_browser_show_progress)
    load_switch.change(fn=oneMinusS, inputs=[turn_page_switch], outputs=[page_index, turn_page_switch], show_progress=opts.image_browser_show_progress)
    filename_keyword_search.submit(fn=oneMinusS, inputs=[turn_page_switch], outputs=[page_index, turn_page_switch], show_progress=opts.image_browser_show_progress)
    exif_keyword_search.submit(fn=oneMinusS, inputs=[turn_page_switch], outputs=[page_index, turn_page_switch], show_progress=opts.image_browser_show_progress)
    ranking_filter_min.submit(fn=oneMinusS, inputs=[turn_page_switch], outputs=[page_index, turn_page_switch], show_progress=opts.image_browser_show_progress)
    ranking_filter_max.submit(fn=oneMinusS, inputs=[turn_page_switch], outputs=[page_index, turn_page_switch], show_progress=opts.image_browser_show_progress)
    aes_filter_min.submit(fn=oneMinusS, inputs=[turn_page_switch], outputs=[page_index, turn_page_switch], show_progress=opts.image_browser_show_progress)
    aes_filter_max.submit(fn=oneMinusS, inputs=[turn_page_switch], outputs=[page_index, turn_page_switch], show_progress=opts.image_browser_show_progress)
    sort_by.change(fn=oneMinusS, inputs=[turn_page_switch], outputs=[page_index, turn_page_switch], show_progress=opts.image_browser_show_progress)
    page_index.submit(fn=minusS, inputs=[turn_page_switch], outputs=[turn_page_switch], show_progress=opts.image_browser_show_progress)
    renew_page.click(fn=minusS, inputs=[turn_page_switch], outputs=[turn_page_switch], show_progress=opts.image_browser_show_progress)
    refresh_index_button.click(fn=refreshIndexButton, inputs=[page_index, turn_page_switch], outputs=[page_index, turn_page_switch], show_progress=opts.image_browser_show_progress)
    img_path_depth.change(fn=minusS, inputs=[turn_page_switch], outputs=[turn_page_switch], show_progress=opts.image_browser_show_progress)

    hide_on_thumbnail_view = [delete_panel, button_panel, ranking_panel, to_dir_panel, info_add_panel]

    sort_order.click(
        fn=sort_order_flip,
        inputs=[turn_page_switch, sort_order],
        outputs=[page_index, turn_page_switch, sort_order],
        show_progress=opts.image_browser_show_progress
    )
    ranking_filter.change(
        fn=ranking_filter_settings,
        inputs=[page_index, turn_page_switch, ranking_filter],
        outputs=[page_index, turn_page_switch, ranking_filter_min, ranking_filter_max],
        show_progress=opts.image_browser_show_progress
    )

    # Others
    img_path_subdirs_button.click(
        fn=img_path_subdirs_get, 
        inputs=[img_path], 
        outputs=[img_path_subdirs],
        show_progress=opts.image_browser_show_progress
    )
    img_path_subdirs.change(
        fn=change_dir, 
        inputs=[img_path_subdirs, path_recorder, load_switch, img_path_browser, img_path_depth, img_path], 
        outputs=change_dir_outputs,
        show_progress=opts.image_browser_show_progress
    )
    img_path_save_button.click(
        fn=img_path_add_remove, 
        inputs=[img_path, path_recorder, img_path_add, img_path_depth], 
        outputs=[path_recorder, img_path_browser],
        show_progress=opts.image_browser_show_progress
    )
    img_path_remove_button.click(
        fn=img_path_add_remove, 
        inputs=[img_path, path_recorder, img_path_remove, img_path_depth],
        outputs=[path_recorder, img_path_browser],
        show_progress=opts.image_browser_show_progress
    )
    maint_exif_rebuild.click(
        fn=exif_rebuild,
        inputs=[maint_wait],
        outputs=[maint_wait, maint_last_msg],
        show_progress=True
    )
    maint_exif_delete_0.click(
        fn=exif_delete_0,
        inputs=[maint_wait],
        outputs=[maint_wait, maint_last_msg],
        show_progress=True
    )
    maint_update_dirs.click(
        fn=exif_update_dirs,
        inputs=[maint_update_dirs_path_recorder, maint_update_dirs_exif_data, maint_update_dirs_ranking, maint_update_dirs_from, maint_update_dirs_to, maint_wait],
        outputs=[maint_wait, maint_last_msg],
        show_progress=True
    )
    maint_recreate_hash.click(
        fn=recreate_hash,
        inputs=[maint_wait],
        outputs=[maint_wait, maint_last_msg],
        show_progress=True
    )
    maint_reapply_ranking.click(
        fn=reapply_ranking,
        inputs=[path_recorder, maint_wait],
        outputs=[maint_wait, maint_last_msg],
        show_progress=True
    )
    maint_restart_debug.click(
        fn=restart_debug,
        inputs=[maint_wait],
        outputs=[maint_wait, maint_last_msg],
        show_progress=True
    )
    maint_get_js_logs.click(
        fn=js_logs_output,
        _js="get_js_logs",
        inputs=[js_logs],
        outputs=[maint_show_logs],
        show_progress=True
    )

    # other functions
    if opts.image_browser_use_thumbnail:
        set_index_outputs = [img_file_name, img_file_time, image_index, hidden, turn_page_switch, img_file_info_add, image_gallery]
    else:
        set_index_outputs = [img_file_name, img_file_time, image_index, hidden, turn_page_switch, img_file_info_add]
    set_index.click(
        fn=show_image_info,
        _js="image_browser_get_current_img",
        inputs=[tab_base_tag_box, image_index, page_index, filenames, turn_page_switch, image_gallery],
        outputs=set_index_outputs,
        show_progress=opts.image_browser_show_progress
    ).then(
        fn=None,
        _js="image_browser_img_show_progress_update",
        inputs=[tab_base_tag_box],
        outputs=[js_dummy_return],
        show_progress=opts.image_browser_show_progress
    )

    set_index.click(fn=lambda:(gr.update(visible=delete_panel not in override_hidden), gr.update(visible=button_panel not in override_hidden), gr.update(visible=ranking_panel not in override_hidden), gr.update(visible=to_dir_panel not in override_hidden), gr.update(visible=info_add_panel not in override_hidden)), inputs=None, outputs=hide_on_thumbnail_view, show_progress=opts.image_browser_show_progress)

    favorites_btn.click(save_image, inputs=[img_file_name, filenames, page_index, turn_page_switch, favorites_path], outputs=[collected_warning, filenames, page_index, turn_page_switch], show_progress=opts.image_browser_show_progress)
    img_file_name.change(img_file_name_changed, inputs=[img_file_name, favorites_btn, to_dir_btn], outputs=[ranking_current, ranking, collected_warning, favorites_btn, to_dir_btn], show_progress=opts.image_browser_show_progress)
   
    hidden.change(fn=run_pnginfo, inputs=[hidden, img_path, img_file_name], outputs=[info1, img_file_info, info2, image_browser_prompt, image_browser_neg_prompt], show_progress=opts.image_browser_show_progress)
    
    #ranking
    ranking.change(update_ranking, inputs=[img_file_name, ranking_current, ranking, img_file_info], outputs=[ranking_current, ranking, img_file_info], show_progress=opts.image_browser_show_progress)
        
    try:
        modules.generation_parameters_copypaste.bind_buttons(send_to_buttons, hidden, img_file_info)
    except:
        pass
    
    if standard_ui:
        current_gr_tab.select(
            fn=tab_select, 
            inputs=[],
            outputs=[path_recorder, to_dir_saved],
            show_progress=opts.image_browser_show_progress
        )
        open_folder_button.click(
            fn=lambda: open_folder(dir_name),
            inputs=[],
            outputs=[],
            show_progress=opts.image_browser_show_progress
        )
    elif others_dir:
        open_folder_button.click(
            fn=open_folder,
            inputs=[img_path],
            outputs=[],
            show_progress=opts.image_browser_show_progress
        )
    if standard_ui or others_dir:
        turn_page_switch.change(
            fn=get_image_page, 
            inputs=[img_path, page_index, filenames, filename_keyword_search, sort_by, sort_order, tab_base_tag_box, img_path_depth, ranking_filter, ranking_filter_min, ranking_filter_max, aes_filter_min, aes_filter_max, exif_keyword_search, negative_prompt_search, use_regex, case_sensitive], 
            outputs=[filenames, page_index, image_gallery, img_file_name, img_file_time, img_file_info, visible_img_num, warning_box, hidden, image_page_list, image_browser_img_info],
            show_progress=opts.image_browser_show_progress
        ).then(
            fn=None,
            _js="image_browser_turnpage",
            inputs=[tab_base_tag_box],
            outputs=[js_dummy_return],
            show_progress=opts.image_browser_show_progress
        )
        turn_page_switch.change(fn=lambda:(gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)), inputs=None, outputs=hide_on_thumbnail_view, show_progress=opts.image_browser_show_progress)
        sendto_openoutpaint.click(
            fn=None,
            inputs=[tab_base_tag_box, image_index, image_browser_prompt, image_browser_neg_prompt],
            outputs=[js_dummy_return],
            _js="image_browser_openoutpaint_send",
            show_progress=opts.image_browser_show_progress        )
        sendto_controlnet_txt2img.click(
            fn=None,
            inputs=[tab_base_tag_box, image_index, sendto_controlnet_num, sendto_controlnet_type],
            outputs=[js_dummy_return],
            _js="image_browser_controlnet_send_txt2img",
            show_progress=opts.image_browser_show_progress
        )
        sendto_controlnet_img2img.click(
            fn=None,
            inputs=[tab_base_tag_box, image_index, sendto_controlnet_num, sendto_controlnet_type],
            outputs=[js_dummy_return],
            _js="image_browser_controlnet_send_img2img",
            show_progress=opts.image_browser_show_progress
        )

def run_pnginfo(image, image_path, image_file_name):
    if image is None:
        return '', '', '', '', ''
    try:
        geninfo, items = images.read_info_from_image(image)
        info = pnginfo2html(geninfo, items)
    except UnidentifiedImageError as e:
        geninfo = None
        info = ""
    
    if geninfo is None:
        try:
            filename = os.path.splitext(image_file_name)[0] + ".txt"
            geninfo = ""
            with open(filename) as f:
                for line in f:
                    geninfo += line
        except Exception:
            logger.warning(f"run_pnginfo: No EXIF in image or txt file")

    if openoutpaint:
        prompt, neg_prompt = wib_db.select_prompts(image_file_name)
        if prompt == "0":
            prompt = ""
        if neg_prompt == "0":
            neg_prompt = ""
    else:
        prompt = ""
        neg_prompt = ""

    return '', geninfo, info, prompt, neg_prompt


def on_ui_tabs():
    global num_of_imgs_per_page, loads_files_num, js_dummy_return
    num_of_imgs_per_page = int(opts.image_browser_page_columns * opts.image_browser_page_rows)
    loads_files_num = int(opts.image_browser_pages_perload * num_of_imgs_per_page)
    with gr.Blocks(analytics_enabled=False) as image_browser:
        gradio_needed = "3.23.0"
        if version.parse(gr.__version__) < version.parse(gradio_needed):
            gr.HTML(f'<p style="color: red; font-weight: bold;">You are running Gradio version {gr.__version__}. This version of the extension requires at least Gradio version {gradio_needed}.</p><p style="color: red; font-weight: bold;">For more details see <a href="https://github.com/AlUlkesh/stable-diffusion-webui-images-browser/issues/116#issuecomment-1493259585" target="_blank">https://github.com/AlUlkesh/stable-diffusion-webui-images-browser/issues/116#issuecomment-1493259585</a></p>')
        else:
            with gr.Tabs(elem_id="image_browser_tabs_container") as tabs:
                js_dummy_return = gr.Textbox(interactive=False, visible=False)
                for i, tab in enumerate(tabs_list):
                    with gr.Tab(tab.name, elem_id=f"{tab.base_tag}_image_browser_container") as current_gr_tab:
                        with gr.Blocks(analytics_enabled=False):
                            create_tab(tab, current_gr_tab)
            gr.Checkbox(value=opts.image_browser_preload, elem_id="image_browser_preload", visible=False)
            gr.Textbox(",".join( [tab.base_tag for tab in tabs_list] ), elem_id="image_browser_tab_base_tags_list", visible=False)
            gr.Checkbox(value=opts.image_browser_swipe, elem_id=f"image_browser_swipe", visible=False)

            javascript_level_value, (javascript_level, javascript_level_text) = debug_levels(arg_level="javascript")
            level_value, (level, level_text) = debug_levels(arg_text=opts.image_browser_debug_level)
            if level_value >= javascript_level_value:
                debug_level_option = level
            else:
                debug_level_option = ""
            gr.Textbox(value=debug_level_option, elem_id="image_browser_debug_level_option", visible=False)

    # Webui's ui_loadsave uses gradio labels as keys, this does not work with image browser, as the same labels are used on different tabs
    # For this reason the do_not_save_to_config attribute is added to each gradio element
    for key, value in image_browser.blocks.items():
        setattr(value, "do_not_save_to_config", True)

    return (image_browser, "Image Browser", "image_browser"),

def move_setting(cur_setting_name, old_setting_name, option_info, section, added):
    try:
        old_value = shared.opts.__getattr__(old_setting_name)
    except AttributeError:
        old_value = None
    try:
        new_value = shared.opts.__getattr__(cur_setting_name)
    except AttributeError:
        new_value = None
    if old_value is not None and new_value is None:
        # Add new option
        shared.opts.add_option(cur_setting_name, shared.OptionInfo(*option_info, section=section))
        shared.opts.__setattr__(cur_setting_name, old_value)
        added = added + 1
        # Remove old option
        shared.opts.data.pop(old_setting_name, None)

    return added

def on_ui_settings():
    # [current setting_name], [old setting_name], [default], [label], [component], [component_args]
    active_tabs_description = f"List of active tabs (separated by commas). Available options are {', '.join(default_tab_options)}. Custom folders are also supported by specifying their path."
    debug_level_choices = []
    for i in range(len(debug_level_types)):
        level_value, (level, level_text) = debug_levels(arg_value=i)
        debug_level_choices.append(level_text)

    image_browser_options = [
        ("image_browser_active_tabs", None, ", ".join(default_tab_options), active_tabs_description),
        ("image_browser_hidden_components", None, [], "Select components to hide", DropdownMulti, lambda: {"choices": components_list}),
        ("image_browser_with_subdirs", "images_history_with_subdirs", True, "Include images in sub directories"),
        ("image_browser_preload", "images_history_preload", False, "Preload images at startup for first tab"),
        ("image_browser_copy_image", "images_copy_image", False, "Move buttons copy instead of move"),
        ("image_browser_delete_message", "images_delete_message", True, "Print image deletion messages to the console"),
        ("image_browser_txt_files", "images_txt_files", True, "Move/Copy/Delete matching .txt files"),
        ("image_browser_debug_level", None, debug_level_choices[0], "Debug level", gr.Dropdown, lambda: {"choices": debug_level_choices}),
        ("image_browser_delete_recycle", "images_delete_recycle", True, "Use recycle bin when deleting images"),
        ("image_browser_scan_exif", "images_scan_exif", True, "Scan Exif-/.txt-data (initially slower, but required for many features to work)"),
        ("image_browser_mod_shift", None, False, "Change CTRL keybindings to SHIFT"),
        ("image_browser_mod_ctrl_shift", None, False, "or to CTRL+SHIFT"),
        ("image_browser_ranking_pnginfo", None, False, "Save ranking in image's pnginfo"),
        ("image_browser_page_columns", "images_history_page_columns", 6, "Number of columns on the page"),
        ("image_browser_page_rows", "images_history_page_rows", 6, "Number of rows on the page"),
        ("image_browser_pages_perload", "images_history_pages_perload", 20, "Minimum number of pages per load"),
        ("image_browser_height_auto", None, False, "Use automatic height for gallery (requires Gradio >= 3.36.0)"),
        ("image_browser_use_thumbnail", None, False, "Use optimized images in the thumbnail interface (significantly reduces the amount of data transferred)"),
        ("image_browser_thumbnail_size", None, 200, "Size of the thumbnails (px)"),
        ("image_browser_swipe", None, False, "Swipe left/right navigates to the next image"),
        ("image_browser_img_tooltips", None, True, "Enable thumbnail tooltips"),
        ("image_browser_show_progress", None, True, "Show progress indicator"),
        ("image_browser_info_add", None, False, "Show Additional Generation Info"),        
    ]

    section = ('image-browser', "Image Browser")
    # Move historic setting names to current names
    added = 0
    for cur_setting_name, old_setting_name, *option_info in image_browser_options:
        if old_setting_name is not None:
            added = move_setting(cur_setting_name, old_setting_name, option_info, section, added)
    if added > 0:
        shared.opts.save(shared.config_filename)

    for cur_setting_name, _, *option_info in image_browser_options:
        shared.opts.add_option(cur_setting_name, shared.OptionInfo(*option_info, section=section))

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)
