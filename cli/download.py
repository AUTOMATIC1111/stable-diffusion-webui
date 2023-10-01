#!/usr/bin/env python
import os
import time
import argparse
import tempfile
import urllib
import requests
import urllib3
import rich.progress as p
from rich import print # pylint: disable=redefined-builtin


pbar = p.Progress(p.TextColumn('[cyan]{task.description}'), p.DownloadColumn(), p.BarColumn(), p.TaskProgressColumn(), p.TimeRemainingColumn(), p.TimeElapsedColumn(), p.TransferSpeedColumn())
headers = {
    'Content-type': 'application/json',
    'User-Agent': 'Mozilla/5.0',
}


def get_filename(args, res):
    content_fn = (res.headers.get('content-disposition', '').split('filename=')[1]).strip().strip('\"') if 'filename=' in res.headers.get('content-disposition', '') else None
    return args.file or content_fn or next(tempfile._get_candidate_names()) # pylint: disable=protected-access


def download_requests(args):
    res = requests.get(args.url, timeout=30, headers=headers, verify=False, allow_redirects=True, stream=True)
    content_length = int(res.headers.get('content-length', 0))
    fn = get_filename(args, res)
    print(f'downloading: url={args.url} file={fn} size={content_length if content_length > 0 else "unknown"} lib=requests block={args.block}')
    with open(fn, 'wb') as f:
        with pbar:
            task = pbar.add_task(description="Download starting", total=content_length)
            for data in res.iter_content(args.block):
                f.write(data)
                pbar.update(task, advance=args.block, description="Downloading")
    return fn


def download_urllib(args):
    fn = ''
    req = urllib.request.Request(args.url, headers=headers)
    res = urllib.request.urlopen(req)
    res.getheader('content-length')
    content_length = int(res.getheader('content-length') or 0)
    fn = get_filename(args, res)
    print(f'downloading: url={args.url} file={fn} size={content_length if content_length > 0 else "unknown"} lib=urllib block={args.block}')
    with open(fn, 'wb') as f:
        with pbar:
            task = pbar.add_task(description="Download starting", total=content_length)
            while True:
                buf = res.read(args.block)
                if not buf:
                    break
                f.write(buf)
                pbar.update(task, advance=args.block, description="Downloading")
    return fn


def download_urllib3(args):
    http_pool = urllib3.PoolManager()
    res = http_pool.request('GET', args.url, preload_content=False, headers=headers)
    fn = get_filename(args, res)
    content_length = int(res.headers.get('content-length', 0))
    print(f'downloading: url={args.url} file={fn} size={content_length if content_length > 0 else "unknown"} lib=urllib3 block={args.block}')
    with open(fn, 'wb') as f:
        with pbar:
            task = pbar.add_task(description="Download starting", total=content_length)
            while True:
                buf = res.read(args.block)
                if not buf:
                    break
                f.write(buf)
                pbar.update(task, advance=args.block, description="Downloading")
    return fn


def download_httpx(args):
    try:
        import httpx
    except ImportError:
        print('httpx is not installed')
        return None
    with httpx.stream("GET", args.url, headers=headers, verify=False, follow_redirects=True) as res:
        fn = get_filename(args, res)
        content_length = int(res.headers.get('content-length', 0))
        print(f'downloading: url={args.url} file={fn} size={content_length if content_length > 0 else "unknown"} lib=httpx block=internal')
        with open(fn, 'wb') as f:
            with pbar:
                task = pbar.add_task(description="Download starting", total=content_length)
                for buf in res.iter_bytes():
                    f.write(buf)
                    pbar.update(task, advance=args.block, description="Downloading")
        return fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'downloader')
    parser.add_argument('--url', required=True, help="download url, required")
    parser.add_argument('--file', required=False, help="output file, default: autodetect")
    parser.add_argument('--lib', required=False, default='requests', choices=['urllib', 'urllib3', 'requests', 'httpx'], help="download mode, default: %(default)s")
    parser.add_argument('--block', required=False, type=int, default=16384, help="download block size, default: %(default)s")
    parsed = parser.parse_args()
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    try:
        t0 = time.time()
        if parsed.lib == 'requests':
            filename = download_requests(parsed)
        elif parsed.lib == 'urllib':
            filename = download_urllib(parsed)
        elif parsed.lib == 'urllib3':
            filename = download_urllib3(parsed)
        elif parsed.lib == 'httpx':
            filename = download_httpx(parsed)
        else:
            print(f'unknown download library: {parsed.lib}')
            exit(1)
        t1 = time.time()
        if filename is None:
            print(f'download error: args={parsed}')
            exit(1)
        speed = round(os.path.getsize(filename) / (t1 - t0) / 1024 / 1024, 3)
        print(f'download complete: url={parsed.url} file={filename} speed={speed} mb/s')
    except KeyboardInterrupt:
        print(f'download cancelled: args={parsed}')
    except Exception as e:
        print(f'download error: args={parsed} {e}')
