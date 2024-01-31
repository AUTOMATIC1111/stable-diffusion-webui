from ..core import *
from ..vision.data import *
from ipywidgets import widgets, Layout, Output, HBox, VBox, Text, BoundedIntText, Button, Dropdown, Box
from IPython.display import clear_output, display
from urllib.parse import quote
from bs4 import BeautifulSoup
import time

__all__ = ['ImageDownloader', 'download_google_images']

_img_sizes = {'>400*300':'isz:lt,islt:qsvga','>640*480':'isz:lt,islt:vga','>800*600':'isz:lt,islt:svga',
              '>1024*768':'visz:lt,islt:xga', '>2MP':'isz:lt,islt:2mp','>4MP':'isz:lt,islt:4mp','>6MP':'isz:lt,islt:6mp',
              '>8MP':'isz:lt,islt:8mp', '>10MP':'isz:lt,islt:10mp','>12MP':'isz:lt,islt:12mp','>15MP':'isz:lt,islt:15mp',
              '>20MP':'isz:lt,islt:20mp','>40MP':'isz:lt,islt:40mp','>70MP':'isz:lt,islt:70mp'}

class ImageDownloader():
    """
    Displays a widget that allows searching and downloading images from google images search
    in a Jupyter Notebook or Lab.
    """
    def __init__(self, path:Union[Path,str]='data'):
        "Setup path to save images to, init the UI, and render the widgets."
        self._path = Path(path)
        self._ui = self._init_ui()
        self.render()

    def _init_ui(self) -> VBox:
        "Initialize the widget UI and return the UI."
        self._search_input = Text(placeholder="What images to search for?")
        self._count_input = BoundedIntText(placeholder="How many pics?", value=10, min=1, max=5000, step=1,
                                           layout=Layout(width='60px'))
        self._size_input = Dropdown(options= _img_sizes.keys(), value='>400*300', layout=Layout(width='120px'))
        self._download_button = Button(description="Search & Download", icon="download", layout=Layout(width='200px'))
        self._download_button.on_click(self.on_download_button_click)
        self._output = Output()
        self._controls_pane  = HBox([self._search_input, self._count_input, self._size_input, self._download_button],
                                    layout=Layout(width='auto', height='40px'))
        self._heading = ""
        self._download_complete_heading = "<h3>Download complete. Here are a few images</h3>"
        self._preview_header = widgets.HTML(self._heading, layout=Layout(height='60px'))
        self._img_pane = Box(layout=Layout(display='inline'))
        return VBox([self._controls_pane, self._preview_header, self._img_pane])

    def render(self) -> None:
        clear_output()
        display(self._ui)

    def clear_imgs(self) -> None:
        "Clear the widget's images preview pane."
        self._preview_header.value = self._heading
        self._img_pane.children = tuple()

    def validate_search_input(self) -> bool:
        "Check if input value is empty."
        input = self._search_input
        if input.value == str(): input.layout = Layout(border="solid 2px red", height='auto')
        else:                    self._search_input.layout = Layout()
        return input.value != str()

    def on_download_button_click(self, btn) -> None:
        "Download button click handler: validate search term and download images."
        term = self._search_input.value
        limit = int(self._count_input.value)
        size = self._size_input.value
        if not self.validate_search_input(): return
        self.clear_imgs()
        downloaded_images = download_google_images(self._path, term, n_images=limit, size=size)
        self.display_images_widgets(downloaded_images[:min(limit, 12)])
        self._preview_header.value = self._download_complete_heading
        self.render()

    def display_images_widgets(self, fnames:list) -> None:
        "Display a few preview images in the notebook"
        imgs = [widgets.Image(value=open(f, 'rb').read(), width='200px') for f in fnames]
        self._img_pane.children = tuple(imgs)


def download_google_images(path:PathOrStr, search_term:str, size:str='>400*300', n_images:int=10, format:str='jpg',
                            max_workers:int=defaults.cpus, timeout:int=4) -> FilePathList:
    """
    Search for `n_images` images on Google, matching `search_term` and `size` requirements,
    download them into `path`/`search_term` and verify them, using `max_workers` threads.
    """
    label_path = Path(path)/search_term
    search_url = _search_url(search_term, size=size, format=format)
    if n_images <= 100: img_tuples = _fetch_img_tuples(search_url, format=format, n_images=n_images)
    else:               img_tuples = _fetch_img_tuples_webdriver(search_url, format=format, n_images=n_images)
    downloaded_images = _download_images(label_path, img_tuples, max_workers=max_workers, timeout=timeout)
    if len(downloaded_images) == 0: raise RuntimeError(f"Couldn't download any images.")
    verify_images(label_path, max_workers=max_workers)
    return get_image_files(label_path)
    
def _url_params(size:str='>400*300', format:str='jpg') -> str:
    "Build Google Images Search Url params and return them as a string."
    _fmts = {'jpg':'ift:jpg','gif':'ift:gif','png':'ift:png','bmp':'ift:bmp', 'svg':'ift:svg','webp':'webp','ico':'ift:ico'}
    if size not in _img_sizes: 
        raise RuntimeError(f"""Unexpected size argument value: {size}.
                    See `widgets.image_downloader._img_sizes` for supported sizes.""") 
    if format not in _fmts: 
        raise RuntimeError(f"Unexpected image file format: {format}. Use jpg, gif, png, bmp, svg, webp, or ico.")
    return "&tbs=" + _img_sizes[size] + "," + _fmts[format]

def _search_url(search_term:str, size:str='>400*300', format:str='jpg') -> str:
    "Return a Google Images Search URL for a given search term."
    return ('https://www.google.com/search?q=' + quote(search_term) +
            '&espv=2&biw=1366&bih=667&site=webhp&source=lnms&tbm=isch' +
            _url_params(size, format) + '&sa=X&ei=XosDVaCXD8TasATItgE&ved=0CAcQ_AUoAg')

def _img_fname(img_url:str) -> str:
    "Return image file name including the extension given its url."
    return img_url.split('/')[-1]

def _fetch_img_tuples(url:str, format:str='jpg', n_images:int=10) -> list:
    "Parse the Google Images Search for urls and return the image metadata as tuples (fname, url)."
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'}
    html = requests.get(url, headers=headers).text
    return _html_to_img_tuples(html, format=format, n_images=n_images)

def _html_to_img_tuples(html:str, format:str='jpg', n_images:int=10) -> list:    
    "Parse the google images html to img tuples containining `(fname, url)`"
    bs = BeautifulSoup(html, 'html.parser')
    img_tags = bs.find_all('div', {'class': 'rg_meta'})
    metadata_dicts = (json.loads(e.text) for e in img_tags)
    img_tuples = ((_img_fname(d['ou']), d['ou']) for d in metadata_dicts if d['ity'] == format)
    return list(itertools.islice(img_tuples, n_images))

def _fetch_img_tuples_webdriver(url:str, format:str='jpg', n_images:int=150) -> list:
    """
    Parse the Google Images Search for urls and return the image metadata as tuples (fname, url).
    Use this for downloads of >100 images. Requires `selenium`.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.common.keys import Keys
    except:
        print("""Looks like you're trying to download > 100 images and `selenium`
                is not installed. Try running `pip install selenium` to fix this. 
                You'll also need chrome and `chromedriver` installed.""")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    try: driver = webdriver.Chrome(chrome_options=options)
    except: print("""Error initializing chromedriver. 
                    Check if it's in your path by running `which chromedriver`""")
    driver.set_window_size(1440, 900)
    driver.get(url)

    for i in range(n_images // 100 + 1):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(0.5 + random.random()/2.0)

    n_available = len(driver.find_elements_by_css_selector("div.rg_meta"))
    if n_available < n_images:
        raise ValueError(f"Requested {n_images} images, but only found {n_available}.")

    html = driver.page_source
    driver.close()
    return _html_to_img_tuples(html, format=format, n_images=n_images)

def _download_images(label_path:PathOrStr, img_tuples:list, max_workers:int=defaults.cpus, timeout:int=4) -> FilePathList:
    """
    Downloads images in `img_tuples` to `label_path`. 
    If the directory doesn't exist, it'll be created automatically.
    Uses `parallel` to speed things up in `max_workers` when the system has enough CPU cores.
    If something doesn't work, try setting up `max_workers=0` to debug.
    """
    os.makedirs(Path(label_path), exist_ok=True)
    parallel( partial(_download_single_image, label_path, timeout=timeout), img_tuples, max_workers=max_workers)
    return get_image_files(label_path)

def _download_single_image(label_path:Path, img_tuple:tuple, i:int, timeout:int=4) -> None:
    """
    Downloads a single image from Google Search results to `label_path`
    given an `img_tuple` that contains `(fname, url)` of an image to download.
    `i` is just an iteration number `int`. 
    """
    suffix = re.findall(r'\.\w+?(?=(?:\?|$))', img_Tuple[1])
    suffix = suffix[0].lower() if len(suffix)>0  else '.jpg'
    fname = f"{i:08d}{suffix}"
    download_url(img_Tuple[1], label_path/fname, timeout=timeout)
