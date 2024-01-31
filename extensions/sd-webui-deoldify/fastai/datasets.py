from .core import *
import hashlib

__all__ = ['URLs', 'Config', 'untar_data', 'download_data', 'datapath4file', 'url2name', 'url2path']

MODEL_URL = 'http://files.fast.ai/models/'
URL = 'http://files.fast.ai/data/examples/'
class URLs():
    "Global constants for dataset and model URLs."
    LOCAL_PATH = Path.cwd()
    S3 = 'https://s3.amazonaws.com/fast-ai-'

    S3_IMAGE    = f'{S3}imageclas/'
    S3_IMAGELOC = f'{S3}imagelocal/'
    S3_NLP      = f'{S3}nlp/'
    S3_COCO     = f'{S3}coco/'
    S3_MODEL    = f'{S3}modelzoo/'

    # main datasets
    ADULT_SAMPLE        = f'{URL}adult_sample'
    BIWI_SAMPLE         = f'{URL}biwi_sample'
    CIFAR               = f'{URL}cifar10'
    COCO_SAMPLE         = f'{S3_COCO}coco_sample'
    COCO_TINY           = f'{URL}coco_tiny'
    HUMAN_NUMBERS       = f'{URL}human_numbers'
    IMDB                = f'{S3_NLP}imdb'
    IMDB_SAMPLE         = f'{URL}imdb_sample'
    ML_SAMPLE           = f'{URL}movie_lens_sample'
    MNIST_SAMPLE        = f'{URL}mnist_sample'
    MNIST_TINY          = f'{URL}mnist_tiny'
    MNIST_VAR_SIZE_TINY = f'{S3_IMAGE}mnist_var_size_tiny'
    PLANET_SAMPLE       = f'{URL}planet_sample'
    PLANET_TINY         = f'{URL}planet_tiny'
    IMAGENETTE          = f'{S3_IMAGE}imagenette'
    IMAGENETTE_160      = f'{S3_IMAGE}imagenette-160'
    IMAGENETTE_320      = f'{S3_IMAGE}imagenette-320'
    IMAGEWOOF           = f'{S3_IMAGE}imagewoof'
    IMAGEWOOF_160       = f'{S3_IMAGE}imagewoof-160'
    IMAGEWOOF_320       = f'{S3_IMAGE}imagewoof-320'

    # kaggle competitions download dogs-vs-cats -p {DOGS.absolute()}
    DOGS = f'{URL}dogscats'

    # image classification datasets
    CALTECH_101  = f'{S3_IMAGE}caltech_101'
    CARS         = f'{S3_IMAGE}stanford-cars'
    CIFAR_100    = f'{S3_IMAGE}cifar100'
    CUB_200_2011 = f'{S3_IMAGE}CUB_200_2011'
    FLOWERS      = f'{S3_IMAGE}oxford-102-flowers'
    FOOD         = f'{S3_IMAGE}food-101'
    MNIST        = f'{S3_IMAGE}mnist_png'
    PETS         = f'{S3_IMAGE}oxford-iiit-pet'

    # NLP datasets
    AG_NEWS                 = f'{S3_NLP}ag_news_csv'
    AMAZON_REVIEWS          = f'{S3_NLP}amazon_review_full_csv'
    AMAZON_REVIEWS_POLARITY = f'{S3_NLP}amazon_review_polarity_csv'
    DBPEDIA                 = f'{S3_NLP}dbpedia_csv'
    MT_ENG_FRA              = f'{S3_NLP}giga-fren'
    SOGOU_NEWS              = f'{S3_NLP}sogou_news_csv'
    WIKITEXT                = f'{S3_NLP}wikitext-103'
    WIKITEXT_TINY           = f'{S3_NLP}wikitext-2'
    YAHOO_ANSWERS           = f'{S3_NLP}yahoo_answers_csv'
    YELP_REVIEWS            = f'{S3_NLP}yelp_review_full_csv'
    YELP_REVIEWS_POLARITY   = f'{S3_NLP}yelp_review_polarity_csv'

    # Image localization datasets
    BIWI_HEAD_POSE     = f"{S3_IMAGELOC}biwi_head_pose"
    CAMVID             = f'{S3_IMAGELOC}camvid'
    CAMVID_TINY        = f'{URL}camvid_tiny'
    LSUN_BEDROOMS      = f'{S3_IMAGE}bedroom'
    PASCAL_2007        = f'{S3_IMAGELOC}pascal_2007'
    PASCAL_2012        = f'{S3_IMAGELOC}pascal_2012'

    #Pretrained models
    OPENAI_TRANSFORMER = f'{S3_MODEL}transformer'
    WT103_FWD          = f'{S3_MODEL}wt103-fwd'
    WT103_BWD          = f'{S3_MODEL}wt103-bwd'

# to create/update a checksum for ./mnist_var_size_tiny.tgz, run:
# python -c 'import fastai.datasets; print(fastai.datasets._check_file("mnist_var_size_tiny.tgz"))'
_checks = {
    URLs.ADULT_SAMPLE:(968212, '64eb9d7e23732de0b138f7372d15492f'),
    URLs.AG_NEWS:(11784419, 'b86f328f4dbd072486591cb7a5644dcd'),
    URLs.AMAZON_REVIEWS_POLARITY:(688339454, '676f7e5208ec343c8274b4bb085bc938'),
    URLs.AMAZON_REVIEWS:(643695014, '4a1196cf0adaea22f4bc3f592cddde90'),
    URLs.BIWI_HEAD_POSE:(452316199, '00f4ccf66e8cba184bc292fdc08fb237'),
    URLs.BIWI_SAMPLE:(593774, '9179f4c1435f4b291f0d5b072d60c2c9'),
    URLs.CALTECH_101:(131740031, 'd673425306e98ee4619fcdeef8a0e876'),
    URLs.CAMVID:(598913237, '648371e4f3a833682afb39b08a3ce2aa'),
    URLs.CAMVID_TINY:(2314212, '2cf6daf91b7a2083ecfa3e9968e9d915'),
    URLs.CARS:(1957803273, '9045d6673c9ced0889f41816f6bf2f9f'),
    URLs.CIFAR:(168168549, 'a5f8c31371b63a406b23368042812d3c'),
    URLs.CIFAR_100:(169168619, 'e5e65dcb54b9d3913f7b8a9ad6607e62'),
    URLs.COCO_SAMPLE:(3245877008, '006cd55d633d94b36ecaf661467830ec'),
    URLs.COCO_TINY:(801038, '367467451ac4fba79a647753c2c66d3a'),
    URLs.CUB_200_2011:(1150585339, 'd2acaa99439dff0483c7bbac1bfe2a92'),
    URLs.DBPEDIA:(68341743, '239c7837b9e79db34486f3de6a00e38e'),
    URLs.DOGS:(839285364, '3e483c8d6ef2175e9d395a6027eb92b7'),
    URLs.FLOWERS:(345236087, '5666e01c1311b4c67fcf20d2b3850a88'),
    URLs.FOOD:(5686607260, '1a540ebf1fb40b2bf3f2294234ba7907'),
    URLs.HUMAN_NUMBERS:(30252, '8a19c3bfa2bcb08cd787e741261f3ea2'),
    URLs.IMDB:(144440600, '90f9b1c4ff43a90d67553c9240dc0249'),
    URLs.IMDB_SAMPLE:(571827, '0842e61a9867caa2e6fbdb14fa703d61'),
    URLs.LSUN_BEDROOMS:(4579163978, '35d84f38f8a15fe47e66e460c8800d68'),
    URLs.ML_SAMPLE:(51790, '10961384dfe7c5181460390a460c1f77'),
    URLs.MNIST:(15683414, '03639f83c4e3d19e0a3a53a8a997c487'),
    URLs.MNIST_SAMPLE:(3214948, '2dbc7ec6f9259b583af0072c55816a88'),
    URLs.MNIST_TINY:(342207, '56143e8f24db90d925d82a5a74141875'),
    URLs.MNIST_VAR_SIZE_TINY:(565372, 'b71a930f4eb744a4a143a6c7ff7ed67f'),
    URLs.MT_ENG_FRA:(2598183296, '69573f58e2c850b90f2f954077041d8c'),
    URLs.OPENAI_TRANSFORMER:(432848315, '024b0d2203ebb0cd1fc64b27cf8af18e'),
    URLs.PASCAL_2007:(1636130334, 'a70574e9bc592bd3b253f5bf46ce12e3'),
    URLs.PASCAL_2012:(2611715776, '2ae7897038383836f86ce58f66b09e31'),
    URLs.PETS:(811706944, 'e4db5c768afd933bb91f5f594d7417a4'),
    URLs.PLANET_SAMPLE:(15523994, '8bfb174b3162f07fbde09b54555bdb00'),
    URLs.PLANET_TINY:(997569, '490873c5683454d4b2611fb1f00a68a9'),
    URLs.SOGOU_NEWS:(384269937, '950f1366d33be52f5b944f8a8b680902'),
    URLs.WIKITEXT:(190200704, '2dd8cf8693b3d27e9c8f0a7df054b2c7'),
    URLs.WIKITEXT_TINY:(4070055, '2a82d47a7b85c8b6a8e068dc4c1d37e7'),
    URLs.WT103_FWD:(105067061, '7d1114cd9684bf9d1ca3c9f6a54da6f9'),
    URLs.WT103_BWD:(105205312, '20b06f5830fd5a891d21044c28d3097f'),
    URLs.YAHOO_ANSWERS:(319476345, '0632a0d236ef3a529c0fa4429b339f68'),
    URLs.YELP_REVIEWS_POLARITY:(166373201, '48c8451c1ad30472334d856b5d294807'),
    URLs.YELP_REVIEWS:(196146755, '1efd84215ea3e30d90e4c33764b889db'),
}

#TODO: This can probably be coded more shortly and nicely.
class Config():
    "Creates a default config file 'config.yml' in $FASTAI_HOME (default `~/.fastai/`)"
    DEFAULT_CONFIG_LOCATION = os.path.expanduser(os.getenv('FASTAI_HOME', '~/.fastai'))
    DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_LOCATION + '/config.yml'
    DEFAULT_CONFIG = {
        'data_path': DEFAULT_CONFIG_LOCATION + '/data',
        'data_archive_path': DEFAULT_CONFIG_LOCATION + '/data',
        'model_path': DEFAULT_CONFIG_LOCATION + '/models'
    }

    @classmethod
    def get_key(cls, key):
        "Get the path to `key` in the config file."
        return cls.get().get(key, cls.DEFAULT_CONFIG.get(key,None))

    @classmethod
    def get_path(cls, path):
        "Get the `path` in the config file."
        return _expand_path(cls.get_key(path))

    @classmethod
    def data_path(cls):
        "Get the path to data in the config file."
        return cls.get_path('data_path')

    @classmethod
    def data_archive_path(cls):
        "Get the path to data archives in the config file."
        return cls.get_path('data_archive_path')

    @classmethod
    def model_path(cls):
        "Get the path to fastai pretrained models in the config file."
        return cls.get_path('model_path')

    @classmethod
    def get(cls, fpath=None, create_missing=True):
        "Retrieve the `Config` in `fpath`."
        fpath = _expand_path(fpath or cls.DEFAULT_CONFIG_PATH)
        if not fpath.exists() and create_missing: cls.create(fpath)
        assert fpath.exists(), f'Could not find config at: {fpath}. Please create'
        with open(fpath, 'r') as yaml_file: return yaml.safe_load(yaml_file)

    @classmethod
    def create(cls, fpath):
        "Creates a `Config` from `fpath`."
        fpath = _expand_path(fpath)
        assert(fpath.suffix == '.yml')
        if fpath.exists(): return
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, 'w') as yaml_file:
            yaml.dump(cls.DEFAULT_CONFIG, yaml_file, default_flow_style=False)

def _expand_path(fpath): return Path(fpath).expanduser()
def url2name(url): return url.split('/')[-1]

#TODO: simplify this mess
def url2path(url, data=True, ext:str='.tgz'):
    "Change `url` to a path."
    name = url2name(url)
    return datapath4file(name, ext=ext, archive=False) if data else modelpath4file(name, ext=ext)
def _url2tgz(url, data=True, ext:str='.tgz'):
    return datapath4file(f'{url2name(url)}{ext}', ext=ext) if data else modelpath4file(f'{url2name(url)}{ext}', ext=ext)

def modelpath4file(filename, ext:str='.tgz'):
    "Return model path to `filename`, checking locally first then in the config file."
    local_path = URLs.LOCAL_PATH/'models'/filename
    if local_path.exists() or local_path.with_suffix(ext).exists(): return local_path
    else: return Config.model_path()/filename

def datapath4file(filename, ext:str='.tgz', archive=True):
    "Return data path to `filename`, checking locally first then in the config file."
    local_path = URLs.LOCAL_PATH/'data'/filename
    if local_path.exists() or local_path.with_suffix(ext).exists(): return local_path
    elif archive: return Config.data_archive_path() / filename
    else: return Config.data_path() / filename

def download_data(url:str, fname:PathOrStr=None, data:bool=True, ext:str='.tgz') -> Path:
    "Download `url` to destination `fname`."
    fname = Path(ifnone(fname, _url2tgz(url, data, ext=ext)))
    os.makedirs(fname.parent, exist_ok=True)
    if not fname.exists():
        print(f'Downloading {url}')
        download_url(f'{url}{ext}', fname)
    return fname

def _check_file(fname):
    size = os.path.getsize(fname)
    with open(fname, "rb") as f:
        hash_nb = hashlib.md5(f.read(2**20)).hexdigest()
    return size,hash_nb

def untar_data(url:str, fname:PathOrStr=None, dest:PathOrStr=None, data=True, force_download=False) -> Path:
    "Download `url` to `fname` if `dest` doesn't exist, and un-tgz to folder `dest`."
    dest = url2path(url, data) if dest is None else Path(dest)/url2name(url)
    fname = Path(ifnone(fname, _url2tgz(url, data)))
    if force_download or (fname.exists() and url in _checks and _check_file(fname) != _checks[url]):
        print(f"A new version of the {'dataset' if data else 'model'} is available.")
        if fname.exists(): os.remove(fname)
        if dest.exists(): shutil.rmtree(dest)
    if not dest.exists():
        fname = download_data(url, fname=fname, data=data)
        if url in _checks:
            assert _check_file(fname) == _checks[url], f"Downloaded file {fname} does not match checksum expected! Remove that file from {Config().data_archive_path()} and try your code again."
        tarfile.open(fname, 'r:gz').extractall(dest.parent)
    return dest
