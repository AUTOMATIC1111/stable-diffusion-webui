"Manages data input pipeline - folderstransformbatch input. Includes support for classification, segmentation and bounding boxes"
from numbers import Integral
from ..torch_core import *
from .image import *
from .transform import *
from ..data_block import *
from ..basic_data import *
from ..layers import *
from .learner import *
from torchvision import transforms as tvt

__all__ = ['get_image_files', 'denormalize', 'get_annotations', 'ImageDataBunch',
           'ImageList', 'normalize', 'normalize_funcs', 'resize_to',
           'channel_view', 'mnist_stats', 'cifar_stats', 'imagenet_stats', 'imagenet_stats_inception', 'download_images',
           'verify_images', 'bb_pad_collate', 'ImageImageList', 'PointsLabelList',
           'ObjectCategoryList', 'ObjectItemList', 'SegmentationLabelList', 'SegmentationItemList', 'PointsItemList']

image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))

def get_image_files(c:PathOrStr, check_ext:bool=True, recurse=False)->FilePathList:
    "Return list of files in `c` that are images. `check_ext` will filter to `image_extensions`."
    return get_files(c, extensions=(image_extensions if check_ext else None), recurse=recurse)

def get_annotations(fname, prefix=None):
    "Open a COCO style json in `fname` and returns the lists of filenames (with maybe `prefix`) and labelled bboxes."
    annot_dict = json.load(open(fname))
    id2images, id2bboxes, id2cats = {}, collections.defaultdict(list), collections.defaultdict(list)
    classes = {}
    for o in annot_dict['categories']:
        classes[o['id']] = o['name']
    for o in annot_dict['annotations']:
        bb = o['bbox']
        id2bboxes[o['image_id']].append([bb[1],bb[0], bb[3]+bb[1], bb[2]+bb[0]])
        id2cats[o['image_id']].append(classes[o['category_id']])
    for o in annot_dict['images']:
        if o['id'] in id2bboxes:
            id2images[o['id']] = ifnone(prefix, '') + o['file_name']
    ids = list(id2images.keys())
    return [id2images[k] for k in ids], [[id2bboxes[k], id2cats[k]] for k in ids]

def bb_pad_collate(samples:BatchSamples, pad_idx:int=0) -> Tuple[FloatTensor, Tuple[LongTensor, LongTensor]]:
    "Function that collect `samples` of labelled bboxes and adds padding with `pad_idx`."
    if isinstance(samples[0][1], int): return data_collate(samples)
    max_len = max([len(s[1].data[1]) for s in samples])
    bboxes = torch.zeros(len(samples), max_len, 4)
    labels = torch.zeros(len(samples), max_len).long() + pad_idx
    imgs = []
    for i,s in enumerate(samples):
        imgs.append(s[0].data[None])
        bbs, lbls = s[1].data
        if not (bbs.nelement() == 0):
            bboxes[i,-len(lbls):] = bbs
            labels[i,-len(lbls):] = tensor(lbls)
    return torch.cat(imgs,0), (bboxes,labels)

def normalize(x:TensorImage, mean,std:Tensor)->TensorImage:
    "Normalize `x` with `mean` and `std`."
    return (x-mean[...,None,None]) / std[...,None,None]

def denormalize(x:TensorImage, mean,std:Tensor, do_x:bool=True)->TensorImage:
    "Denormalize `x` with `mean` and `std`."
    return x.cpu().float()*std[...,None,None] + mean[...,None,None] if do_x else x.cpu()

def _normalize_batch(b:Tuple[Tensor,Tensor], mean:Tensor, std:Tensor, do_x:bool=True, do_y:bool=False)->Tuple[Tensor,Tensor]:
    "`b` = `x`,`y` - normalize `x` array of imgs and `do_y` optionally `y`."
    x,y = b
    mean,std = mean.to(x.device),std.to(x.device)
    if do_x: x = normalize(x,mean,std)
    if do_y and len(y.shape) == 4: y = normalize(y,mean,std)
    return x,y

def normalize_funcs(mean:Tensor, std:Tensor, do_x:bool=True, do_y:bool=False)->Tuple[Callable,Callable]:
    "Create normalize/denormalize func using `mean` and `std`, can specify `do_y` and `device`."
    mean,std = tensor(mean),tensor(std)
    return (partial(_normalize_batch, mean=mean, std=std, do_x=do_x, do_y=do_y),
            partial(denormalize,      mean=mean, std=std, do_x=do_x))

cifar_stats = ([0.491, 0.482, 0.447], [0.247, 0.243, 0.261])
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
imagenet_stats_inception = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
mnist_stats = ([0.15]*3, [0.15]*3)

def channel_view(x:Tensor)->Tensor:
    "Make channel the first axis of `x` and flatten remaining axes"
    return x.transpose(0,1).contiguous().view(x.shape[1],-1)

class ImageDataBunch(DataBunch):
    "DataBunch suitable for computer vision."
    _square_show = True

    @classmethod
    def create_from_ll(cls, lls:LabelLists, bs:int=64, val_bs:int=None, ds_tfms:Optional[TfmList]=None,
                num_workers:int=defaults.cpus, dl_tfms:Optional[Collection[Callable]]=None, device:torch.device=None,
                test:Optional[PathOrStr]=None, collate_fn:Callable=data_collate, size:int=None, no_check:bool=False,
                resize_method:ResizeMethod=None, mult:int=None, padding_mode:str='reflection',
                mode:str='bilinear', tfm_y:bool=False)->'ImageDataBunch':
        "Create an `ImageDataBunch` from `LabelLists` `lls` with potential `ds_tfms`."
        lls = lls.transform(tfms=ds_tfms, size=size, resize_method=resize_method, mult=mult, padding_mode=padding_mode,
                            mode=mode, tfm_y=tfm_y)
        if test is not None: lls.add_test_folder(test)
        return lls.databunch(bs=bs, val_bs=val_bs, dl_tfms=dl_tfms, num_workers=num_workers, collate_fn=collate_fn,
                             device=device, no_check=no_check)

    @classmethod
    def from_folder(cls, path:PathOrStr, train:PathOrStr='train', valid:PathOrStr='valid',
                    valid_pct=None, seed:int=None, classes:Collection=None, **kwargs:Any)->'ImageDataBunch':
        "Create from imagenet style dataset in `path` with `train`,`valid`,`test` subfolders (or provide `valid_pct`)."
        path=Path(path)
        il = ImageList.from_folder(path)
        if valid_pct is None: src = il.split_by_folder(train=train, valid=valid)
        else: src = il.split_by_rand_pct(valid_pct, seed)
        src = src.label_from_folder(classes=classes)
        return cls.create_from_ll(src, **kwargs)

    @classmethod
    def from_df(cls, path:PathOrStr, df:pd.DataFrame, folder:PathOrStr=None, label_delim:str=None, valid_pct:float=0.2,
                seed:int=None, fn_col:IntsOrStrs=0, label_col:IntsOrStrs=1, suffix:str='', **kwargs:Any)->'ImageDataBunch':
        "Create from a `DataFrame` `df`."
        src = (ImageList.from_df(df, path=path, folder=folder, suffix=suffix, cols=fn_col)
                .split_by_rand_pct(valid_pct, seed)
                .label_from_df(label_delim=label_delim, cols=label_col))
        return cls.create_from_ll(src, **kwargs)

    @classmethod
    def from_csv(cls, path:PathOrStr, folder:PathOrStr=None, label_delim:str=None, csv_labels:PathOrStr='labels.csv',
                 valid_pct:float=0.2, seed:int=None, fn_col:int=0, label_col:int=1, suffix:str='', delimiter:str=None,
                 header:Optional[Union[int,str]]='infer', **kwargs:Any)->'ImageDataBunch':
        "Create from a csv file in `path/csv_labels`."
        path = Path(path)
        df = pd.read_csv(path/csv_labels, header=header, delimiter=delimiter)
        return cls.from_df(path, df, folder=folder, label_delim=label_delim, valid_pct=valid_pct, seed=seed,
                fn_col=fn_col, label_col=label_col, suffix=suffix, **kwargs)

    @classmethod
    def from_lists(cls, path:PathOrStr, fnames:FilePathList, labels:Collection[str], valid_pct:float=0.2, seed:int=None,
                   item_cls:Callable=None, **kwargs):
        "Create from list of `fnames` in `path`."
        item_cls = ifnone(item_cls, ImageList)
        fname2label = {f:l for (f,l) in zip(fnames, labels)}
        src = (item_cls(fnames, path=path).split_by_rand_pct(valid_pct, seed)
                                .label_from_func(lambda x:fname2label[x]))
        return cls.create_from_ll(src, **kwargs)

    @classmethod
    def from_name_func(cls, path:PathOrStr, fnames:FilePathList, label_func:Callable, valid_pct:float=0.2, seed:int=None,
                       **kwargs):
        "Create from list of `fnames` in `path` with `label_func`."
        src = ImageList(fnames, path=path).split_by_rand_pct(valid_pct, seed)
        return cls.create_from_ll(src.label_from_func(label_func), **kwargs)

    @classmethod
    def from_name_re(cls, path:PathOrStr, fnames:FilePathList, pat:str, valid_pct:float=0.2, **kwargs):
        "Create from list of `fnames` in `path` with re expression `pat`."
        pat = re.compile(pat)
        def _get_label(fn):
            if isinstance(fn, Path): fn = fn.as_posix()
            res = pat.search(str(fn))
            assert res,f'Failed to find "{pat}" in "{fn}"'
            return res.group(1)
        return cls.from_name_func(path, fnames, _get_label, valid_pct=valid_pct, **kwargs)

    @staticmethod
    def single_from_classes(path:Union[Path, str], classes:Collection[str], ds_tfms:TfmList=None, **kwargs):
        "Create an empty `ImageDataBunch` in `path` with `classes`. Typically used for inference."
        warn("""This method is deprecated and will be removed in a future version, use `load_learner` after
             `Learner.export()`""", DeprecationWarning)
        sd = ImageList([], path=path, ignore_empty=True).split_none()
        return sd.label_const(0, label_cls=CategoryList, classes=classes).transform(ds_tfms, **kwargs).databunch()

    def batch_stats(self, funcs:Collection[Callable]=None, ds_type:DatasetType=DatasetType.Train)->Tensor:
        "Grab a batch of data and call reduction function `func` per channel"
        funcs = ifnone(funcs, [torch.mean,torch.std])
        x = self.one_batch(ds_type=ds_type, denorm=False)[0].cpu()
        return [func(channel_view(x), 1) for func in funcs]

    def normalize(self, stats:Collection[Tensor]=None, do_x:bool=True, do_y:bool=False)->None:
        "Add normalize transform using `stats` (defaults to `DataBunch.batch_stats`)"
        if getattr(self,'norm',False): raise Exception('Can not call normalize twice')
        if stats is None: self.stats = self.batch_stats()
        else:             self.stats = stats
        self.norm,self.denorm = normalize_funcs(*self.stats, do_x=do_x, do_y=do_y)
        self.add_tfm(self.norm)
        return self

def download_image(url,dest, timeout=4):
    try: r = download_url(url, dest, overwrite=True, show_progress=False, timeout=timeout)
    except Exception as e: print(f"Error {url} {e}")

def _download_image_inner(dest, url, i, timeout=4):
    suffix = re.findall(r'\.\w+?(?=(?:\?|$))', url)
    suffix = suffix[0] if len(suffix)>0  else '.jpg'
    download_image(url, dest/f"{i:08d}{suffix}", timeout=timeout)

def download_images(urls:Collection[str], dest:PathOrStr, max_pics:int=1000, max_workers:int=8, timeout=4):
    "Download images listed in text file `urls` to path `dest`, at most `max_pics`"
    urls = open(urls).read().strip().split("\n")[:max_pics]
    dest = Path(dest)
    dest.mkdir(exist_ok=True)
    parallel(partial(_download_image_inner, dest, timeout=timeout), urls, max_workers=max_workers)

def resize_to(img, targ_sz:int, use_min:bool=False):
    "Size to resize to, to hit `targ_sz` at same aspect ratio, in PIL coords (i.e w*h)"
    w,h = img.size
    min_sz = (min if use_min else max)(w,h)
    ratio = targ_sz/min_sz
    return int(w*ratio),int(h*ratio)

def verify_image(file:Path, idx:int, delete:bool, max_size:Union[int,Tuple[int,int]]=None, dest:Path=None, n_channels:int=3,
                 interp=PIL.Image.BILINEAR, ext:str=None, img_format:str=None, resume:bool=False, **kwargs):
    "Check if the image in `file` exists, maybe resize it and copy it in `dest`."
    try:
        # deal with partially broken images as indicated by PIL warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                with open(file, 'rb') as img_file: PIL.Image.open(img_file)
            except Warning as w:
                if "Possibly corrupt EXIF data" in str(w):
                    if delete: # green light to modify files
                        print(f"{file}: Removing corrupt EXIF data")
                        warnings.simplefilter("ignore")
                        # save EXIF-cleaned up image, which happens automatically
                        PIL.Image.open(file).save(file)
                    else: # keep user's files intact
                        print(f"{file}: Not removing corrupt EXIF data, pass `delete=True` to do that")
                else: warnings.warn(w)

        img = PIL.Image.open(file)
        imgarr = np.array(img)
        img_channels = 1 if len(imgarr.shape) == 2 else imgarr.shape[2]
        if (max_size is not None and (img.height > max_size or img.width > max_size)) or img_channels != n_channels:
            assert isinstance(dest, Path), "You should provide `dest` Path to save resized image"
            dest_fname = dest/file.name
            if ext is not None: dest_fname=dest_fname.with_suffix(ext)
            if resume and os.path.isfile(dest_fname): return
            if max_size is not None:
                new_sz = resize_to(img, max_size)
                img = img.resize(new_sz, resample=interp)
            if n_channels == 3: img = img.convert("RGB")
            img.save(dest_fname, img_format, **kwargs)
    except Exception as e:
        print(f'{e}')
        if delete: file.unlink()

def verify_images(path:PathOrStr, delete:bool=True, max_workers:int=4, max_size:Union[int]=None, recurse:bool=False,
                  dest:PathOrStr='.', n_channels:int=3, interp=PIL.Image.BILINEAR, ext:str=None, img_format:str=None,
                  resume:bool=None, **kwargs):
    "Check if the images in `path` aren't broken, maybe resize them and copy it in `dest`."
    path = Path(path)
    if resume is None and dest == '.': resume=False
    dest = path/Path(dest)
    os.makedirs(dest, exist_ok=True)
    files = get_image_files(path, recurse=recurse)
    func = partial(verify_image, delete=delete, max_size=max_size, dest=dest, n_channels=n_channels, interp=interp,
                   ext=ext, img_format=img_format, resume=resume, **kwargs)
    parallel(func, files, max_workers=max_workers)

class ImageList(ItemList):
    "`ItemList` suitable for computer vision."
    _bunch,_square_show,_square_show_res = ImageDataBunch,True,True
    def __init__(self, *args, convert_mode='RGB', after_open:Callable=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.convert_mode,self.after_open = convert_mode,after_open
        self.copy_new += ['convert_mode', 'after_open']
        self.c,self.sizes = 3,{}

    def open(self, fn):
        "Open image in `fn`, subclass and overwrite for custom behavior."
        return open_image(fn, convert_mode=self.convert_mode, after_open=self.after_open)

    def get(self, i):
        fn = super().get(i)
        res = self.open(fn)
        self.sizes[i] = res.size
        return res
    
    @classmethod
    def from_folder(cls, path:PathOrStr='.', extensions:Collection[str]=None, **kwargs)->ItemList:
        "Get the list of files in `path` that have an image suffix. `recurse` determines if we search subfolders."
        extensions = ifnone(extensions, image_extensions)
        return super().from_folder(path=path, extensions=extensions, **kwargs)

    @classmethod
    def from_df(cls, df:DataFrame, path:PathOrStr, cols:IntsOrStrs=0, folder:PathOrStr=None, suffix:str='', **kwargs)->'ItemList':
        "Get the filenames in `cols` of `df` with `folder` in front of them, `suffix` at the end."
        suffix = suffix or ''
        res = super().from_df(df, path=path, cols=cols, **kwargs)
        pref = f'{res.path}{os.path.sep}'
        if folder is not None: pref += f'{folder}{os.path.sep}'
        res.items = np.char.add(np.char.add(pref, res.items.astype(str)), suffix)
        return res

    @classmethod
    def from_csv(cls, path:PathOrStr, csv_name:str, header:str='infer', delimiter:str=None, **kwargs)->'ItemList':
        "Get the filenames in `path/csv_name` opened with `header`."
        path = Path(path)
        df = pd.read_csv(path/csv_name, header=header, delimiter=delimiter)
        return cls.from_df(df, path=path, **kwargs)

    def reconstruct(self, t:Tensor): return Image(t.float().clamp(min=0,max=1))

    def show_xys(self, xs, ys, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Show the `xs` (inputs) and `ys` (targets) on a figure of `figsize`."
        rows = int(np.ceil(math.sqrt(len(xs))))
        axs = subplots(rows, rows, imgsize=imgsize, figsize=figsize)
        for x,y,ax in zip(xs, ys, axs.flatten()): x.show(ax=ax, y=y, **kwargs)
        for ax in axs.flatten()[len(xs):]: ax.axis('off')
        plt.tight_layout()

    def show_xyzs(self, xs, ys, zs, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`."
        if self._square_show_res:
            title = 'Ground truth\nPredictions'
            rows = int(np.ceil(math.sqrt(len(xs))))
            axs = subplots(rows, rows, imgsize=imgsize, figsize=figsize, title=title, weight='bold', size=12)
            for x,y,z,ax in zip(xs,ys,zs,axs.flatten()): x.show(ax=ax, title=f'{str(y)}\n{str(z)}', **kwargs)
            for ax in axs.flatten()[len(xs):]: ax.axis('off')
        else:
            title = 'Ground truth/Predictions'
            axs = subplots(len(xs), 2, imgsize=imgsize, figsize=figsize, title=title, weight='bold', size=14)
            for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
                x.show(ax=axs[i,0], y=y, **kwargs)
                x.show(ax=axs[i,1], y=z, **kwargs)

class ObjectCategoryProcessor(MultiCategoryProcessor):
    "`PreProcessor` for labelled bounding boxes."
    def __init__(self, ds:ItemList, pad_idx:int=0):
        super().__init__(ds)
        self.pad_idx = pad_idx
        self.state_attrs.append('pad_idx')

    def process(self, ds:ItemList):
        ds.pad_idx = self.pad_idx
        super().process(ds)

    def process_one(self,item): return [item[0], [self.c2i.get(o,None) for o in item[1]]]

    def generate_classes(self, items):
        "Generate classes from unique `items` and add `background`."
        classes = super().generate_classes([o[1] for o in items])
        classes = ['background'] + list(classes)
        return classes

def _get_size(xs,i):
    size = xs.sizes.get(i,None)
    if size is None:
        # Image hasn't been accessed yet, so we don't know its size
        _ = xs[i]
        size = xs.sizes[i]
    return size

class ObjectCategoryList(MultiCategoryList):
    "`ItemList` for labelled bounding boxes."
    _processor = ObjectCategoryProcessor

    def get(self, i):
        return ImageBBox.create(*_get_size(self.x,i), *self.items[i], classes=self.classes, pad_idx=self.pad_idx)

    def analyze_pred(self, pred): return pred

    def reconstruct(self, t, x):
        (bboxes, labels) = t
        if len((labels - self.pad_idx).nonzero()) == 0: return
        i = (labels - self.pad_idx).nonzero().min()
        bboxes,labels = bboxes[i:],labels[i:]
        return ImageBBox.create(*x.size, bboxes, labels=labels, classes=self.classes, scale=False)

class ObjectItemList(ImageList):
    "`ItemList` suitable for object detection."
    _label_cls,_square_show_res = ObjectCategoryList,False

class SegmentationProcessor(PreProcessor):
    "`PreProcessor` that stores the classes for segmentation."
    def __init__(self, ds:ItemList): self.classes = ds.classes
    def process(self, ds:ItemList):  ds.classes,ds.c = self.classes,len(self.classes)

class SegmentationLabelList(ImageList):
    "`ItemList` for segmentation masks."
    _processor=SegmentationProcessor
    def __init__(self, items:Iterator, classes:Collection=None, **kwargs):
        super().__init__(items, **kwargs)
        self.copy_new.append('classes')
        self.classes,self.loss_func = classes,CrossEntropyFlat(axis=1)

    def open(self, fn): return open_mask(fn)
    def analyze_pred(self, pred, thresh:float=0.5): return pred.argmax(dim=0)[None]
    def reconstruct(self, t:Tensor): return ImageSegment(t)

class SegmentationItemList(ImageList):
    "`ItemList` suitable for segmentation tasks."
    _label_cls,_square_show_res = SegmentationLabelList,False

class PointsProcessor(PreProcessor):
    "`PreProcessor` that stores the number of targets for point regression."
    def __init__(self, ds:ItemList): self.c = len(ds.items[0].reshape(-1))
    def process(self, ds:ItemList):  ds.c = self.c

class PointsLabelList(ItemList):
    "`ItemList` for points."
    _processor = PointsProcessor
    def __init__(self, items:Iterator, **kwargs):
        super().__init__(items, **kwargs)
        self.loss_func = MSELossFlat()

    def get(self, i):
        o = super().get(i)
        return ImagePoints(FlowField(_get_size(self.x,i), o), scale=True)

    def analyze_pred(self, pred, thresh:float=0.5): return pred.view(-1,2)
    def reconstruct(self, t, x): return ImagePoints(FlowField(x.size, t), scale=False)

class PointsItemList(ImageList):
    "`ItemList` for `Image` to `ImagePoints` tasks."
    _label_cls,_square_show_res = PointsLabelList,False

class ImageImageList(ImageList):
    "`ItemList` suitable for `Image` to `Image` tasks."
    _label_cls,_square_show,_square_show_res = ImageList,False,False

    def show_xys(self, xs, ys, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Show the `xs` (inputs) and `ys`(targets)  on a figure of `figsize`."
        axs = subplots(len(xs), 2, imgsize=imgsize, figsize=figsize)
        for i, (x,y) in enumerate(zip(xs,ys)):
            x.show(ax=axs[i,0], **kwargs)
            y.show(ax=axs[i,1], **kwargs)
        plt.tight_layout()

    def show_xyzs(self, xs, ys, zs, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`."
        title = 'Input / Prediction / Target'
        axs = subplots(len(xs), 3, imgsize=imgsize, figsize=figsize, title=title, weight='bold', size=14)
        for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
            x.show(ax=axs[i,0], **kwargs)
            y.show(ax=axs[i,2], **kwargs)
            z.show(ax=axs[i,1], **kwargs)


def _ll_pre_transform(self, train_tfm:List[Callable], valid_tfm:List[Callable]):
    "Call `train_tfm` and `valid_tfm` after opening image, before converting from `PIL.Image`"
    self.train.x.after_open = compose(train_tfm)
    self.valid.x.after_open = compose(valid_tfm)
    return self

def _db_pre_transform(self, train_tfm:List[Callable], valid_tfm:List[Callable]):
    "Call `train_tfm` and `valid_tfm` after opening image, before converting from `PIL.Image`"
    self.train_ds.x.after_open = compose(train_tfm)
    self.valid_ds.x.after_open = compose(valid_tfm)
    return self

def _presize(self, size:int, val_xtra_size:int=32, scale:Tuple[float]=(0.08, 1.0), ratio:Tuple[float]=(0.75, 4./3.),
             interpolation:int=2):
    "Resize images to `size` using `RandomResizedCrop`, passing along `kwargs` to train transform"
    return self.pre_transform(
        tvt.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=interpolation), 
        [tvt.Resize(size+val_xtra_size), tvt.CenterCrop(size)])

LabelLists.pre_transform = _ll_pre_transform
DataBunch.pre_transform = _db_pre_transform
LabelLists.presize = _presize
DataBunch.presize = _presize

