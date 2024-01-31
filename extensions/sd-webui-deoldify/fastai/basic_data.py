"`fastai.data` loads and manages datasets with `DataBunch`"
from .torch_core import *
from torch.utils.data.dataloader import default_collate

DatasetType = Enum('DatasetType', 'Train Valid Test Single Fix')
__all__ = ['DataBunch', 'DeviceDataLoader', 'DatasetType', 'load_data']

old_dl_init = torch.utils.data.DataLoader.__init__

def intercept_args(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, collate_fn=default_collate, pin_memory=True, drop_last=False,
                 timeout=0, worker_init_fn=None):
    self.init_kwargs = {'batch_size':batch_size, 'shuffle':shuffle, 'sampler':sampler, 'batch_sampler':batch_sampler,
                        'num_workers':num_workers, 'collate_fn':collate_fn, 'pin_memory':pin_memory,
                        'drop_last': drop_last, 'timeout':timeout, 'worker_init_fn':worker_init_fn}
    old_dl_init(self, dataset, **self.init_kwargs)

torch.utils.data.DataLoader.__init__ = intercept_args

def DataLoader___getattr__(dl, k:str)->Any: return getattr(dl.dataset, k)
DataLoader.__getattr__ = DataLoader___getattr__

def DataLoader___setstate__(dl, data:Any): dl.__dict__.update(data)
DataLoader.__setstate__ = DataLoader___setstate__

@dataclass
class DeviceDataLoader():
    "Bind a `DataLoader` to a `torch.device`."
    dl: DataLoader
    device: torch.device
    tfms: List[Callable]=None
    collate_fn: Callable=data_collate
    def __post_init__(self):
        self.dl.collate_fn=self.collate_fn
        self.tfms = listify(self.tfms)

    def __len__(self)->int: return len(self.dl)
    def __getattr__(self,k:str)->Any: return getattr(self.dl, k)
    def __setstate__(self,data:Any): self.__dict__.update(data)

    @property
    def batch_size(self):   return self.dl.batch_size
    @batch_size.setter
    def batch_size(self,v):
        new_kwargs = {**self.dl.init_kwargs, 'batch_size':v, 'collate_fn':self.collate_fn}
        self.dl = self.dl.__class__(self.dl.dataset, **new_kwargs)
        if hasattr(self.dl.dataset, 'bs'): self.dl.dataset.bs = v

    @property
    def num_workers(self):   return self.dl.num_workers
    @num_workers.setter
    def num_workers(self,v): self.dl.num_workers = v

    def add_tfm(self,tfm:Callable)->None:
        "Add `tfm` to `self.tfms`."
        self.tfms.append(tfm)
    def remove_tfm(self,tfm:Callable)->None:
        "Remove `tfm` from `self.tfms`."
        if tfm in self.tfms: self.tfms.remove(tfm)

    def new(self, **kwargs):
        "Create a new copy of `self` with `kwargs` replacing current values."
        new_kwargs = {**self.dl.init_kwargs, **kwargs}
        return DeviceDataLoader(self.dl.__class__(self.dl.dataset, **new_kwargs), self.device, self.tfms,
                                self.collate_fn)

    def proc_batch(self,b:Tensor)->Tensor:
        "Process batch `b` of `TensorImage`."
        b = to_device(b, self.device)
        for f in listify(self.tfms): b = f(b)
        return b

    def __iter__(self):
        "Process and returns items from `DataLoader`."
        for b in self.dl: yield self.proc_batch(b)

    @classmethod
    def create(cls, dataset:Dataset, bs:int=64, shuffle:bool=False, device:torch.device=defaults.device,
               tfms:Collection[Callable]=tfms, num_workers:int=defaults.cpus, collate_fn:Callable=data_collate, **kwargs:Any):
        "Create DeviceDataLoader from `dataset` with `bs` and `shuffle`: process using `num_workers`."
        return cls(DataLoader(dataset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, **kwargs),
                   device=device, tfms=tfms, collate_fn=collate_fn)

class DataBunch():
    "Bind `train_dl`,`valid_dl` and `test_dl` in a data object."

    def __init__(self, train_dl:DataLoader, valid_dl:DataLoader, fix_dl:DataLoader=None, test_dl:Optional[DataLoader]=None,
                 device:torch.device=None, dl_tfms:Optional[Collection[Callable]]=None, path:PathOrStr='.',
                 collate_fn:Callable=data_collate, no_check:bool=False):
        self.dl_tfms = listify(dl_tfms)
        self.device = defaults.device if device is None else device
        assert not isinstance(train_dl,DeviceDataLoader)
        def _create_dl(dl, **kwargs):
            if dl is None: return None
            return DeviceDataLoader(dl, self.device, self.dl_tfms, collate_fn, **kwargs)
        self.train_dl,self.valid_dl,self.fix_dl,self.test_dl = map(_create_dl, [train_dl,valid_dl,fix_dl,test_dl])
        if fix_dl is None: self.fix_dl = self.train_dl.new(shuffle=False, drop_last=False)
        self.single_dl = _create_dl(DataLoader(valid_dl.dataset, batch_size=1, num_workers=0))
        self.path = Path(path)
        if not no_check: self.sanity_check()

    def __repr__(self)->str:
        return f'{self.__class__.__name__};\n\nTrain: {self.train_ds};\n\nValid: {self.valid_ds};\n\nTest: {self.test_ds}'

    @staticmethod
    def _init_ds(train_ds:Dataset, valid_ds:Dataset, test_ds:Optional[Dataset]=None):
        # train_ds, but without training tfms
        fix_ds = valid_ds.new(train_ds.x, train_ds.y) if hasattr(valid_ds,'new') else train_ds
        return [o for o in (train_ds,valid_ds,fix_ds,test_ds) if o is not None]

    @classmethod
    def create(cls, train_ds:Dataset, valid_ds:Dataset, test_ds:Optional[Dataset]=None, path:PathOrStr='.', bs:int=64,
               val_bs:int=None, num_workers:int=defaults.cpus, dl_tfms:Optional[Collection[Callable]]=None,
               device:torch.device=None, collate_fn:Callable=data_collate, no_check:bool=False, **dl_kwargs)->'DataBunch':
        "Create a `DataBunch` from `train_ds`, `valid_ds` and maybe `test_ds` with a batch size of `bs`. Passes `**dl_kwargs` to `DataLoader()`"
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)
        dls = [DataLoader(d, b, shuffle=s, drop_last=s, num_workers=num_workers, **dl_kwargs) for d,b,s in
               zip(datasets, (bs,val_bs,val_bs,val_bs), (True,False,False,False)) if d is not None]
        return cls(*dls, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)

    def __getattr__(self,k:int)->Any: return getattr(self.train_dl, k)
    def __setstate__(self,data:Any): self.__dict__.update(data)

    def dl(self, ds_type:DatasetType=DatasetType.Valid)->DeviceDataLoader:
        "Returns appropriate `Dataset` for validation, training, or test (`ds_type`)."
        #TODO: refactor
        return (self.train_dl if ds_type == DatasetType.Train else
                self.test_dl if ds_type == DatasetType.Test else
                self.valid_dl if ds_type == DatasetType.Valid else
                self.single_dl if ds_type == DatasetType.Single else
                self.fix_dl)

    @property
    def dls(self)->List[DeviceDataLoader]:
        "Returns a list of all DeviceDataLoaders. If you need a specific DeviceDataLoader, access via the relevant property (`train_dl`, `valid_dl`, etc) as the index of DLs in this list is not guaranteed to remain constant."
        res = [self.train_dl, self.fix_dl, self.single_dl]
        # Preserve the original ordering of Train, Valid, Fix, Single, Test Data Loaders
        # (Unknown/not verified as of 1.0.47 whether there are other methods explicitly using DLs their list index)
        if self.valid_dl: res.insert(1, self.valid_dl)
        return res if not self.test_dl else res + [self.test_dl]

    def add_tfm(self,tfm:Callable)->None:
        for dl in self.dls: dl.add_tfm(tfm)

    def remove_tfm(self,tfm:Callable)->None:
        for dl in self.dls: dl.remove_tfm(tfm)

    def save(self, file:PathLikeOrBinaryStream= 'data_save.pkl')->None:
        "Save the `DataBunch` in `self.path/file`. `file` can be file-like (file or buffer)"
        if not getattr(self, 'label_list', False):
            warn("Serializing the `DataBunch` only works when you created it using the data block API.")
            return
        try_save(self.label_list, self.path, file)

    def add_test(self, items:Iterator, label:Any=None, tfms=None, tfm_y=None)->None:
        "Add the `items` as a test set. Pass along `label` otherwise label them with `EmptyLabel`."
        self.label_list.add_test(items, label=label, tfms=tfms, tfm_y=tfm_y)
        vdl = self.valid_dl
        dl = DataLoader(self.label_list.test, vdl.batch_size, shuffle=False, drop_last=False, num_workers=vdl.num_workers)
        self.test_dl = DeviceDataLoader(dl, vdl.device, vdl.tfms, vdl.collate_fn)

    def one_batch(self, ds_type:DatasetType=DatasetType.Train, detach:bool=True, denorm:bool=True, cpu:bool=True)->Collection[Tensor]:
        "Get one batch from the data loader of `ds_type`. Optionally `detach` and `denorm`."
        dl = self.dl(ds_type)
        w = self.num_workers
        self.num_workers = 0
        try:     x,y = next(iter(dl))
        finally: self.num_workers = w
        if detach: x,y = to_detach(x,cpu=cpu),to_detach(y,cpu=cpu)
        norm = getattr(self,'norm',False)
        if denorm and norm:
            x = self.denorm(x)
            if norm.keywords.get('do_y',False): y = self.denorm(y, do_x=True)
        return x,y

    def one_item(self, item, detach:bool=False, denorm:bool=False, cpu:bool=False):
        "Get `item` into a batch. Optionally `detach` and `denorm`."
        ds = self.single_ds
        with ds.set_item(item):
            return self.one_batch(ds_type=DatasetType.Single, detach=detach, denorm=denorm, cpu=cpu)

    def show_batch(self, rows:int=5, ds_type:DatasetType=DatasetType.Train, reverse:bool=False, **kwargs)->None:
        "Show a batch of data in `ds_type` on a few `rows`."
        x,y = self.one_batch(ds_type, True, True)
        if reverse: x,y = x.flip(0),y.flip(0)
        n_items = rows **2 if self.train_ds.x._square_show else rows
        if self.dl(ds_type).batch_size < n_items: n_items = self.dl(ds_type).batch_size
        xs = [self.train_ds.x.reconstruct(grab_idx(x, i)) for i in range(n_items)]
        #TODO: get rid of has_arg if possible
        if has_arg(self.train_ds.y.reconstruct, 'x'):
            ys = [self.train_ds.y.reconstruct(grab_idx(y, i), x=x) for i,x in enumerate(xs)]
        else : ys = [self.train_ds.y.reconstruct(grab_idx(y, i)) for i in range(n_items)]
        self.train_ds.x.show_xys(xs, ys, **kwargs)
 
    def export(self, file:PathLikeOrBinaryStream='export.pkl'):
        "Export the minimal state of `self` for inference in `self.path/file`. `file` can be file-like (file or buffer)"
        xtra = dict(normalize=self.norm.keywords) if getattr(self, 'norm', False) else {}
        try_save(self.valid_ds.get_state(**xtra), self.path, file)

    def _grab_dataset(self, dl:DataLoader):
        ds = dl.dl.dataset
        while hasattr(ds, 'dataset'): ds = ds.dataset
        return ds

    @property
    def train_ds(self)->Dataset: return self._grab_dataset(self.train_dl)
    @property
    def valid_ds(self)->Dataset: return self._grab_dataset(self.valid_dl)
    @property
    def single_ds(self)->Dataset: return self._grab_dataset(self.single_dl)
    @property
    def loss_func(self)->OptLossFunc:
        return getattr(self.train_ds.y, 'loss_func', F.nll_loss) if hasattr(self.train_ds, 'y') else F.nll_loss

    @property
    def test_ds(self)->Dataset:
        return self._grab_dataset(self.test_dl) if self.test_dl is not None else None

    @property
    def empty_val(self)->bool:
        if not hasattr(self, 'valid_dl') or self.valid_dl is None:            return True
        if hasattr(self.valid_ds, 'items') and len(self.valid_ds.items) == 0: return True
        return (len(self.valid_ds) == 0)

    @property
    def is_empty(self)->bool:
        return not ((self.train_dl and len(self.train_ds.items) != 0) or 
                    (self.valid_dl and len(self.valid_ds.items) != 0) or 
                    (self.test_dl  and len(self.test_ds.items)  != 0))
    
    @property
    def batch_size(self):   return self.train_dl.batch_size
    @batch_size.setter
    def batch_size(self,v):
        self.train_dl.batch_size,self.valid_dl.batch_size = v,v
        if self.test_dl is not None: self.test_dl.batch_size = v

    def sanity_check(self):
        "Check the underlying data in the training set can be properly loaded."
        final_message = "You can deactivate this warning by passing `no_check=True`."
        if not hasattr(self.train_ds, 'items') or len(self.train_ds.items) == 0 or not hasattr(self.train_dl, 'batch_sampler'): return
        if len(self.train_dl) == 0:
            warn(f"""Your training dataloader is empty, you have only {len(self.train_dl.dataset)} items in your training set.
                 Your batch size is {self.train_dl.batch_size}, you should lower it.""")
            print(final_message)
            return
        idx = next(iter(self.train_dl.batch_sampler))
        samples,fails = [],[]
        for i in idx:
            try:    samples.append(self.train_dl.dataset[i])
            except: fails.append(i)
        if len(fails) > 0:
            warn_msg = "There seems to be something wrong with your dataset, for example, in the first batch can't access"
            if len(fails) == len(idx):
                warn_msg += f" any element of self.train_ds.\nTried: {show_some(idx)}"
            else:
                warn_msg += f" these elements in self.train_ds: {show_some(fails)}"
            warn(warn_msg)
            print(final_message)
            return
        try: batch = self.collate_fn(samples)
        except:
            message = "It's not possible to collate samples of your dataset together in a batch."
            try:
                shapes = [[o[i].data.shape for o in samples] for i in range(2)]
                message += f'\nShapes of the inputs/targets:\n{shapes}'
            except: pass
            warn(message)
            print(final_message)

def load_data(path:PathOrStr, file:PathLikeOrBinaryStream='data_save.pkl', bs:int=64, val_bs:int=None, num_workers:int=defaults.cpus,
              dl_tfms:Optional[Collection[Callable]]=None, device:torch.device=None, collate_fn:Callable=data_collate,
              no_check:bool=False, **kwargs)->DataBunch:
    "Load a saved `DataBunch` from `path/file`. `file` can be file-like (file or buffer)"
    source = Path(path)/file if is_pathlike(file) else file
    ll = torch.load(source, map_location='cpu') if defaults.device == torch.device('cpu') else torch.load(source)
    return ll.databunch(path=path, bs=bs, val_bs=val_bs, num_workers=num_workers, dl_tfms=dl_tfms, device=device,
                        collate_fn=collate_fn, no_check=no_check, **kwargs)
