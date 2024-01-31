"NLP data loading pipeline. Supports csv, folders, and preprocessed data."
from ..torch_core import *
from .transform import *
from ..basic_data import *
from ..data_block import *
from ..layers import *
from ..callback import Callback

__all__ = ['LanguageModelPreLoader', 'SortSampler', 'SortishSampler', 'TextList', 'pad_collate', 'TextDataBunch',
           'TextLMDataBunch', 'TextClasDataBunch', 'Text', 'open_text', 'TokenizeProcessor', 'NumericalizeProcessor',
           'OpenFileProcessor', 'LMLabelList', 'LMTextList', 'SPProcessor']

TextMtd = IntEnum('TextMtd', 'DF TOK IDS')
text_extensions = {'.txt'}

class LanguageModelPreLoader(Callback):
    "Transforms the tokens in `dataset` to a stream of contiguous batches for language modelling."

    class CircularIndex():
        "Handles shuffle, direction of indexing, wraps around to head tail in the ragged array as needed"
        def __init__(self, length:int, forward:bool): self.idx, self.forward = np.arange(length), forward
        def __getitem__(self, i):
            return self.idx[ i%len(self.idx) if self.forward else len(self.idx)-1-i%len(self.idx)]
        def __len__(self) -> int: return len(self.idx)
        def shuffle(self): np.random.shuffle(self.idx)

    def __init__(self, dataset:LabelList, lengths:Collection[int]=None, bs:int=32, bptt:int=70, backwards:bool=False,
                 shuffle:bool=False):
        self.dataset,self.bs,self.bptt,self.shuffle,self.backwards,self.lengths = dataset,bs,bptt,shuffle,backwards,lengths
        self.bs *= num_distrib() or 1
        self.totalToks,self.ite_len,self.idx = int(0),None,None

    def __len__(self):
        if self.ite_len is None:
            if self.lengths is None: self.lengths = np.array([len(item) for item in self.dataset.x.items])
            self.totalToks = self.lengths.sum()
            self.ite_len   = self.bs*int( math.ceil( self.totalToks/(self.bptt*self.bs) )) if self.item is None else 1
        return self.ite_len

    def __getattr__(self,k:str)->Any: return getattr(self.dataset, k)

    def allocate_buffers(self):
        "Create the ragged array that will be filled when we ask for items."
        if self.ite_len is None: len(self)
        self.idx   = LanguageModelPreLoader.CircularIndex(len(self.dataset.x.items), not self.backwards)
        self.batch = np.zeros((self.bs, self.bptt+1), dtype=np.int64)
        self.batch_x, self.batch_y = self.batch[:,0:self.bptt], self.batch[:,1:self.bptt+1]
        #ro: index of the text we're at inside our datasets for the various batches
        self.ro    = np.zeros(self.bs, dtype=np.int64)
        #ri: index of the token we're at inside our current text for the various batches
        self.ri    = np.zeros(self.bs, dtype=np.int)

    def on_epoch_begin(self, **kwargs):
        if self.idx is None or len(self.idx) != len(self.dataset.x.items): self.allocate_buffers()
        elif self.shuffle:   self.idx.shuffle()
        self.idx.forward = not self.backwards

        step = self.totalToks / self.bs
        ln_rag, countTokens, i_rag = 0, 0, -1
        for i in range(0,self.bs):
            #Compute the initial values for ro and ri
            while ln_rag + countTokens <= int(step * i):
                countTokens += ln_rag
                i_rag       += 1
                ln_rag       = self.lengths[self.idx[i_rag]]
            self.ro[i] = i_rag
            self.ri[i] = ( ln_rag - int(step * i - countTokens) ) if self.backwards else int(step * i - countTokens)

    #Training dl gets on_epoch_begin called, val_dl, on_epoch_end
    def on_epoch_end(self, **kwargs): self.on_epoch_begin()

    def __getitem__(self, k:int):
        j = k % self.bs
        if self.item is not None: return self.dataset[0]
        if self.idx is None: self.on_epoch_begin()
        self.ro[j],self.ri[j] = self.fill_row(not self.backwards, self.dataset.x.items, self.idx, self.batch[j],
                                              self.ro[j], self.ri[j], overlap=1, lengths=self.lengths)
        return self.batch_x[j], self.batch_y[j]

    def fill_row(self, forward, items, idx, row, ro, ri, overlap,lengths):
        "Fill the row with tokens from the ragged array. --OBS-- overlap != 1 has not been implemented"
        ibuf = n = 0
        ro  -= 1
        while ibuf < row.size:
            ro   += 1
            ix    = idx[ro]
            rag   = items[ix]
            if forward:
                ri = 0 if ibuf else ri
                n  = min(lengths[ix] - ri, row.size - ibuf)
                row[ibuf:ibuf+n] = rag[ri:ri+n]
            else:
                ri = lengths[ix] if ibuf else ri
                n  = min(ri, row.size - ibuf)
                row[ibuf:ibuf+n] = rag[ri-n:ri][::-1]
            ibuf += n
        return ro, ri + ((n-overlap) if forward else -(n-overlap))

class SortSampler(Sampler):
    "Go through the text data by order of length."

    def __init__(self, data_source:NPArrayList, key:KeyFunc): self.data_source,self.key = data_source,key
    def __len__(self) -> int: return len(self.data_source)
    def __iter__(self):
        return iter(sorted(range_of(self.data_source), key=self.key, reverse=True))

class SortishSampler(Sampler):
    "Go through the text data by order of length with a bit of randomness."

    def __init__(self, data_source:NPArrayList, key:KeyFunc, bs:int):
        self.data_source,self.key,self.bs = data_source,key,bs

    def __len__(self) -> int: return len(self.data_source)

    def __iter__(self):
        idxs = np.random.permutation(len(self.data_source))
        sz = self.bs*50
        ck_idx = [idxs[i:i+sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate([sorted(s, key=self.key, reverse=True) for s in ck_idx])
        sz = self.bs
        ck_idx = [sort_idx[i:i+sz] for i in range(0, len(sort_idx), sz)]
        max_ck = np.argmax([self.key(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
        ck_idx[0],ck_idx[max_ck] = ck_idx[max_ck],ck_idx[0]     # then make sure it goes first.
        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([],dtype=np.int)
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)

def pad_collate(samples:BatchSamples, pad_idx:int=1, pad_first:bool=True, backwards:bool=False) -> Tuple[LongTensor, LongTensor]:
    "Function that collect samples and adds padding. Flips token order if needed"
    samples = to_data(samples)
    max_len = max([len(s[0]) for s in samples])
    res = torch.zeros(len(samples), max_len).long() + pad_idx
    if backwards: pad_first = not pad_first
    for i,s in enumerate(samples):
        if pad_first: res[i,-len(s[0]):] = LongTensor(s[0])
        else:         res[i,:len(s[0]):] = LongTensor(s[0])
    if backwards: res = res.flip(1)
    return res, tensor(np.array([s[1] for s in samples]))

def _get_processor(tokenizer:Tokenizer=None, vocab:Vocab=None, chunksize:int=10000, max_vocab:int=60000,
                   min_freq:int=2, mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False):
    return [TokenizeProcessor(tokenizer=tokenizer, chunksize=chunksize, 
                              mark_fields=mark_fields, include_bos=include_bos, include_eos=include_eos),
            NumericalizeProcessor(vocab=vocab, max_vocab=max_vocab, min_freq=min_freq)]

class TextDataBunch(DataBunch):
    "General class to get a `DataBunch` for NLP. Subclassed by `TextLMDataBunch` and `TextClasDataBunch`."

    @classmethod
    def from_ids(cls, path:PathOrStr, vocab:Vocab, train_ids:Collection[Collection[int]], valid_ids:Collection[Collection[int]],
                 test_ids:Collection[Collection[int]]=None, train_lbls:Collection[Union[int,float]]=None,
                 valid_lbls:Collection[Union[int,float]]=None, classes:Collection[Any]=None,
                 processor:PreProcessor=None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from ids, labels and a `vocab`. `kwargs` are passed to the dataloader creation."
        src = ItemLists(path, TextList(train_ids, vocab, path=path, processor=[]),
                        TextList(valid_ids, vocab, path=path, processor=[]))
        src = src.label_for_lm() if cls==TextLMDataBunch else src.label_from_lists(train_lbls, valid_lbls, classes=classes, processor=[])
        if not is1d(train_lbls): src.train.y.one_hot,src.valid.y.one_hot = True,True
        if test_ids is not None: src.add_test(TextList(test_ids, vocab, path=path), label=train_lbls[0])
        src.valid.x.processor = ifnone(processor, [TokenizeProcessor(), NumericalizeProcessor(vocab=vocab)])
        if classes is not None: src.valid.y.processor = ifnone(processor, [CategoryProcessor(src.valid.y)])
        return src.databunch(**kwargs)

    @classmethod
    def load(cls, path:PathOrStr, cache_name:PathOrStr='tmp', processor:PreProcessor=None, **kwargs):
        "Load a `TextDataBunch` from `path/cache_name`. `kwargs` are passed to the dataloader creation."
        warn("""This method is deprecated and only kept to load data serialized in v1.0.43 or earlier.
                Use `load_data` for data saved with v1.0.44 or later.""", DeprecationWarning)
        cache_path = Path(path)/cache_name
        vocab = Vocab(pickle.load(open(cache_path/'itos.pkl','rb')))
        train_ids,train_lbls = np.load(cache_path/f'train_ids.npy'), np.load(cache_path/f'train_lbl.npy')
        valid_ids,valid_lbls = np.load(cache_path/f'valid_ids.npy'), np.load(cache_path/f'valid_lbl.npy')
        test_ids = np.load(cache_path/f'test_ids.npy') if os.path.isfile(cache_path/f'test_ids.npy') else None
        classes = loadtxt_str(cache_path/'classes.txt') if os.path.isfile(cache_path/'classes.txt') else None
        return cls.from_ids(path, vocab, train_ids, valid_ids, test_ids, train_lbls, valid_lbls, classes, processor, **kwargs)

    @classmethod#TODO: test
    def from_tokens(cls, path:PathOrStr, trn_tok:Collection[Collection[str]], trn_lbls:Collection[Union[int,float]],
                 val_tok:Collection[Collection[str]], val_lbls:Collection[Union[int,float]], vocab:Vocab=None,
                 tst_tok:Collection[Collection[str]]=None, classes:Collection[Any]=None, max_vocab:int=60000, min_freq:int=3,
                 **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from tokens and labels. `kwargs` are passed to the dataloader creation."
        processor = NumericalizeProcessor(vocab=vocab, max_vocab=max_vocab, min_freq=min_freq)
        src = ItemLists(path, TextList(trn_tok, path=path, processor=processor),
                        TextList(val_tok, path=path, processor=processor))
        src = src.label_for_lm() if cls==TextLMDataBunch else src.label_from_lists(trn_lbls, val_lbls, classes=classes)
        if tst_tok is not None: src.add_test(TextList(tst_tok, path=path))
        return src.databunch(**kwargs)

    @classmethod
    def from_df(cls, path:PathOrStr, train_df:DataFrame, valid_df:DataFrame, test_df:Optional[DataFrame]=None,
                tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, text_cols:IntsOrStrs=1,
                label_cols:IntsOrStrs=0, label_delim:str=None, chunksize:int=10000, max_vocab:int=60000,
                min_freq:int=2, mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from DataFrames. `kwargs` are passed to the dataloader creation."
        processor = _get_processor(tokenizer=tokenizer, vocab=vocab, chunksize=chunksize, max_vocab=max_vocab,
                                   min_freq=min_freq, mark_fields=mark_fields, 
                                   include_bos=include_bos, include_eos=include_eos)
        if classes is None and is_listy(label_cols) and len(label_cols) > 1: classes = label_cols
        src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),
                        TextList.from_df(valid_df, path, cols=text_cols, processor=processor))
        if cls==TextLMDataBunch: src = src.label_for_lm()
        else: 
            if label_delim is not None: src = src.label_from_df(cols=label_cols, classes=classes, label_delim=label_delim)
            else: src = src.label_from_df(cols=label_cols, classes=classes)
        if test_df is not None: src.add_test(TextList.from_df(test_df, path, cols=text_cols))
        return src.databunch(**kwargs)

    @classmethod
    def from_csv(cls, path:PathOrStr, csv_name, valid_pct:float=0.2, test:Optional[str]=None,
                 tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, delimiter:str=None, header='infer',
                 text_cols:IntsOrStrs=1, label_cols:IntsOrStrs=0, label_delim:str=None,
                 chunksize:int=10000, max_vocab:int=60000, min_freq:int=2, 
                 mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from texts in csv files. `kwargs` are passed to the dataloader creation."
        df = pd.read_csv(Path(path)/csv_name, header=header, delimiter=delimiter)
        df = df.iloc[np.random.permutation(len(df))]
        cut = int(valid_pct * len(df)) + 1
        train_df, valid_df = df[cut:], df[:cut]
        test_df = None if test is None else pd.read_csv(Path(path)/test, header=header, delimiter=delimiter)
        return cls.from_df(path, train_df, valid_df, test_df, tokenizer=tokenizer, vocab=vocab, classes=classes, text_cols=text_cols,
                           label_cols=label_cols, label_delim=label_delim, chunksize=chunksize, max_vocab=max_vocab,
                           min_freq=min_freq, mark_fields=mark_fields, 
                           include_bos=include_bos, include_eos=include_eos, **kwargs)

    @classmethod
    def from_folder(cls, path:PathOrStr, train:str='train', valid:str='valid', test:Optional[str]=None,
                    classes:Collection[Any]=None, tokenizer:Tokenizer=None, vocab:Vocab=None, chunksize:int=10000, max_vocab:int=60000,
                    min_freq:int=2, mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False, **kwargs):
        "Create a `TextDataBunch` from text files in folders."
        path = Path(path).absolute()
        processor = [OpenFileProcessor()] + _get_processor(tokenizer=tokenizer, vocab=vocab, chunksize=chunksize, max_vocab=max_vocab,
                                   min_freq=min_freq, mark_fields=mark_fields, include_bos=include_bos, include_eos=include_eos)
        src = (TextList.from_folder(path, processor=processor)
                       .split_by_folder(train=train, valid=valid))
        src = src.label_for_lm() if cls==TextLMDataBunch else src.label_from_folder(classes=classes)
        if test is not None: src.add_test_folder(path/test)
        return src.databunch(**kwargs)

class TextLMDataBunch(TextDataBunch):
    "Create a `TextDataBunch` suitable for training a language model."
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', no_check:bool=False, bs=64, val_bs:int=None,
               num_workers:int=0, device:torch.device=None, collate_fn:Callable=data_collate,
               dl_tfms:Optional[Collection[Callable]]=None, bptt:int=70, backwards:bool=False, **dl_kwargs) -> DataBunch:
        "Create a `TextDataBunch` in `path` from the `datasets` for language modelling. Passes `**dl_kwargs` on to `DataLoader()`"
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)
        datasets = [LanguageModelPreLoader(ds, shuffle=(i==0), bs=(bs if i==0 else val_bs), bptt=bptt, backwards=backwards)
                    for i,ds in enumerate(datasets)]
        val_bs = bs
        dls = [DataLoader(d, b, shuffle=False, **dl_kwargs) for d,b in zip(datasets, (bs,val_bs,val_bs,val_bs)) if d is not None]
        return cls(*dls, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)

class TextClasDataBunch(TextDataBunch):
    "Create a `TextDataBunch` suitable for training an RNN classifier."
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', bs:int=32, val_bs:int=None, pad_idx=1,
               pad_first=True, device:torch.device=None, no_check:bool=False, backwards:bool=False, 
               dl_tfms:Optional[Collection[Callable]]=None, **dl_kwargs) -> DataBunch:
        "Function that transform the `datasets` in a `DataBunch` for classification. Passes `**dl_kwargs` on to `DataLoader()`"
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)
        collate_fn = partial(pad_collate, pad_idx=pad_idx, pad_first=pad_first, backwards=backwards)
        train_sampler = SortishSampler(datasets[0].x, key=lambda t: len(datasets[0][t][0].data), bs=bs)
        train_dl = DataLoader(datasets[0], batch_size=bs, sampler=train_sampler, drop_last=True, **dl_kwargs)
        dataloaders = [train_dl]
        for ds in datasets[1:]:
            lengths = [len(t) for t in ds.x.items]
            sampler = SortSampler(ds.x, key=lengths.__getitem__)
            dataloaders.append(DataLoader(ds, batch_size=val_bs, sampler=sampler, **dl_kwargs))
        return cls(*dataloaders, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)

def open_text(fn:PathOrStr, enc='utf-8'):
    "Read the text in `fn`."
    with open(fn,'r', encoding = enc) as f: return ''.join(f.readlines())

class Text(ItemBase):
    "Basic item for <code>text</code> data in numericalized `ids`."
    def __init__(self, ids, text): self.data,self.text = np.array(ids, dtype=np.int64),text
    def __str__(self):  return str(self.text)

class TokenizeProcessor(PreProcessor):
    "`PreProcessor` that tokenizes the texts in `ds`."
    def __init__(self, ds:ItemList=None, tokenizer:Tokenizer=None, chunksize:int=10000, 
                 mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False):
        self.tokenizer,self.chunksize,self.mark_fields = ifnone(tokenizer, Tokenizer()),chunksize,mark_fields
        self.include_bos, self.include_eos = include_bos, include_eos

    def process_one(self, item):
        return self.tokenizer._process_all_1(_join_texts([item], self.mark_fields, self.include_bos, self.include_eos))[0]

    def process(self, ds):
        ds.items = _join_texts(ds.items, self.mark_fields, self.include_bos, self.include_eos)
        tokens = []
        for i in progress_bar(range(0,len(ds),self.chunksize), leave=False):
            tokens += self.tokenizer.process_all(ds.items[i:i+self.chunksize])
        ds.items = tokens

class NumericalizeProcessor(PreProcessor):
    "`PreProcessor` that numericalizes the tokens in `ds`."
    def __init__(self, ds:ItemList=None, vocab:Vocab=None, max_vocab:int=60000, min_freq:int=3):
        vocab = ifnone(vocab, ds.vocab if ds is not None else None)
        self.vocab,self.max_vocab,self.min_freq = vocab,max_vocab,min_freq

    def process_one(self,item): return np.array(self.vocab.numericalize(item), dtype=np.int64)
    def process(self, ds):
        if self.vocab is None: self.vocab = Vocab.create(ds.items, self.max_vocab, self.min_freq)
        ds.vocab = self.vocab
        super().process(ds)

class OpenFileProcessor(PreProcessor):
    "`PreProcessor` that opens the filenames and read the texts."
    def process(self, ds:Collection): ds.items = array([self.process_one(item) for item in ds.items], dtype=np.object)
    def process_one(self,item): return open_text(item) if isinstance(item, Path) else item

class TextList(ItemList):
    "Basic `ItemList` for text data."
    _bunch = TextClasDataBunch
    _processor = [TokenizeProcessor, NumericalizeProcessor]
    _is_lm = False

    def __init__(self, items:Iterator, vocab:Vocab=None, pad_idx:int=1, sep=' ', **kwargs):
        super().__init__(items, **kwargs)
        self.vocab,self.pad_idx,self.sep = vocab,pad_idx,sep
        self.copy_new += ['vocab', 'pad_idx', 'sep']

    def get(self, i):
        o = super().get(i)
        return o if self.vocab is None else Text(o, self.vocab.textify(o, self.sep))

    def label_for_lm(self, **kwargs):
        "A special labelling method for language models."
        self.__class__ = LMTextList
        kwargs['label_cls'] = LMLabelList
        return self.label_const(0, **kwargs)

    def reconstruct(self, t:Tensor):
        idx_min = (t != self.pad_idx).nonzero().min()
        idx_max = (t != self.pad_idx).nonzero().max()
        return Text(t[idx_min:idx_max+1], self.vocab.textify(t[idx_min:idx_max+1]))

    @classmethod
    def from_folder(cls, path:PathOrStr='.', extensions:Collection[str]=text_extensions, vocab:Vocab=None,
                    processor:PreProcessor=None, **kwargs)->'TextList':
        "Get the list of files in `path` that have a text suffix. `recurse` determines if we search subfolders."
        processor = ifnone(processor, [OpenFileProcessor(), TokenizeProcessor(), NumericalizeProcessor(vocab=vocab)])
        return super().from_folder(path=path, extensions=extensions, processor=processor, **kwargs)

    def show_xys(self, xs, ys, max_len:int=70)->None:
        "Show the `xs` (inputs) and `ys` (targets). `max_len` is the maximum number of tokens displayed."
        from IPython.display import display, HTML
        names = ['idx','text'] if self._is_lm else ['text','target']
        items = []
        for i, (x,y) in enumerate(zip(xs,ys)):
            txt_x = ' '.join(x.text.split(' ')[:max_len]) if max_len is not None else x.text
            items.append([i, txt_x] if self._is_lm else [txt_x, y])
        items = np.array(items)
        df = pd.DataFrame({n:items[:,i] for i,n in enumerate(names)}, columns=names)
        with pd.option_context('display.max_colwidth', -1):
            display(HTML(df.to_html(index=False)))

    def show_xyzs(self, xs, ys, zs, max_len:int=70):
        "Show `xs` (inputs), `ys` (targets) and `zs` (predictions). `max_len` is the maximum number of tokens displayed."
        from IPython.display import display, HTML
        items,names = [],['text','target','prediction']
        for i, (x,y,z) in enumerate(zip(xs,ys,zs)):
            txt_x = ' '.join(x.text.split(' ')[:max_len]) if max_len is not None else x.text
            items.append([txt_x, y, z])
        items = np.array(items)
        df = pd.DataFrame({n:items[:,i] for i,n in enumerate(names)}, columns=names)
        with pd.option_context('display.max_colwidth', -1):
            display(HTML(df.to_html(index=False)))

class LMLabelList(EmptyLabelList):
    "Basic `ItemList` for dummy labels."
    def __init__(self, items:Iterator, **kwargs):
        super().__init__(items, **kwargs)
        self.loss_func = CrossEntropyFlat()

class LMTextList(TextList):
    "Special `TextList` for a language model."
    _bunch = TextLMDataBunch
    _is_lm = True

def _join_texts(texts:Collection[str], mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False):
    if not isinstance(texts, np.ndarray): texts = np.array(texts)
    if is1d(texts): texts = texts[:,None]
    df = pd.DataFrame({i:texts[:,i] for i in range(texts.shape[1])})
    bos_tok = f'{BOS} ' if include_bos else ''
    text_col = f'{bos_tok}{FLD} {1} ' + df[0].astype(str) if mark_fields else f'{bos_tok}' + df[0].astype(str)
    for i in range(1,len(df.columns)):
        text_col += (f' {FLD} {i+1} ' if mark_fields else ' ') + df[i].astype(str)
    if include_eos: text_col = text_col + f' {EOS}'
    return text_col.values

def apply_rules(text, pre_rules=None, post_rules=None):
    "Apply `pre_rules` and `post_rules` to `text`"
    text = text.strip(' ')
    for r in ifnone(pre_rules, defaults.text_pre_rules): text = r(text)
    toks = text.split()
    for r in ifnone(post_rules, defaults.text_post_rules): toks = r(toks)
    return ' '.join(toks) 

def get_default_size(texts, max_vocab_sz):
    "Either max_vocab_sz or one quarter of the number of unique words in `texts`"
    cnt = Counter()
    for t in texts: 
        cnt.update(t.split())
        if len(cnt)//4 > max_vocab_sz: return max_vocab_sz
    res = len(cnt)//4
    while res%8 != 0: res+=1
    return res

full_char_coverage_langs = ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "ga", "hr", "hu",
                       "it","lt","lv","mt","nl","pl","pt","ro","sk","sl","sv"] # all European langs

def train_sentencepiece(texts:Collection[str], path:PathOrStr, pre_rules: ListRules=None, post_rules:ListRules=None, 
    vocab_sz:int=None, max_vocab_sz:int=30000, model_type:str='unigram', max_sentence_len:int=20480, lang='en',
    char_coverage=None, tmp_dir='tmp'):
    "Train a sentencepiece tokenizer on `texts` and save it in `path/tmp_dir`"
    from sentencepiece import SentencePieceTrainer
    cache_dir = Path(path)/tmp_dir
    os.makedirs(cache_dir, exist_ok=True)
    if vocab_sz is None: vocab_sz=get_default_size(texts, max_vocab_sz)
    raw_text_path = cache_dir / 'all_text.out'
    with open(raw_text_path, 'w') as f: f.write("\n".join(texts))
    spec_tokens = ['\u2581'+s for s in defaults.text_spec_tok]
    SentencePieceTrainer.Train(" ".join([
        f"--input={raw_text_path} --max_sentence_length={max_sentence_len}",
        f"--character_coverage={ifnone(char_coverage, 0.99999 if lang in full_char_coverage_langs else 0.9998)}",
        f"--unk_id={len(defaults.text_spec_tok)} --pad_id=-1 --bos_id=-1 --eos_id=-1",
        f"--user_defined_symbols={','.join(spec_tokens)}",
        f"--model_prefix={cache_dir/'spm'} --vocab_size={vocab_sz} --model_type={model_type}"]))
    raw_text_path.unlink()
    return cache_dir

class SPProcessor(PreProcessor):
    "`PreProcessor` that tokenizes and numericalizes with `sentencepiece`"
    def __init__(self, ds:ItemList=None, pre_rules: ListRules=None, post_rules:ListRules=None, vocab_sz:int=None,
                 max_vocab_sz:int=30000, model_type:str='unigram', max_sentence_len:int=20480, lang='en',
                 char_coverage=None, tmp_dir='tmp', mark_fields:bool=False, include_bos:bool=True, 
                 include_eos:bool=False, sp_model=None, sp_vocab=None, n_cpus:int=None):
        try: from sentencepiece import SentencePieceTrainer,SentencePieceProcessor
        except ImportError:
            raise Exception('sentencepiece module is missing: run `pip install sentencepiece`')
        self.pre_rules,self.post_rules = pre_rules,post_rules
        self.mark_fields,self.include_bos,self.include_eos = mark_fields,include_bos,include_eos
        self.sp_model,self.sp_vocab,self.n_cpus = sp_model,sp_vocab,ifnone(n_cpus,defaults.cpus)
        self.train_func = partial(train_sentencepiece, pre_rules=pre_rules, post_rules=post_rules, vocab_sz=vocab_sz,
                max_vocab_sz=max_vocab_sz, model_type=model_type, max_sentence_len=max_sentence_len, lang=lang,
                char_coverage=char_coverage, tmp_dir=tmp_dir)

    def process_one(self, item, join=True):
        if join: text = _join_texts([item], self.mark_fields, self.include_bos, self.include_eos)[0]
        text = apply_rules(text, pre_rules=self.pre_rules, post_rules=self.post_rules)
        return self._encode_batch([text])[0]

    def process(self, ds):
        ds.items = _join_texts(ds.items, self.mark_fields, self.include_bos, self.include_eos)
        ds.items = [apply_rules(t, pre_rules=self.pre_rules, post_rules=self.post_rules) 
                    for t in progress_bar(ds.items, leave=False)]
        if self.sp_model is None or self.sp_vocab is None:
            cache_dir = self.train_func(ds.items, ds.path)
            self.sp_model,self.sp_vocab = cache_dir/'spm.model',cache_dir/'spm.vocab'
        if not getattr(self, 'vocab', False): 
            with open(self.sp_vocab, 'r') as f: self.vocab = Vocab([line.split('\t')[0] for line in f.readlines()])
        if self.n_cpus <= 1: ds.items = self._encode_batch(ds.items)
        else:
            with ProcessPoolExecutor(self.n_cpus) as e:
                ds.items = np.array(sum(e.map(self._encode_batch, partition_by_cores(ds.items, self.n_cpus)), []))
        ds.vocab = self.vocab

    def _encode_batch(self, texts):
        from sentencepiece import SentencePieceProcessor
        tok = SentencePieceProcessor()
        tok.Load(str(self.sp_model))
        return [np.array(tok.EncodeAsIds(t)) for t in texts]

    @classmethod
    def load(cls, path:PathOrStr, tmp_dir:PathOrStr='tmp', name:str='spm'):
        cache_dir = Path(path)/tmp_dir
        return cls(sp_model=cache_dir/f'{name}.model', sp_vocab=cache_dir/f'{name}.vocab')
