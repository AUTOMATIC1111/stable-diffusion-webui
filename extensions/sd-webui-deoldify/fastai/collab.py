"Module support for Collaborative Filtering"
from .tabular import *
from . import tabular

__all__ = [*tabular.__all__, 'EmbeddingDotBias', 'EmbeddingNN', 'collab_learner', 'CollabDataBunch', 'CollabLine',
           'CollabList', 'CollabLearner']

class CollabProcessor(TabularProcessor):
    "Subclass `TabularProcessor for `process_one`."
    def process_one(self, item):
        res = super().process_one(item)
        return CollabLine(res.cats,res.conts,res.classes,res.names)

class CollabLine(TabularLine):
    "Base item for collaborative filtering, subclasses `TabularLine`."
    def __init__(self, cats, conts, classes, names):
        super().__init__(cats, conts, classes, names)
        self.data = [self.data[0][0],self.data[0][1]]

class CollabList(TabularList):
    "Base `ItemList` for collaborative filtering, subclasses `TabularList`."
    _item_cls,_label_cls,_processor = CollabLine,FloatList,CollabProcessor

    def reconstruct(self, t:Tensor): return CollabLine(tensor(t), tensor([]), self.classes, self.col_names)

class EmbeddingNN(TabularModel):
    "Subclass `TabularModel` to create a NN suitable for collaborative filtering."
    def __init__(self, emb_szs:ListSizes, layers:Collection[int]=None, ps:Collection[float]=None,
                 emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True, bn_final:bool=False):
        super().__init__(emb_szs=emb_szs, n_cont=0, out_sz=1, layers=layers, ps=ps, emb_drop=emb_drop, y_range=y_range,
                         use_bn=use_bn, bn_final=bn_final)

    def forward(self, users:LongTensor, items:LongTensor) -> Tensor:
        return super().forward(torch.stack([users,items], dim=1), None)

class EmbeddingDotBias(Module):
    "Base dot model for collaborative filtering."
    def __init__(self, n_factors:int, n_users:int, n_items:int, y_range:Tuple[float,float]=None):
        self.y_range = y_range
        (self.u_weight, self.i_weight, self.u_bias, self.i_bias) = [embedding(*o) for o in [
            (n_users, n_factors), (n_items, n_factors), (n_users,1), (n_items,1)
        ]]

    def forward(self, users:LongTensor, items:LongTensor) -> Tensor:
        dot = self.u_weight(users)* self.i_weight(items)
        res = dot.sum(1) + self.u_bias(users).squeeze() + self.i_bias(items).squeeze()
        if self.y_range is None: return res
        return torch.sigmoid(res) * (self.y_range[1]-self.y_range[0]) + self.y_range[0]

class CollabDataBunch(DataBunch):
    "Base `DataBunch` for collaborative filtering."
    @classmethod
    def from_df(cls, ratings:DataFrame, valid_pct:float=0.2, user_name:Optional[str]=None, item_name:Optional[str]=None,
                rating_name:Optional[str]=None, test:DataFrame=None, seed:int=None, path:PathOrStr='.', bs:int=64, 
                val_bs:int=None, num_workers:int=defaults.cpus, dl_tfms:Optional[Collection[Callable]]=None, 
                device:torch.device=None, collate_fn:Callable=data_collate, no_check:bool=False) -> 'CollabDataBunch':
        "Create a `DataBunch` suitable for collaborative filtering from `ratings`."
        user_name   = ifnone(user_name,  ratings.columns[0])
        item_name   = ifnone(item_name,  ratings.columns[1])
        rating_name = ifnone(rating_name,ratings.columns[2])
        cat_names = [user_name,item_name]
        src = (CollabList.from_df(ratings, cat_names=cat_names, procs=Categorify)
               .split_by_rand_pct(valid_pct=valid_pct, seed=seed).label_from_df(cols=rating_name))
        if test is not None: src.add_test(CollabList.from_df(test, cat_names=cat_names))
        return src.databunch(path=path, bs=bs, val_bs=val_bs, num_workers=num_workers, device=device, 
                             collate_fn=collate_fn, no_check=no_check)

class CollabLearner(Learner):
    "`Learner` suitable for collaborative filtering."
    def get_idx(self, arr:Collection, is_item:bool=True):
        "Fetch item or user (based on `is_item`) for all in `arr`. (Set model to `cpu` and no grad.)"
        m = self.model.eval().cpu()
        requires_grad(m,False)
        u_class,i_class = self.data.train_ds.x.classes.values()
        classes = i_class if is_item else u_class
        c2i = {v:k for k,v in enumerate(classes)}
        try: return tensor([c2i[o] for o in arr])
        except Exception as e: 
            print(f"""You're trying to access {'an item' if is_item else 'a user'} that isn't in the training data.
                  If it was in your original data, it may have been split such that it's only in the validation set now.""")

    def bias(self, arr:Collection, is_item:bool=True):
        "Bias for item or user (based on `is_item`) for all in `arr`. (Set model to `cpu` and no grad.)"
        idx = self.get_idx(arr, is_item)
        m = self.model
        layer = m.i_bias if is_item else m.u_bias
        return layer(idx).squeeze()

    def weight(self, arr:Collection, is_item:bool=True):
        "Bias for item or user (based on `is_item`) for all in `arr`. (Set model to `cpu` and no grad.)"
        idx = self.get_idx(arr, is_item)
        m = self.model
        layer = m.i_weight if is_item else m.u_weight
        return layer(idx)

def collab_learner(data, n_factors:int=None, use_nn:bool=False, emb_szs:Dict[str,int]=None, layers:Collection[int]=None, 
                   ps:Collection[float]=None, emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True, 
                   bn_final:bool=False, **learn_kwargs)->Learner:
    "Create a Learner for collaborative filtering on `data`."
    emb_szs = data.get_emb_szs(ifnone(emb_szs, {}))
    u,m = data.train_ds.x.classes.values()
    if use_nn: model = EmbeddingNN(emb_szs=emb_szs, layers=layers, ps=ps, emb_drop=emb_drop, y_range=y_range, 
                                   use_bn=use_bn, bn_final=bn_final, **learn_kwargs)
    else:      model = EmbeddingDotBias(n_factors, len(u), len(m), y_range=y_range)
    return CollabLearner(data, model, **learn_kwargs)

