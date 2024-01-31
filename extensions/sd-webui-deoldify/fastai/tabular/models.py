from ..torch_core import *
from ..layers import *
from ..basic_data import *
from ..basic_train import *
from ..train import ClassificationInterpretation

__all__ = ['TabularModel']

class TabularModel(Module):
    "Basic model for tabular data."
    def __init__(self, emb_szs:ListSizes, n_cont:int, out_sz:int, layers:Collection[int], ps:Collection[float]=None,
                 emb_drop:float=0., y_range:OptRange=None, use_bn:bool=True, bn_final:bool=False):
        super().__init__()
        ps = ifnone(ps, [0]*len(layers))
        ps = listify(ps, layers)
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont,self.y_range = n_emb,n_cont,y_range
        sizes = self.get_sizes(layers, out_sz)
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes)-2)] + [None]
        layers = []
        for i,(n_in,n_out,dp,act) in enumerate(zip(sizes[:-1],sizes[1:],[0.]+ps,actns)):
            layers += bn_drop_lin(n_in, n_out, bn=use_bn and i!=0, p=dp, actn=act)
        if bn_final: layers.append(nn.BatchNorm1d(sizes[-1]))
        self.layers = nn.Sequential(*layers)

    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.n_cont] + layers + [out_sz]

    def forward(self, x_cat:Tensor, x_cont:Tensor) -> Tensor:
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        x = self.layers(x)
        if self.y_range is not None:
            x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
        return x

@classmethod
def _cl_int_from_learner(cls, learn:Learner, ds_type=DatasetType.Valid, activ:nn.Module=None):
    "Creates an instance of 'ClassificationInterpretation"
    preds = learn.get_preds(ds_type=ds_type, activ=activ, with_loss=True)
    return cls(learn, *preds, ds_type=ds_type)

def _cl_int_plot_top_losses(self, k, largest:bool=True, return_table:bool=False)->Optional[plt.Figure]:
    "Generates a dataframe of 'top_losses' along with their prediction, actual, loss, and probability of the actual class."
    tl_val, tl_idx = self.top_losses(k, largest)
    classes = self.data.classes
    cat_names = self.data.x.cat_names
    cont_names = self.data.x.cont_names
    df = pd.DataFrame(columns=[['Prediction', 'Actual', 'Loss', 'Probability'] + cat_names + cont_names])
    for i, idx in enumerate(tl_idx):
        da, cl = self.data.dl(self.ds_type).dataset[idx]
        cl = int(cl)
        t1 = str(da)
        t1 = t1.split(';')
        arr = []
        arr.extend([classes[self.pred_class[idx]], classes[cl], f'{self.losses[idx]:.2f}',
                    f'{self.preds[idx][cl]:.2f}'])
        for x in range(len(t1)-1):
            _, value = t1[x].rsplit(' ', 1)
            arr.append(value)
        df.loc[i] = arr
    display(df)
    return_fig = return_table
    if ifnone(return_fig, defaults.return_fig): return df


ClassificationInterpretation.from_learner = _cl_int_from_learner
ClassificationInterpretation.plot_top_losses = _cl_int_plot_top_losses

def _learner_interpret(learn:Learner, ds_type:DatasetType = DatasetType.Valid):
    "Create a 'ClassificationInterpretation' object from 'learner' on 'ds_type'."
    return ClassificationInterpretation.from_learner(learn, ds_type=ds_type)

Learner.interpret = _learner_interpret
