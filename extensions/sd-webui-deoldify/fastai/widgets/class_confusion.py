import math
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import permutations
from ..tabular import TabularDataBunch
from ..train import ClassificationInterpretation
import ipywidgets as widgets

class ClassConfusion():
    "Plot the most confused datapoints and statistics for the models misses." 
    def __init__(self, interp:ClassificationInterpretation, classlist:list, 
               is_ordered:bool=False, cut_off:int=100, varlist:list=None,
               figsize:tuple=(8,8)):
        self.interp = interp
        self._is_tab = isinstance(interp.learn.data, TabularDataBunch)
        if self._is_tab:
            if interp.learn.data.train_ds.x.cont_names != []: 
                for x in range(len(interp.learn.data.procs)):
                      if "Normalize" in str(interp.learn.data.procs[x]):
                            self.means = interp.learn.data.train_ds.x.processor[0].procs[x].means
                            self.stds = interp.learn.data.train_ds.x.processor[0].procs[x].stds
        self.is_ordered = is_ordered
        self.cut_off = cut_off
        self.figsize = figsize
        self.varlist = varlist
        self.classl = classlist
        self._show_losses(classlist)
        
    def _show_losses(self, classl:list, **kwargs):
        "Checks if the model is for Tabular or Images and gathers top losses"
        _, self.tl_idx = self.interp.top_losses(len(self.interp.losses))
        self._tab_losses() if self._is_tab else self._create_tabs()
        
    def _create_tabs(self):
        "Creates a tab for each variable"
        self.lis = self.classl if self.is_ordered else list(permutations(self.classl, 2))
        if self._is_tab:
            self._boxes = len(self.df_list)
            self._cols = math.ceil(math.sqrt(self._boxes))
            self._rows = math.ceil(self._boxes/self._cols)
            self.tbnames = list(self.df_list[0].columns)[:-1] if self.varlist is None else self.varlist
        else:
            vals = self.interp.most_confused()
            self._ranges = []
            self.tbnames = []
            self._boxes = int(input('Please enter a value for `k`, or the top images you will see: '))
            for x in iter(vals):
                for y in range(len(self.lis)):
                    if x[0:2] == self.lis[y]:
                        self._ranges.append(x[2])
                        self.tbnames.append(str(x[0] + ' | ' + x[1]))
        items = [widgets.Output() for i, tab in enumerate(self.tbnames)]
        self.tabs = widgets.Tab()
        self.tabs.children = items
        for i in range(len(items)):
            self.tabs.set_title(i, self.tbnames[i])
        self._populate_tabs()
        
    def _populate_tabs(self):
        "Adds relevant graphs to each tab"
        with tqdm(total=len(self.tbnames)) as pbar:
            for i, tab in enumerate(self.tbnames):
                with self.tabs.children[i]:
                    self._plot_tab(tab) if self._is_tab else self._plot_imgs(tab, i)
                pbar.update(1)
        display(self.tabs)
        
    def _plot_tab(self, tab:str):
        "Generates graphs"
        if self._boxes is not None:
            fig, ax = plt.subplots(self._boxes, figsize=self.figsize)
        else:
            fig, ax = plt.subplots(self._cols, self._rows, figsize=self.figsize)
        fig.subplots_adjust(hspace=.5)
        for j, x in enumerate(self.df_list):
            title = f'{"".join(x.columns[-1])} {tab} distribution'
            
            if self._boxes is None:
                row = int(j / self._cols)
                col = j % row
            if tab in self.cat_names:
                vals = pd.value_counts(x[tab].values)
                if self._boxes is not None:
                    if vals.nunique() < 10:
                        fig = vals.plot(kind='bar', title=title,  ax=ax[j], rot=0, width=.75)
                    elif vals.nunique() > self.cut_off:
                        print(f'Number of values is above {self.cut_off}')
                    else:
                        fig = vals.plot(kind='barh', title=title,  ax=ax[j], width=.75)   
                else:
                    fig = vals.plot(kind='barh', title=title,  ax=ax[row, col], width=.75)
            else:
                vals = x[tab]
                if self._boxes is not None:
                    axs = vals.plot(kind='hist', ax=ax[j], title=title, y='Frequency')
                else:
                    axs = vals.plot(kind='hist', ax=ax[row, col], title=title, y='Frequency')
                axs.set_ylabel('Frequency')
                if len(set(vals)) > 1:
                    vals.plot(kind='kde', ax=axs, title=title, secondary_y=True)
                else:
                    print('Less than two unique values, cannot graph the KDE')
        plt.show(fig)
        plt.tight_layout()

    def _plot_imgs(self, tab:str, i:int ,**kwargs):
        "Plots the most confused images"
        classes_gnd = self.interp.data.classes
        x = 0
        if self._ranges[i] < self._boxes:
            cols = math.ceil(math.sqrt(self._ranges[i]))
            rows = math.ceil(self._ranges[i]/cols)
        if self._ranges[i] < 4 or self._boxes < 4:
            cols = 2
            rows = 2
        else:
            cols = math.ceil(math.sqrt(self._boxes))
            rows = math.ceil(self._boxes/cols)
        fig, ax = plt.subplots(rows, cols, figsize=self.figsize)
        [axi.set_axis_off() for axi in ax.ravel()]
        for j, idx in enumerate(self.tl_idx):
            if self._boxes < x+1 or x > self._ranges[i]:
                break
            da, cl = self.interp.data.dl(self.interp.ds_type).dataset[idx]
            row = (int)(x / cols)
            col = x % cols
            if str(cl) == tab.split(' ')[0] and str(classes_gnd[self.interp.pred_class[idx]]) == tab.split(' ')[2]:
                img, lbl = self.interp.data.valid_ds[idx]
                fn = self.interp.data.valid_ds.x.items[idx]
                fn = re.search('([^/*]+)_\d+.*$', str(fn)).group(0)
                img.show(ax=ax[row, col])
                ax[row,col].set_title(fn)
                x += 1
        plt.show(fig)
        plt.tight_layout()

    def _tab_losses(self, **kwargs):
        "Gathers dataframes of the combinations data"
        classes = self.interp.data.classes
        cat_names = self.interp.data.x.cat_names
        cont_names = self.interp.data.x.cont_names
        comb = self.classl if self.is_ordered else list(permutations(self.classl,2))
        self.df_list = []
        arr = []
        for i, idx in enumerate(self.tl_idx):
            da, _ = self.interp.data.dl(self.interp.ds_type).dataset[idx]
            res = ''
            for c, n in zip(da.cats, da.names[:len(da.cats)]):
                string = f'{da.classes[n][c]}'
                if string == 'True' or string == 'False':
                    string += ';'
                    res += string
                else:
                    string = string[1:]
                    res += string + ';'
            for c, n in zip(da.conts, da.names[len(da.cats):]):
                res += f'{c:.4f};'
            arr.append(res)
        f = pd.DataFrame([ x.split(';')[:-1] for x in arr], columns=da.names)
        for i, var in enumerate(self.interp.data.cont_names):
            f[var] = f[var].apply(lambda x: float(x) * self.stds[var] + self.means[var])
        f['Original'] = 'Original'
        self.df_list.append(f)
        for j, x in enumerate(comb):
            arr = []
            for i, idx in enumerate(self.tl_idx):
                da, cl = self.interp.data.dl(self.interp.ds_type).dataset[idx]
                cl = int(cl)
                if classes[self.interp.pred_class[idx]] == comb[j][0] and classes[cl] == comb[j][1]:
                    res = ''
                    for c, n in zip(da.cats, da.names[:len(da.cats)]):
                        string = f'{da.classes[n][c]}'
                        if string == 'True' or string == 'False':
                            string += ';'
                            res += string
                        else:
                            string = string[1:]
                            res += string + ';'
                    for c, n in zip(da.conts, da.names[len(da.cats):]):
                        res += f'{c:.4f};'
                    arr.append(res)      
            f = pd.DataFrame([ x.split(';')[:-1] for x in arr], columns=da.names)
            for i, var in enumerate(self.interp.data.cont_names):
                f[var] = f[var].apply(lambda x: float(x) * self.stds[var] + self.means[var])
            f[str(x)] = str(x)
            self.df_list.append(f)
        self.cat_names = cat_names
        self._create_tabs()
