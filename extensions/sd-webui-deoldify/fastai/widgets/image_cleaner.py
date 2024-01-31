from ..torch_core import *
from ..basic_train import *
from ..basic_data import *
from ..vision.data import *
from ..vision.transform import *
from ..vision.image import *
from ..callbacks.hooks import *
from ..layers import *
from ipywidgets import widgets, Layout
from IPython.display import clear_output, display

__all__ = ['DatasetFormatter', 'ImageCleaner']

class DatasetFormatter():
    "Returns a dataset with the appropriate format and file indices to be displayed."
    @classmethod
    def from_toplosses(cls, learn, n_imgs=None, **kwargs):
        "Gets indices with top losses."
        train_ds, train_idxs = cls.get_toplosses_idxs(learn, n_imgs, **kwargs)
        return train_ds, train_idxs

    @classmethod
    def get_toplosses_idxs(cls, learn, n_imgs, **kwargs):
        "Sorts `ds_type` dataset by top losses and returns dataset and sorted indices."
        dl = learn.data.fix_dl
        if not n_imgs: n_imgs = len(dl.dataset)
        _,_,top_losses = learn.get_preds(ds_type=DatasetType.Fix, with_loss=True)
        idxs = torch.topk(top_losses, n_imgs)[1]
        return cls.padded_ds(dl.dataset, **kwargs), idxs

    def padded_ds(ll_input, size=(250, 300), resize_method=ResizeMethod.CROP, padding_mode='zeros', **kwargs):
        "For a LabelList `ll_input`, resize each image to `size` using `resize_method` and `padding_mode`."
        return ll_input.transform(tfms=crop_pad(), size=size, resize_method=resize_method, padding_mode=padding_mode)
    
    @classmethod
    def from_similars(cls, learn, layer_ls:list=[0, 7, 2], **kwargs):
        "Gets the indices for the most similar images."
        train_ds, train_idxs = cls.get_similars_idxs(learn, layer_ls, **kwargs)
        return train_ds, train_idxs

    @classmethod
    def get_similars_idxs(cls, learn, layer_ls, **kwargs):
        "Gets the indices for the most similar images in `ds_type` dataset"
        hook = hook_output(learn.model[layer_ls[0]][layer_ls[1]][layer_ls[2]])
        dl = learn.data.fix_dl

        ds_actns = cls.get_actns(learn, hook=hook, dl=dl, **kwargs)
        similarities = cls.comb_similarity(ds_actns, ds_actns, **kwargs)
        idxs = cls.sort_idxs(similarities)
        return cls.padded_ds(dl, **kwargs), idxs

    @staticmethod
    def get_actns(learn, hook:Hook, dl:DataLoader, pool=AdaptiveConcatPool2d, pool_dim:int=4, **kwargs):
        "Gets activations at the layer specified by `hook`, applies `pool` of dim `pool_dim` and concatenates"
        print('Getting activations...')

        actns = []
        learn.model.eval()
        with torch.no_grad():
            for (xb,yb) in progress_bar(dl):
                learn.model(xb)
                actns.append((hook.stored).cpu())

        if pool:
            pool = pool(pool_dim)
            return pool(torch.cat(actns)).view(len(dl.x),-1)
        else: return torch.cat(actns).view(len(dl.x),-1)


    @staticmethod
    def comb_similarity(t1: torch.Tensor, t2: torch.Tensor, **kwargs):
        # https://github.com/pytorch/pytorch/issues/11202
        "Computes the similarity function between each embedding of `t1` and `t2` matrices."
        print('Computing similarities...')

        w1 = t1.norm(p=2, dim=1, keepdim=True)
        w2 = w1 if t2 is t1 else t2.norm(p=2, dim=1, keepdim=True)

        t = torch.mm(t1, t2.t()) / (w1 * w2.t()).clamp(min=1e-8)
        return torch.tril(t, diagonal=-1) 

    def largest_indices(arr, n):
        "Returns the `n` largest indices from a numpy array `arr`."
        #https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        flat = arr.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, arr.shape)

    @classmethod
    def sort_idxs(cls, similarities):
        "Sorts `similarities` and return the indexes in pairs ordered by highest similarity."
        idxs = cls.largest_indices(similarities, len(similarities))
        idxs = [(idxs[0][i], idxs[1][i]) for i in range(len(idxs[0]))]
        return [e for l in idxs for e in l]

class ImageCleaner():
    "Displays images for relabeling or deletion and saves changes in `path` as 'cleaned.csv'."
    def __init__(self, dataset, fns_idxs, path, batch_size:int=5, duplicates=False):
        self._all_images,self._batch = [],[]
        self._path = Path(path)
        self._batch_size = batch_size
        if duplicates: self._batch_size = 2
        self._duplicates = duplicates
        self._labels = dataset.classes
        self._all_images = self.create_image_list(dataset, fns_idxs)
        self._csv_dict = {dataset.x.items[i]: dataset.y[i] for i in range(len(dataset))}
        self._deleted_fns = []
        self._skipped = 0
        self.render()

    @classmethod
    def make_img_widget(cls, img, layout=Layout(), format='jpg'):
        "Returns an image widget for specified file name `img`."
        return widgets.Image(value=img, format=format, layout=layout)

    @classmethod
    def make_button_widget(cls, label, file_path=None, handler=None, style=None, layout=Layout(width='auto')):
        "Return a Button widget with specified `handler`."
        btn = widgets.Button(description=label, layout=layout)
        if handler is not None: btn.on_click(handler)
        if style is not None: btn.button_style = style
        btn.file_path = file_path
        btn.flagged_for_delete = False
        return btn

    @classmethod
    def make_dropdown_widget(cls, description='Description', options=['Label 1', 'Label 2'], value='Label 1',
                            file_path=None, layout=Layout(), handler=None):
        "Return a Dropdown widget with specified `handler`."
        dd = widgets.Dropdown(description=description, options=options, value=value, layout=layout)
        if file_path is not None: dd.file_path = file_path
        if handler is not None: dd.observe(handler, names=['value'])
        return dd

    @classmethod
    def make_horizontal_box(cls, children, layout=Layout()):
        "Make a horizontal box with `children` and `layout`."
        return widgets.HBox(children, layout=layout)

    @classmethod
    def make_vertical_box(cls, children, layout=Layout(), duplicates=False):
        "Make a vertical box with `children` and `layout`."
        if not duplicates: return widgets.VBox(children, layout=layout)
        else: return widgets.VBox([children[0], children[2]], layout=layout)

    def create_image_list(self, dataset, fns_idxs):
        "Create a list of images, filenames and labels but first removing files that are not supposed to be displayed."
        items = dataset.x.items
        if self._duplicates:
            chunked_idxs = chunks(fns_idxs, 2)
            chunked_idxs = [chunk for chunk in chunked_idxs if Path(items[chunk[0]]).is_file() and Path(items[chunk[1]]).is_file()]
            return  [(dataset.x[i]._repr_jpeg_(), items[i], self._labels[dataset.y[i].data]) for chunk in chunked_idxs for i in chunk]
        else:
            return [(dataset.x[i]._repr_jpeg_(), items[i], self._labels[dataset.y[i].data]) for i in fns_idxs if
                    Path(items[i]).is_file()]

    def relabel(self, change):
        "Relabel images by moving from parent dir with old label `class_old` to parent dir with new label `class_new`."
        class_new,class_old,file_path = change.new,change.old,change.owner.file_path
        fp = Path(file_path)
        parent = fp.parents[1]
        self._csv_dict[fp] = class_new

    def next_batch(self, _):
        "Handler for 'Next Batch' button click. Delete all flagged images and renders next batch."
        for img_widget, delete_btn, fp, in self._batch:
            fp = delete_btn.file_path
            if (delete_btn.flagged_for_delete == True):
                self.delete_image(fp)
                self._deleted_fns.append(fp)
        self._all_images = self._all_images[self._batch_size:]
        self.empty_batch()
        self.render()

    def on_delete(self, btn):
        "Flag this image as delete or keep."
        btn.button_style = "" if btn.flagged_for_delete else "danger"
        btn.flagged_for_delete = not btn.flagged_for_delete

    def empty_batch(self): self._batch[:] = []

    def delete_image(self, file_path):
        del self._csv_dict[file_path]

    def empty(self):
        return len(self._all_images) == 0

    def get_widgets(self, duplicates):
        "Create and format widget set."
        widgets = []
        for (img,fp,human_readable_label) in self._all_images[:self._batch_size]:
            img_widget = self.make_img_widget(img, layout=Layout(height='250px', width='300px'))
            dropdown = self.make_dropdown_widget(description='', options=self._labels, value=human_readable_label,
                                                 file_path=fp, handler=self.relabel, layout=Layout(width='auto'))
            delete_btn = self.make_button_widget('Delete', file_path=fp, handler=self.on_delete)
            widgets.append(self.make_vertical_box([img_widget, dropdown, delete_btn],
                                                  layout=Layout(width='auto', height='300px',
                                                      overflow_x="hidden"), duplicates=duplicates))
            self._batch.append((img_widget, delete_btn, fp))
        return widgets

    def batch_contains_deleted(self):
        "Check if current batch contains already deleted images."
        if not self._duplicates: return False
        imgs = [self._all_images[:self._batch_size][0][1], self._all_images[:self._batch_size][1][1]]
        return any(img in self._deleted_fns for img in imgs)

    def write_csv(self):
        # Get first element's file path so we write CSV to same directory as our data
        csv_path = self._path/'cleaned.csv'
        with open(csv_path, 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['name','label'])
            for pair in self._csv_dict.items():
                pair = [os.path.relpath(pair[0], self._path), pair[1]]
                csv_writer.writerow(pair)
        return csv_path

    def render(self):
        "Re-render Jupyter cell for batch of images."
        clear_output()
        self.write_csv()
        if self.empty() and self._skipped>0:
            return display(f'No images to show :). {self._skipped} pairs were '
                    f'skipped since at least one of the images was deleted by the user.')
        elif self.empty():
            return display('No images to show :)')
        if self.batch_contains_deleted():
            self.next_batch(None)
            self._skipped += 1
        else:
            display(self.make_horizontal_box(self.get_widgets(self._duplicates)))
            display(self.make_button_widget('Next Batch', handler=self.next_batch, style="primary"))
