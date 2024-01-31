"A `Callback` that saves tracked metrics into a persistent file."
#Contribution from devforfu: https://nbviewer.jupyter.org/gist/devforfu/ea0b3fcfe194dad323c3762492b05cae
from ..torch_core import *
from ..basic_data import DataBunch
from ..callback import *
from ..basic_train import Learner, LearnerCallback
from time import time
from fastprogress.fastprogress import format_time

__all__ = ['CSVLogger']

class CSVLogger(LearnerCallback):
    "A `LearnerCallback` that saves history of metrics while training `learn` into CSV `filename`."
    def __init__(self, learn:Learner, filename: str = 'history', append: bool = False): 
        super().__init__(learn)
        self.filename,self.path,self.append = filename,self.learn.path/f'{filename}.csv',append
        self.add_time = True

    def read_logged_file(self):  
        "Read the content of saved file"
        return pd.read_csv(self.path)

    def on_train_begin(self, **kwargs: Any) -> None:
        "Prepare file with metric names."
        self.path.parent.mkdir(parents=True, exist_ok=True)      
        self.file = self.path.open('a') if self.append else self.path.open('w')
        self.file.write(','.join(self.learn.recorder.names[:(None if self.add_time else -1)]) + '\n')
    
    def on_epoch_begin(self, **kwargs:Any)->None:
        if self.add_time: self.start_epoch = time()
        
    def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
        "Add a line with `epoch` number, `smooth_loss` and `last_metrics`."
        last_metrics = ifnone(last_metrics, [])
        stats = [str(stat) if isinstance(stat, int) else '#na#' if stat is None else f'{stat:.6f}'
                 for name, stat in zip(self.learn.recorder.names, [epoch, smooth_loss] + last_metrics)]
        if self.add_time: stats.append(format_time(time() - self.start_epoch))
        str_stats = ','.join(stats)
        self.file.write(str_stats + '\n')

    def on_train_end(self, **kwargs: Any) -> None:  
        "Close the file."
        self.file.close()
