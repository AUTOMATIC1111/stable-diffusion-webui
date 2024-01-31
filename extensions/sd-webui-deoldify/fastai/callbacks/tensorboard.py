"Provides convenient callbacks for Learners that write model images, metrics/losses, stats and histograms to Tensorboard"
from ..basic_train import Learner
from ..basic_data import DatasetType, DataBunch
from ..vision import Image
from ..vision.gan import GANLearner
from ..callbacks import LearnerCallback
from ..core import *
from ..torch_core import *
from threading import Thread, Event
from time import sleep
from queue import Queue
import statistics
import torchvision.utils as vutils
from abc import ABC
#This is an optional dependency in fastai.  Must install separately.
try: from tensorboardX import SummaryWriter
except: print("To use this tracker, please run 'pip install tensorboardx'. Also you must have Tensorboard running to see results")

__all__=['LearnerTensorboardWriter', 'GANTensorboardWriter', 'ImageGenTensorboardWriter']

#---Example usage (applies to any of the callbacks)--- 
# proj_id = 'Colorize'
# tboard_path = Path('data/tensorboard/' + proj_id)
# learn.callback_fns.append(partial(GANTensorboardWriter, base_dir=tboard_path, name='GanLearner'))

class LearnerTensorboardWriter(LearnerCallback):
    "Broadly useful callback for Learners that writes to Tensorboard.  Writes model histograms, losses/metrics, and gradient stats."
    def __init__(self, learn:Learner, base_dir:Path, name:str, loss_iters:int=25, hist_iters:int=500, stats_iters:int=100):
        super().__init__(learn=learn)
        self.base_dir,self.name,self.loss_iters,self.hist_iters,self.stats_iters  = base_dir,name,loss_iters,hist_iters,stats_iters
        log_dir = base_dir/name
        self.tbwriter = SummaryWriter(str(log_dir))
        self.hist_writer = HistogramTBWriter()
        self.stats_writer = ModelStatsTBWriter()
        #self.graph_writer = GraphTBWriter()
        self.data = None
        self.metrics_root = '/metrics/'
        self._update_batches_if_needed()

    def _get_new_batch(self, ds_type:DatasetType)->Collection[Tensor]:
        "Retrieves new batch of DatasetType, and detaches it."
        return self.learn.data.one_batch(ds_type=ds_type, detach=True, denorm=False, cpu=False)

    def _update_batches_if_needed(self)->None:
        "one_batch function is extremely slow with large datasets.  This is caching the result as an optimization."
        if self.learn.data.valid_dl is None: return # Running learning rate finder, so return
        update_batches = self.data is not self.learn.data
        if not update_batches: return
        self.data = self.learn.data
        self.trn_batch = self._get_new_batch(ds_type=DatasetType.Train)
        self.val_batch = self._get_new_batch(ds_type=DatasetType.Valid)

    def _write_model_stats(self, iteration:int)->None:
        "Writes gradient statistics to Tensorboard."
        self.stats_writer.write(model=self.learn.model, iteration=iteration, tbwriter=self.tbwriter)

    def _write_training_loss(self, iteration:int, last_loss:Tensor)->None:
        "Writes training loss to Tensorboard."
        scalar_value = to_np(last_loss)
        tag = self.metrics_root + 'train_loss'
        self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=iteration)

    def _write_weight_histograms(self, iteration:int)->None:
        "Writes model weight histograms to Tensorboard."
        self.hist_writer.write(model=self.learn.model, iteration=iteration, tbwriter=self.tbwriter)

    def _write_scalar(self, name:str, scalar_value, iteration:int)->None:
        "Writes single scalar value to Tensorboard."
        tag = self.metrics_root + name
        self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=iteration)

    #TODO:  Relying on a specific hardcoded start_idx here isn't great.  Is there a better solution?
    def _write_metrics(self, iteration:int, last_metrics:MetricsList, start_idx:int=2)->None:
        "Writes training metrics to Tensorboard."
        recorder = self.learn.recorder
        for i, name in enumerate(recorder.names[start_idx:]):
            if last_metrics is None or len(last_metrics) < i+1: return
            scalar_value = last_metrics[i]
            self._write_scalar(name=name, scalar_value=scalar_value, iteration=iteration)

    def on_train_begin(self, **kwargs: Any) -> None:
        #self.graph_writer.write(model=self.learn.model, tbwriter=self.tbwriter,
                                #input_to_model=next(iter(self.learn.data.dl(DatasetType.Single)))[0])
        return

    def on_batch_end(self, last_loss:Tensor, iteration:int, **kwargs)->None:
        "Callback function that writes batch end appropriate data to Tensorboard."
        if iteration == 0: return
        self._update_batches_if_needed()
        if iteration % self.loss_iters == 0: self._write_training_loss(iteration=iteration, last_loss=last_loss)
        if iteration % self.hist_iters == 0: self._write_weight_histograms(iteration=iteration)

    # Doing stuff here that requires gradient info, because they get zeroed out afterwards in training loop
    def on_backward_end(self, iteration:int, **kwargs)->None:
        "Callback function that writes backward end appropriate data to Tensorboard."
        if iteration == 0: return
        self._update_batches_if_needed()
        if iteration % self.stats_iters == 0: self._write_model_stats(iteration=iteration)

    def on_epoch_end(self, last_metrics:MetricsList, iteration:int, **kwargs)->None:
        "Callback function that writes epoch end appropriate data to Tensorboard."
        self._write_metrics(iteration=iteration, last_metrics=last_metrics)

# TODO:  We're overriding almost everything here.  Seems like a good idea to question that ("is a" vs "has a")
class GANTensorboardWriter(LearnerTensorboardWriter):
    "Callback for GANLearners that writes to Tensorboard.  Extends LearnerTensorboardWriter and adds output image writes."
    def __init__(self, learn:GANLearner, base_dir:Path, name:str, loss_iters:int=25, hist_iters:int=500, 
                stats_iters:int=100, visual_iters:int=100):
        super().__init__(learn=learn, base_dir=base_dir, name=name, loss_iters=loss_iters, hist_iters=hist_iters, stats_iters=stats_iters)
        self.visual_iters = visual_iters
        self.img_gen_vis = ImageTBWriter()
        self.gen_stats_updated = True
        self.crit_stats_updated = True

    def _write_weight_histograms(self, iteration:int)->None:
        "Writes model weight histograms to Tensorboard."
        generator, critic = self.learn.gan_trainer.generator, self.learn.gan_trainer.critic
        self.hist_writer.write(model=generator, iteration=iteration, tbwriter=self.tbwriter, name='generator')
        self.hist_writer.write(model=critic,    iteration=iteration, tbwriter=self.tbwriter, name='critic')

    def _write_gen_model_stats(self, iteration:int)->None:
        "Writes gradient statistics for generator to Tensorboard."
        generator = self.learn.gan_trainer.generator
        self.stats_writer.write(model=generator, iteration=iteration, tbwriter=self.tbwriter, name='gen_model_stats')
        self.gen_stats_updated = True

    def _write_critic_model_stats(self, iteration:int)->None:
        "Writes gradient statistics for critic to Tensorboard."
        critic = self.learn.gan_trainer.critic
        self.stats_writer.write(model=critic, iteration=iteration, tbwriter=self.tbwriter, name='crit_model_stats')
        self.crit_stats_updated = True

    def _write_model_stats(self, iteration:int)->None:
        "Writes gradient statistics to Tensorboard."
        # We don't want to write stats when model is not iterated on and hence has zeroed out gradients
        gen_mode = self.learn.gan_trainer.gen_mode
        if gen_mode and not self.gen_stats_updated: self._write_gen_model_stats(iteration=iteration)
        if not gen_mode and not self.crit_stats_updated: self._write_critic_model_stats(iteration=iteration)

    def _write_training_loss(self, iteration:int, last_loss:Tensor)->None:
        "Writes training loss to Tensorboard."
        recorder = self.learn.gan_trainer.recorder
        if len(recorder.losses) == 0: return
        scalar_value = to_np((recorder.losses[-1:])[0])
        tag = self.metrics_root + 'train_loss'
        self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=iteration)

    def _write_images(self, iteration:int)->None:
        "Writes model generated, original and real images to Tensorboard."
        trainer = self.learn.gan_trainer
        #TODO:  Switching gen_mode temporarily seems a bit hacky here.  Certainly not a good side-effect.  Is there a better way?
        gen_mode = trainer.gen_mode
        try:
            trainer.switch(gen_mode=True)
            self.img_gen_vis.write(learn=self.learn, trn_batch=self.trn_batch, val_batch=self.val_batch, 
                                    iteration=iteration, tbwriter=self.tbwriter)
        finally: trainer.switch(gen_mode=gen_mode)

    def on_batch_end(self, iteration:int, **kwargs)->None:
        "Callback function that writes batch end appropriate data to Tensorboard."
        super().on_batch_end(iteration=iteration, **kwargs)
        if iteration == 0: return
        if iteration % self.visual_iters == 0: self._write_images(iteration=iteration)

    def on_backward_end(self, iteration:int, **kwargs)->None:
        "Callback function that writes backward end appropriate data to Tensorboard."
        if iteration == 0: return
        self._update_batches_if_needed()
        #TODO:  This could perhaps be implemented as queues of requests instead but that seemed like overkill. 
        # But I'm not the biggest fan of maintaining these boolean flags either... Review pls.
        if iteration % self.stats_iters == 0: self.gen_stats_updated, self.crit_stats_updated = False, False
        if not (self.gen_stats_updated and self.crit_stats_updated): self._write_model_stats(iteration=iteration)

class ImageGenTensorboardWriter(LearnerTensorboardWriter):
    "Callback for non-GAN image generating Learners that writes to Tensorboard.  Extends LearnerTensorboardWriter and adds output image writes."
    def __init__(self, learn:Learner, base_dir:Path, name:str, loss_iters:int=25, hist_iters:int=500, stats_iters:int=100, 
                 visual_iters:int=100):
        super().__init__(learn=learn, base_dir=base_dir, name=name, loss_iters=loss_iters, hist_iters=hist_iters, 
                         stats_iters=stats_iters)
        self.visual_iters = visual_iters
        self.img_gen_vis = ImageTBWriter()

    def _write_images(self, iteration:int)->None:
        "Writes model generated, original and real images to Tensorboard"
        self.img_gen_vis.write(learn=self.learn, trn_batch=self.trn_batch, val_batch=self.val_batch, iteration=iteration, 
                               tbwriter=self.tbwriter)

    def on_batch_end(self, iteration:int, **kwargs)->None:
        "Callback function that writes batch end appropriate data to Tensorboard."
        super().on_batch_end(iteration=iteration, **kwargs)
        if iteration == 0: return
        if iteration % self.visual_iters == 0: 
            self._write_images(iteration=iteration)

class TBWriteRequest(ABC):
    "A request object for Tensorboard writes.  Useful for queuing up and executing asynchronous writes."
    def __init__(self, tbwriter: SummaryWriter, iteration:int):
        super().__init__()
        self.tbwriter = tbwriter
        self.iteration = iteration

    @abstractmethod
    def write(self)->None: pass   

# SummaryWriter writes tend to block quite a bit.  This gets around that and greatly boosts performance.
# Not all tensorboard writes are using this- just the ones that take a long time.  Note that the 
# SummaryWriter does actually use a threadsafe consumer/producer design ultimately to write to Tensorboard, 
# so writes done outside of this async loop should be fine.
class AsyncTBWriter():
    "Callback for GANLearners that writes to Tensorboard.  Extends LearnerTensorboardWriter and adds output image writes."
    def __init__(self):
        super().__init__()
        self.stop_request = Event()
        self.queue = Queue()
        self.thread = Thread(target=self._queue_processor, daemon=True)
        self.thread.start()

    def request_write(self, request: TBWriteRequest)->None:
        "Queues up an asynchronous write request to Tensorboard."
        if self.stop_request.isSet(): return
        self.queue.put(request)

    def _queue_processor(self)->None:
        "Processes queued up write requests asynchronously to Tensorboard."
        while not self.stop_request.isSet():
            while not self.queue.empty():
                if self.stop_request.isSet(): return
                request = self.queue.get()
                request.write()
            sleep(0.2)

    #Provided this to stop thread explicitly or by context management (with statement) but thread should end on its own 
    # upon program exit, due to being a daemon.  So using this is probably unecessary.
    def close(self)->None:
        "Stops asynchronous request queue processing thread."
        self.stop_request.set()
        self.thread.join()

    # Nothing to do, thread already started.  Could start thread here to enforce use of context manager 
    # (but that sounds like a pain and a bit unweildy and unecessary for actual usage)
    def __enter__(self): pass

    def __exit__(self, exc_type, exc_value, traceback): self.close()

asyncTBWriter = AsyncTBWriter() 

class ModelImageSet():
    "Convenience object that holds the original, real(target) and generated versions of a single image fed to a model."
    @staticmethod
    def get_list_from_model(learn:Learner, ds_type:DatasetType, batch:Tuple)->[]:
        "Factory method to convert a batch of model images to a list of ModelImageSet."
        image_sets = []
        x,y = batch[0],batch[1]
        preds=[]
        preds = learn.pred_batch(ds_type=ds_type, batch=(x,y), reconstruct=True)  
        for orig_px, real_px, gen in zip(x,y,preds):
            orig, real = Image(px=orig_px), Image(px=real_px)
            image_set = ModelImageSet(orig=orig, real=real, gen=gen)
            image_sets.append(image_set)
        return image_sets  

    def __init__(self, orig:Image, real:Image, gen:Image): self.orig, self.real, self.gen = orig, real, gen

class HistogramTBRequest(TBWriteRequest):
    "Request object for model histogram writes to Tensorboard."
    def __init__(self, model:nn.Module, iteration:int, tbwriter:SummaryWriter, name:str):
        super().__init__(tbwriter=tbwriter, iteration=iteration)
        self.params = [(name, values.clone().detach().cpu()) for (name, values) in model.named_parameters()]
        self.name = name

    def _write_histogram(self, param_name:str, values)->None:
        "Writes single model histogram to Tensorboard."
        tag = self.name + '/weights/' + param_name
        self.tbwriter.add_histogram(tag=tag, values=values, global_step=self.iteration)

    def write(self)->None:
        "Writes model histograms to Tensorboard."
        for param_name, values in self.params: self._write_histogram(param_name=param_name, values=values)

#If this isn't done async then this is sloooooow
class HistogramTBWriter():
    "Writes model histograms to Tensorboard."
    def __init__(self): super().__init__()

    def write(self, model:nn.Module, iteration:int, tbwriter:SummaryWriter, name:str='model')->None:
        "Writes model histograms to Tensorboard."
        request = HistogramTBRequest(model=model, iteration=iteration, tbwriter=tbwriter, name=name)
        asyncTBWriter.request_write(request)

class ModelStatsTBRequest(TBWriteRequest):
    "Request object for model gradient statistics writes to Tensorboard."
    def __init__(self, model:nn.Module, iteration:int, tbwriter:SummaryWriter, name:str):
        super().__init__(tbwriter=tbwriter, iteration=iteration)
        self.gradients = [x.grad.clone().detach().cpu() for x in model.parameters() if x.grad is not None]
        self.name = name

    def _add_gradient_scalar(self, name:str, scalar_value)->None:
        "Writes a single scalar value for a gradient statistic to Tensorboard."
        tag = self.name + '/gradients/' + name
        self.tbwriter.add_scalar(tag=tag, scalar_value=scalar_value, global_step=self.iteration)

    def _write_avg_norm(self, norms:[])->None:
        "Writes the average norm of the gradients to Tensorboard."
        avg_norm = sum(norms)/len(self.gradients)
        self._add_gradient_scalar('avg_norm', scalar_value=avg_norm)

    def _write_median_norm(self, norms:[])->None:
        "Writes the median norm of the gradients to Tensorboard."
        median_norm = statistics.median(norms)
        self._add_gradient_scalar('median_norm', scalar_value=median_norm)

    def _write_max_norm(self, norms:[])->None:
        "Writes the maximum norm of the gradients to Tensorboard."
        max_norm = max(norms)
        self._add_gradient_scalar('max_norm', scalar_value=max_norm)

    def _write_min_norm(self, norms:[])->None:
        "Writes the minimum norm of the gradients to Tensorboard."
        min_norm = min(norms)
        self._add_gradient_scalar('min_norm', scalar_value=min_norm)

    def _write_num_zeros(self)->None:
        "Writes the number of zeroes in the gradients to Tensorboard."
        gradient_nps = [to_np(x.data) for x in self.gradients]
        num_zeros = sum((np.asarray(x) == 0.0).sum() for x in gradient_nps)
        self._add_gradient_scalar('num_zeros', scalar_value=num_zeros)

    def _write_avg_gradient(self)->None:
        "Writes the average of the gradients to Tensorboard."
        avg_gradient = sum(x.data.mean() for x in self.gradients)/len(self.gradients)
        self._add_gradient_scalar('avg_gradient', scalar_value=avg_gradient)

    def _write_median_gradient(self)->None:
        "Writes the median of the gradients to Tensorboard."
        median_gradient = statistics.median(x.data.median() for x in self.gradients)
        self._add_gradient_scalar('median_gradient', scalar_value=median_gradient)

    def _write_max_gradient(self)->None:
        "Writes the maximum of the gradients to Tensorboard."
        max_gradient = max(x.data.max() for x in self.gradients)
        self._add_gradient_scalar('max_gradient', scalar_value=max_gradient)

    def _write_min_gradient(self)->None:
        "Writes the minimum of the gradients to Tensorboard."
        min_gradient = min(x.data.min() for x in self.gradients)
        self._add_gradient_scalar('min_gradient', scalar_value=min_gradient)

    def write(self)->None:
        "Writes model gradient statistics to Tensorboard."
        if len(self.gradients) == 0: return
        norms = [x.data.norm() for x in self.gradients]
        self._write_avg_norm(norms=norms)
        self._write_median_norm(norms=norms)
        self._write_max_norm(norms=norms)
        self._write_min_norm(norms=norms)
        self._write_num_zeros()
        self._write_avg_gradient()
        self._write_median_gradient()
        self._write_max_gradient()
        self._write_min_gradient()

class ModelStatsTBWriter():
    "Writes model gradient statistics to Tensorboard."
    def write(self, model:nn.Module, iteration:int, tbwriter:SummaryWriter, name:str='model_stats')->None:
        "Writes model gradient statistics to Tensorboard."
        request = ModelStatsTBRequest(model=model, iteration=iteration, tbwriter=tbwriter, name=name)
        asyncTBWriter.request_write(request)

class ImageTBRequest(TBWriteRequest):
    "Request object for model image output writes to Tensorboard."
    def __init__(self, learn:Learner, batch:Tuple, iteration:int, tbwriter:SummaryWriter, ds_type:DatasetType):
        super().__init__(tbwriter=tbwriter, iteration=iteration)
        self.image_sets = ModelImageSet.get_list_from_model(learn=learn, batch=batch, ds_type=ds_type)
        self.ds_type = ds_type

    def _write_images(self, name:str, images:[Tensor])->None:
        "Writes list of images as tensors to Tensorboard."
        tag = self.ds_type.name + ' ' + name
        self.tbwriter.add_image(tag=tag, img_tensor=vutils.make_grid(images, normalize=True), global_step=self.iteration)

    def _get_image_tensors(self)->([Tensor], [Tensor], [Tensor]):
        "Gets list of image tensors from lists of Image objects, as a tuple of original, generated and real(target) images."
        orig_images, gen_images, real_images = [], [], []
        for image_set in self.image_sets:
            orig_images.append(image_set.orig.px)
            gen_images.append(image_set.gen.px)
            real_images.append(image_set.real.px) 
        return orig_images, gen_images, real_images  

    def write(self)->None:
        "Writes original, generated and real(target) images to Tensorboard."
        orig_images, gen_images, real_images = self._get_image_tensors()
        self._write_images(name='orig images', images=orig_images)
        self._write_images(name='gen images',  images=gen_images)
        self._write_images(name='real images', images=real_images)

#If this isn't done async then this is noticeably slower
class ImageTBWriter():
    "Writes model image output to Tensorboard."
    def __init__(self): super().__init__()

    def write(self, learn:Learner, trn_batch:Tuple, val_batch:Tuple, iteration:int, tbwriter:SummaryWriter)->None:
        "Writes training and validation batch images to Tensorboard."
        self._write_for_dstype(learn=learn, batch=val_batch, iteration=iteration, tbwriter=tbwriter, ds_type=DatasetType.Valid)
        self._write_for_dstype(learn=learn, batch=trn_batch, iteration=iteration, tbwriter=tbwriter, ds_type=DatasetType.Train)

    def _write_for_dstype(self, learn:Learner, batch:Tuple, iteration:int, tbwriter:SummaryWriter, ds_type:DatasetType)->None:
        "Writes batch images of specified DatasetType to Tensorboard."
        request = ImageTBRequest(learn=learn, batch=batch, iteration=iteration, tbwriter=tbwriter, ds_type=ds_type)
        asyncTBWriter.request_write(request)

class GraphTBRequest(TBWriteRequest):
    "Request object for model histogram writes to Tensorboard."
    def __init__(self, model:nn.Module, tbwriter:SummaryWriter, input_to_model:torch.Tensor):
        super().__init__(tbwriter=tbwriter, iteration=0)
        self.model,self.input_to_model = model,input_to_model

    def write(self)->None:
        "Writes single model graph to Tensorboard."
        self.tbwriter.add_graph(model=self.model, input_to_model=self.input_to_model)

class GraphTBWriter():
    "Writes model network graph to Tensorboard."
    def write(self, model:nn.Module, tbwriter:SummaryWriter, input_to_model:torch.Tensor)->None:
        "Writes model graph to Tensorboard."
        request = GraphTBRequest(model=model, tbwriter=tbwriter, input_to_model=input_to_model)
        asyncTBWriter.request_write(request)
