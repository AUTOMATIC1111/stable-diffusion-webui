from ..core import *
from ..callback import *
from ..basic_train import Learner, LearnerCallback

__all__ = ['GeneralScheduler', 'TrainingPhase']

@dataclass
class TrainingPhase():
    "Schedule hyper-parameters for a phase of `length` iterations."
    length:int
    
    def __post_init__(self): self.scheds = dict()
    def schedule_hp(self, name, vals, anneal=None):
        "Adds a schedule for `name` between `vals` using `anneal`."
        self.scheds[name] = Scheduler(vals, self.length, anneal)
        return self

class GeneralScheduler(LearnerCallback):
    "Schedule multiple `TrainingPhase` for a `Learner`."
    def __init__(self, learn:Learner, phases:Collection[TrainingPhase], start_epoch:int=None):
        super().__init__(learn)
        self.phases,self.start_epoch = phases,start_epoch

    def on_train_begin(self, epoch:int, **kwargs:Any)->None:
        "Initialize the schedulers for training."
        res = {'epoch':self.start_epoch} if self.start_epoch is not None else None
        self.start_epoch = ifnone(self.start_epoch, epoch)
        self.scheds = [p.scheds for p in self.phases]
        self.opt = self.learn.opt
        for k,v in self.scheds[0].items(): 
            v.restart()
            self.opt.set_stat(k, v.start)
        self.idx_s = 0
        return res
    
    def jump_to_epoch(self, epoch:int)->None:
        for _ in range(len(self.learn.data.train_dl) * epoch):
            self.on_batch_end(True)

    def on_batch_end(self, train, **kwargs:Any)->None:
        "Take a step in lr,mom sched, start next stepper when the current one is complete."
        if train:
            if self.idx_s >= len(self.scheds): return {'stop_training': True, 'stop_epoch': True}
            sched = self.scheds[self.idx_s]
            for k,v in sched.items(): self.opt.set_stat(k, v.step())
            if list(sched.values())[0].is_done: self.idx_s += 1