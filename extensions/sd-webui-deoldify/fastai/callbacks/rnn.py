"Regroups lr adjustment to seq_len, AR and TAR"
from ..torch_core import *
from ..callback import *
from ..basic_train import Learner, LearnerCallback

__all__ = ['RNNTrainer']

class RNNTrainer(LearnerCallback):
    "`Callback` that regroups lr adjustment to seq_len, AR and TAR."
    def __init__(self, learn:Learner, alpha:float=0., beta:float=0.):
        super().__init__(learn)
        self.not_min += ['raw_out', 'out']
        self.alpha,self.beta = alpha,beta
        
    def on_epoch_begin(self, **kwargs):
        "Reset the hidden state of the model."
        self.learn.model.reset()

    def on_loss_begin(self, last_output:Tuple[Tensor,Tensor,Tensor], **kwargs):
        "Save the extra outputs for later and only returns the true output."
        self.raw_out,self.out = last_output[1],last_output[2]
        return {'last_output': last_output[0]}

    def on_backward_begin(self, last_loss:Rank0Tensor, last_input:Tensor, **kwargs):
        "Apply AR and TAR to `last_loss`."
        #AR and TAR
        if self.alpha != 0.:  last_loss += self.alpha * self.out[-1].float().pow(2).mean()
        if self.beta != 0.:
            h = self.raw_out[-1]
            if len(h)>1: last_loss += self.beta * (h[:,1:] - h[:,:-1]).float().pow(2).mean()
        return {'last_loss': last_loss}
