"Callback support for half precision (fp16) training. Increases training speed."
from ..torch_core import *
from ..callback import *
from ..basic_train import *
from torch._utils import _unflatten_dense_tensors
from torch.nn.utils import parameters_to_vector

__all__ = ['MixedPrecision']

def get_master(layer_groups:ModuleList, flat_master:bool=False) -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
    "Return two lists, one for the model parameters in FP16 and one for the master parameters in FP32."
    split_params = split_no_wd_params(layer_groups)
    model_params = [[param for param in pg if param.requires_grad] for pg in split_params]
    if flat_master:
        master_params = []
        for lg in model_params:
            if len(lg) !=0 :
                mp = parameters_to_vector([param.data.float() for param in lg])
                mp = torch.nn.Parameter(mp, requires_grad=True)
                if mp.grad is None: mp.grad = mp.new(*mp.size())
                master_params.append([mp])
            else: master_params.append([])
        return model_params, master_params
    else:
        master_params = [[param.clone().float().detach() for param in lg] for lg in model_params]
        for mp in master_params:
            for param in mp: param.requires_grad = True
        return model_params, master_params

def model_g2master_g(model_params:Sequence[Tensor], master_params:Sequence[Tensor], flat_master:bool=False)->None:
    "Copy the `model_params` gradients to `master_params` for the optimizer step."
    if flat_master:
        for model_group,master_group in zip(model_params,master_params):
            if len(master_group) != 0:
                if master_group[0].grad is None: master_group[0].grad = master_group[0].data.new(*master_group[0].data.size())
                master_group[0].grad.data.copy_(parameters_to_vector([p.grad.data.float() for p in model_group]))
    else:
        for model_group,master_group in zip(model_params,master_params):
            for model, master in zip(model_group, master_group):
                if model.grad is not None:
                    if master.grad is None: master.grad = master.data.new(*master.data.size())
                    master.grad.data.copy_(model.grad.data)
                else: master.grad = None

def master2model(model_params:Sequence[Tensor], master_params:Sequence[Tensor], flat_master:bool=False)->None:
    "Copy `master_params` to `model_params`."
    if flat_master:
        for model_group,master_group in zip(model_params,master_params):
            if len(model_group) != 0:
                for model, master in zip(model_group, _unflatten_dense_tensors(master_group[0].data, model_group)):
                    model.data.copy_(master)
    else:
        for model_group,master_group in zip(model_params,master_params):
            for model, master in zip(model_group, master_group): model.data.copy_(master.data)

def grad_overflow(param_group):
    for group in param_group:
        for p in group:
            if p.grad is not None:
                s = float(p.grad.data.float().sum())
                if s == float('inf') or s == float('-inf') or s != s: return True
    return False

class MixedPrecision(LearnerCallback):
    _order = 999 #Need to run after things that could call on_backward_begin and change the loss
    "Callback that handles mixed-precision training."
    def __init__(self, learn:Learner, loss_scale:float=None, max_noskip:int=1000, dynamic:bool=True, clip:float=None,
                 flat_master:bool=False, max_scale:float=2**24):
        super().__init__(learn)
        self.flat_master,self.dynamic,self.max_noskip,self.clip,self.max_scale = flat_master,dynamic,max_noskip,clip,max_scale
        self.loss_scale = ifnone(loss_scale, 2**16 if dynamic else 512)
        self.not_min += ['model_params', 'master_params']
        assert torch.backends.cudnn.enabled, "Mixed precision training requires cudnn."
        self.opt = None

    def on_train_begin(self, **kwargs:Any)->None:
        "Prepare the master model."
        #Get a copy of the model params in FP32
        self.model_params, self.master_params = get_master(self.learn.layer_groups, self.flat_master)
        #Changes the optimizer so that the optimization step is done in FP32.
        new_opt = self.learn.opt.new_with_params(self.master_params)
        if self.opt is not None:
            self.opt.lr,self.opt.wd = self.learn.opt.lr,self.learn.opt.wd
            new_opt.load_state_dict(self.opt)
        self.learn.opt.opt = new_opt.opt
        self.noskip = 0

    def on_loss_begin(self, last_output:Tensor, **kwargs:Any) -> Tensor:
        "Convert half precision output to FP32 to avoid reduction overflow."
        return {'last_output': to_float(last_output)}

    def on_backward_begin(self, last_loss:Rank0Tensor, **kwargs:Any) -> Rank0Tensor:
        "Scale gradients up by `self.loss_scale` to prevent underflow."
        #To avoid gradient underflow, we scale the gradients
        ret_loss = last_loss * self.loss_scale
        return {'last_loss': ret_loss}

    def on_backward_end(self, **kwargs:Any)->None:
        "Convert the gradients back to FP32 and divide them by the scale."
        if self.dynamic and grad_overflow(self.model_params) and self.loss_scale > 1:
            self.loss_scale /= 2
            self.noskip = 0
            #The step will be skipped since we don't update the master grads so they are all None or zero
        else:
            model_g2master_g(self.model_params, self.master_params, self.flat_master)
            for group in self.master_params:
                for param in group:
                    if param.grad is not None: param.grad.div_(self.loss_scale)
            if self.clip is not None:
                for group in self.master_params: nn.utils.clip_grad_norm_(group, self.clip)
            if not self.dynamic: return
            self.noskip += 1
            if self.noskip >= self.max_noskip and self.loss_scale < self.max_scale:
                self.loss_scale *= 2
                self.noskip = 0

    def on_step_end(self, **kwargs:Any)->None:
        "Update the params from master to model and zero grad."
        #Zeros the gradients of the model since the optimizer is disconnected.
        self.learn.model.zero_grad()
        #Update the params from master to model.
        master2model(self.model_params, self.master_params, self.flat_master)
