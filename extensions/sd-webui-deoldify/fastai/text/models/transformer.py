from ...torch_core import *
from ...layers import *
from .awd_lstm import RNNDropout, LinearDecoder, SequentialRNN

__all__ = ['Activation', 'PositionalEncoding', 'GeLU', 'Swish', 'feed_forward', 'MultiHeadAttention', 'MultiHeadRelativeAttention',
           'DecoderLayer', 'Transformer', 'TransformerXL', 'tfmer_lm_config', 'tfmer_clas_config', 'tfmer_lm_split', 'tfmer_clas_split',
           'tfmerXL_lm_config', 'tfmerXL_clas_config', 'tfmerXL_lm_split', 'tfmerXL_clas_split']

Activation = Enum('Activation', 'ReLU Swish GeLU')

class PositionalEncoding(Module):
    "Encode the position with a sinusoid."
    def __init__(self, d:int): self.register_buffer('freq', 1 / (10000 ** (torch.arange(0., d, 2.)/d)))

    def forward(self, pos:Tensor):
        inp = torch.ger(pos, self.freq)
        enc = torch.cat([inp.sin(), inp.cos()], dim=-1)
        return enc

class GeLU(Module):
    def forward(self, x): return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Swish(Module):
    def forward(self, x): return x * torch.sigmoid(x)

_activ_func = {Activation.ReLU:nn.ReLU(inplace=True), Activation.GeLU:GeLU(), Activation.Swish: Swish()}

def feed_forward(d_model:int, d_ff:int, ff_p:float=0., act:Activation=Activation.ReLU, double_drop:bool=True):
    layers = [nn.Linear(d_model, d_ff), _activ_func[act]]
    if double_drop: layers.append(nn.Dropout(ff_p))
    return SequentialEx(*layers, nn.Linear(d_ff, d_model), nn.Dropout(ff_p), MergeLayer(), nn.LayerNorm(d_model))

class MultiHeadAttention(Module):
    "MutiHeadAttention."
    def __init__(self, n_heads:int, d_model:int, d_head:int=None, resid_p:float=0., attn_p:float=0., bias:bool=True,
                 scale:bool=True):
        d_head = ifnone(d_head, d_model//n_heads)
        self.n_heads,self.d_head,self.scale = n_heads,d_head,scale
        self.attention = nn.Linear(d_model, 3 * n_heads * d_head, bias=bias)
        self.out = nn.Linear(n_heads * d_head, d_model, bias=bias)
        self.drop_att,self.drop_res = nn.Dropout(attn_p),nn.Dropout(resid_p)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x:Tensor, mask:Tensor=None, **kwargs):
        return self.ln(x + self.drop_res(self.out(self._apply_attention(x, mask=mask, **kwargs))))

    def _apply_attention(self, x:Tensor, mask:Tensor=None):
        bs,x_len = x.size(0),x.size(1)
        wq,wk,wv = torch.chunk(self.attention(x), 3, dim=-1)
        wq,wk,wv = map(lambda x:x.view(bs, x.size(1), self.n_heads, self.d_head), (wq,wk,wv))
        wq,wk,wv = wq.permute(0, 2, 1, 3),wk.permute(0, 2, 3, 1),wv.permute(0, 2, 1, 3)
        attn_score = torch.matmul(wq, wk)
        if self.scale: attn_score.div_(self.d_head ** 0.5)
        if mask is not None:
            attn_score = attn_score.float().masked_fill(mask, -float('inf')).type_as(attn_score)
        attn_prob = self.drop_att(F.softmax(attn_score, dim=-1))
        attn_vec = torch.matmul(attn_prob, wv)
        return attn_vec.permute(0, 2, 1, 3).contiguous().contiguous().view(bs, x_len, -1)

    def _attention_einsum(self, x, mask=None):
        # Permute and matmul is a little bit faster but this implementation is more readable
        bs,x_len = x.size(0),x.size(1)
        wq,wk,wv = torch.chunk(self.attention(x), 3, dim=-1)
        wq,wk,wv = map(lambda x:x.view(bs, x.size(1), self.n_heads, self.d_head), (wq,wk,wv))
        attn_score = torch.einsum('bind,bjnd->bijn', (wq, wk))
        if self.scale: attn_score.mul_(1/(self.d_head ** 0.5))
        if mask is not None:
            attn_score = attn_score.float().masked_fill(mask, -float('inf')).type_as(attn_score)
        attn_prob = self.drop_att(F.softmax(attn_score, dim=2))
        attn_vec = torch.einsum('bijn,bjnd->bind', (attn_prob, wv))
        return attn_vec.contiguous().view(bs, x_len, -1)

#def _line_shift1(x:Tensor, mask:bool=False):
#    "Shift the line i of `x` by p-i elements to the left, is `mask` puts 0s on the diagonal."
#    bs,n,p,nh = x.size()
#    x_pad = torch.cat([x.new_zeros(bs,n,1,nh), x], dim=2)
#    x_shift = x_pad.view(bs,p + 1,n,nh)[:,1:].view_as(x)
#    if mask: x_shift.mul_(torch.tril(x.new_ones(n,p), p-n)[None,:,:,None])
#    return x_shift

def _line_shift(x:Tensor, mask:bool=False):
    "Shift the line i of `x` by p-i elements to the left, is `mask` puts 0s on the diagonal."
    bs,nh,n,p = x.size()
    x_pad = torch.cat([x.new_zeros(bs,nh,n,1), x], dim=3)
    x_shift = x_pad.view(bs,nh,p + 1,n)[:,:,1:].view_as(x)
    if mask: x_shift.mul_(torch.tril(x.new_ones(n,p), p-n)[None,None,])
    return x_shift

class MultiHeadRelativeAttention(MultiHeadAttention):
    "MutiHeadAttention with relative positional encoding."

    def __init__(self, n_heads:int, d_model:int, d_head:int, resid_p:float=0., attn_p:float=0., bias:bool=True,
                 scale:bool=True):
        super().__init__(n_heads, d_model, d_head, resid_p=resid_p, attn_p=attn_p, bias=bias, scale=scale)
        self.r_attn = nn.Linear(d_model, n_heads * d_head, bias=bias)

    def _apply_attention(self, x:Tensor, r:Tensor=None, u:Tensor=None, v:Tensor=None, mask:Tensor=None, mem:Tensor=None):
        #Notations from the paper: x input, r vector of relative distance between two elements, u et v learnable
        #parameters of the model common between all layers, mask to avoid cheating and mem the previous hidden states.
        bs,x_len,seq_len = x.size(0),x.size(1),r.size(0)
        context = x if mem is None else torch.cat([mem, x], dim=1)
        wq,wk,wv = torch.chunk(self.attention(context), 3, dim=-1)
        wq = wq[:,-x_len:]
        wq,wk,wv = map(lambda x:x.view(bs, x.size(1), self.n_heads, self.d_head), (wq,wk,wv))
        wq,wk,wv = wq.permute(0, 2, 1, 3),wk.permute(0, 2, 3, 1),wv.permute(0, 2, 1, 3)
        wkr = self.r_attn(r)
        wkr = wkr.view(seq_len, self.n_heads, self.d_head)
        wkr = wkr.permute(1,2,0)
        #### compute attention score (AC is (a) + (c) and BS is (b) + (d) in the paper)
        AC = torch.matmul(wq+u,wk)
        BD = _line_shift(torch.matmul(wq+v, wkr))
        if self.scale: attn_score = (AC + BD).mul_(1/(self.d_head ** 0.5))
        if mask is not None:
            attn_score = attn_score.float().masked_fill(mask, -float('inf')).type_as(attn_score)
        attn_prob = self.drop_att(F.softmax(attn_score, dim=-1))
        attn_vec = torch.matmul(attn_prob, wv)
        return attn_vec.permute(0, 2, 1, 3).contiguous().view(bs, x_len, -1)

    def _attention_einsum(self, x:Tensor, r:Tensor=None, u:Tensor=None, v:Tensor=None, mask:Tensor=None, mem:Tensor=None):
        # Permute and matmul is a little bit faster but this implementation is more readable
        bs,x_len,seq_len = x.size(0),x.size(1),r.size(0)
        context = x if mem is None else torch.cat([mem, x], dim=1)
        wq,wk,wv = torch.chunk(self.attention(context), 3, dim=-1)
        wq = wq[:,-x_len:]
        wkr = self.r_attn(r)
        wq,wk,wv = map(lambda x:x.view(bs, x.size(1), self.n_heads, self.d_head), (wq,wk,wv))
        wkr = wkr.view(seq_len, self.n_heads, self.d_head)
        #### compute attention score (AC is (a) + (c) and BS is (b) + (d) in the paper)
        AC = torch.einsum('bind,bjnd->bijn', (wq+u, wk))
        BD = _line_shift1(torch.einsum('bind,jnd->bijn', (wq+v, wkr)))
        attn_score = (AC + BD).mul_(1/(self.d_head ** 0.5))
        if mask is not None:
            attn_score = attn_score.float().masked_fill(mask, -float('inf')).type_as(attn_score)
        attn_prob = self.drop_att(F.softmax(attn_score, dim=2))
        attn_vec = torch.einsum('bijn,bjnd->bind', (attn_prob, wv))
        return attn_vec.contiguous().view(bs, x_len, -1)

class DecoderLayer(Module):
    "Basic block of a Transformer model."
    #Can't use Sequential directly cause more than one input...
    def __init__(self, n_heads:int, d_model:int, d_head:int, d_inner:int, resid_p:float=0., attn_p:float=0., ff_p:float=0.,
                 bias:bool=True, scale:bool=True, act:Activation=Activation.ReLU, double_drop:bool=True,
                 attn_cls:Callable=MultiHeadAttention):
        self.mhra = attn_cls(n_heads, d_model, d_head, resid_p=resid_p, attn_p=attn_p, bias=bias, scale=scale)
        self.ff   = feed_forward(d_model, d_inner, ff_p=ff_p, act=act, double_drop=double_drop)

    def forward(self, x:Tensor, mask:Tensor=None, **kwargs): return self.ff(self.mhra(x, mask=mask, **kwargs))

class Transformer(Module):
    "Transformer model: https://arxiv.org/abs/1706.03762."
    def __init__(self, vocab_sz:int, ctx_len:int, n_layers:int, n_heads:int, d_model:int, d_head:int, d_inner:int,
                 resid_p:float=0., attn_p:float=0., ff_p:float=0., embed_p:float=0., bias:bool=True, scale:bool=True,
                 act:Activation=Activation.ReLU, double_drop:bool=True, attn_cls:Callable=MultiHeadAttention,
                 learned_pos_enc:bool=True, mask:bool=True):
        self.mask = mask
        self.encoder = nn.Embedding(vocab_sz, d_model)
        self.pos_enc = nn.Embedding(ctx_len, d_model) if learned_pos_enc else PositionalEncoding(d_model)
        self.drop_emb = nn.Dropout(embed_p)
        self.layers = nn.ModuleList([DecoderLayer(n_heads, d_model, d_head, d_inner, resid_p=resid_p, attn_p=attn_p,
                      ff_p=ff_p, bias=bias, scale=scale, act=act, double_drop=double_drop,
                      attn_cls=attn_cls) for k in range(n_layers)])

    def reset(self): pass

    def forward(self, x):
        bs, x_len = x.size()
        pos = torch.arange(0, x_len, device=x.device, dtype=x.dtype)
        inp = self.drop_emb(self.encoder(x) + self.pos_enc(pos)[None]) #.mul_(self.d_model ** 0.5)
        mask = torch.triu(x.new_ones(x_len, x_len), diagonal=1).byte()[None,None] if self.mask else None
        #[None,:,:None] for einsum implementation of attention
        for layer in self.layers: inp = layer(inp, mask=mask)
        return ([inp],[inp]) #For the LinearDecoder

class TransformerXL(Module):
    "TransformerXL model: https://arxiv.org/abs/1901.02860."
    def __init__(self, vocab_sz:int, ctx_len:int, n_layers:int, n_heads:int, d_model:int, d_head:int, d_inner:int,
                 resid_p:float=0., attn_p:float=0., ff_p:float=0., embed_p:float=0., bias:bool=False, scale:bool=True,
                 act:Activation=Activation.ReLU, double_drop:bool=True, attn_cls:Callable=MultiHeadRelativeAttention,
                 learned_pos_enc:bool=False, mask:bool=True, mem_len:int=0):
        self.encoder = nn.Embedding(vocab_sz, d_model)
        self.pos_enc = nn.Embedding(ctx_len, d_model) if learned_pos_enc else PositionalEncoding(d_model)
        self.drop_emb = nn.Dropout(embed_p)
        self.u = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) #Remove 1 for einsum implementation of attention
        self.v = nn.Parameter(torch.Tensor(n_heads, 1, d_head)) #Remove 1 for einsum implementation of attention
        self.mem_len,self.n_layers,self.d_model,self.mask = mem_len,n_layers,d_model,mask
        self.init = False
        self.layers = nn.ModuleList([DecoderLayer(n_heads, d_model, d_head, d_inner, resid_p=resid_p, attn_p=attn_p,
                      ff_p=ff_p, bias=bias, scale=scale, act=act, double_drop=double_drop,
                      attn_cls=attn_cls) for k in range(n_layers)])

    def reset(self):
        "Reset the internal memory."
        self.hidden = [next(self.parameters()).data.new(0) for i in range(self.n_layers+1)]

    def _update_mems(self, hids):
        if not getattr(self, 'hidden', False): return None
        assert len(hids) == len(self.hidden), 'len(hids) != len(self.hidden)'
        with torch.no_grad():
            for i in range(len(hids)):
                cat = torch.cat([self.hidden[i], hids[i]], dim=1)
                self.hidden[i] = cat[:,-self.mem_len:].detach()

    def select_hidden(self, idxs): self.hidden = [h[idxs] for h in self.hidden]

    def forward(self, x):
        #The hidden state has to be initiliazed in the forward pass for nn.DataParallel
        if self.mem_len > 0 and not self.init:
            self.reset()
            self.init = True
        bs,x_len = x.size()
        inp = self.drop_emb(self.encoder(x)) #.mul_(self.d_model ** 0.5)
        m_len = self.hidden[0].size(1) if hasattr(self, 'hidden') and len(self.hidden[0].size()) > 1 else 0
        seq_len = m_len + x_len
        mask = torch.triu(x.new_ones(x_len, seq_len), diagonal=1+m_len).byte()[None,None] if self.mask else None
        #[None,:,:None] for einsum implementation of attention
        hids = []
        pos = torch.arange(seq_len-1, -1, -1, device=inp.device, dtype=inp.dtype)
        pos_enc = self.pos_enc(pos)
        hids.append(inp)
        for i, layer in enumerate(self.layers):
            mem = self.hidden[i] if self.mem_len > 0 else None
            inp = layer(inp, r=pos_enc, u=self.u, v=self.v, mask=mask, mem=mem)
            hids.append(inp)
        core_out = inp[:,-x_len:]
        if self.mem_len > 0 : self._update_mems(hids)
        return (self.hidden if self.mem_len > 0 else [core_out]),[core_out]

def init_transformer(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None: nn.init.normal_(m.weight, 0., 0.02)
        if hasattr(m, 'bias') and m.bias is not None:     nn.init.constant_(m.bias, 0.)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None: nn.init.normal_(m.weight, 1., 0.02)
        if hasattr(m, 'bias') and m.bias is not None:     nn.init.constant_(m.bias, 0.)
    elif classname.find('TransformerXL') != -1:
        if hasattr(m, 'u'): nn.init.normal_(m.u, 0., 0.02)
        if hasattr(m, 'v'): nn.init.normal_(m.v, 0., 0.02)

tfmer_lm_config = dict(ctx_len=512, n_layers=12, n_heads=12, d_model=768, d_head=64, d_inner=3072, resid_p=0.1, attn_p=0.1,
                         ff_p=0.1, embed_p=0.1, output_p=0., bias=True, scale=True, act=Activation.GeLU, double_drop=False,
                         tie_weights=True, out_bias=False, init=init_transformer, mask=True)

tfmer_clas_config = dict(ctx_len=512, n_layers=12, n_heads=12, d_model=768, d_head=64, d_inner=3072, resid_p=0.1, attn_p=0.1,
                         ff_p=0.1, embed_p=0.1, output_p=0., bias=True, scale=True, act=Activation.GeLU, double_drop=False,
                         init=init_transformer, mask=False)

def tfmer_lm_split(model:nn.Module) -> List[nn.Module]:
    "Split a RNN `model` in groups for differential learning rates."
    encoder = model[0]
    n = len(encoder.layers)//3
    groups = [list(encoder.layers[:n]), list(encoder.layers[n:2*n]), list(encoder.layers[2*n:])]
    return groups + [[encoder.encoder, model[1]]]

def tfmer_clas_split(model:nn.Module) -> List[nn.Module]:
    "Split a RNN `model` in groups for differential learning rates."
    encoder = model[0].module
    n = len(encoder.layers)//3
    groups = [[encoder.encoder], list(encoder.layers[:n]), list(encoder.layers[n:2*n]), list(encoder.layers[2*n:])]
    return groups + [[model[1]]]

tfmerXL_lm_config = dict(ctx_len=150, n_layers=12, n_heads=10, d_model=410, d_head=41, d_inner=2100, resid_p=0.1, attn_p=0.1,
                         ff_p=0.1, embed_p=0.1, output_p=0.1, bias=False, scale=True, act=Activation.ReLU, double_drop=True,
                         tie_weights=True, out_bias=True, init=init_transformer, mem_len=150, mask=True)

tfmerXL_clas_config = dict(ctx_len=150, n_layers=12, n_heads=10, d_model=410, d_head=41, d_inner=2100, resid_p=0.1, attn_p=0.1,
                         ff_p=0.1, embed_p=0.1, output_p=0.1, bias=False, scale=True, act=Activation.ReLU, double_drop=True,
                         init=init_transformer, mem_len=150, mask=False)

def tfmerXL_lm_split(model:nn.Module) -> List[nn.Module]:
    "Split a RNN `model` in groups for differential learning rates."
    encoder = model[0]
    n = len(encoder.layers)//3
    groups = [list(encoder.layers[:n]) + [ParameterModule(encoder.u), ParameterModule(encoder.v)]]
    return groups + [list(encoder.layers[n:2*n]), list(encoder.layers[2*n:]), [encoder.encoder, model[1]]]

def tfmerXL_clas_split(model:nn.Module) -> List[nn.Module]:
    "Split a RNN `model` in groups for differential learning rates."
    encoder = model[0].module
    n = len(encoder.layers)//3
    groups = [[encoder.encoder], list(encoder.layers[:n]) + [ParameterModule(encoder.u), ParameterModule(encoder.v)]]
    return groups + [list(encoder.layers[n:2*n]), list(encoder.layers[2*n:]), [model[1]]]
