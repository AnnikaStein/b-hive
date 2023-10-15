import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial
import numpy as np
from typing import List

def node_distance(x):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance

@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi

@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2

def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2

def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2

def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part

def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x[:,:4,:].split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    rapidity = pz
    phi = (atan2 if for_onnx else torch.atan2)(py, px)

    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)

def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps)**(-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v

def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)

def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8, for_onnx=False):
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    
    ai = torch.ne(pti, 0.0).float()
    aj = torch.ne(ptj, 0.0).float()
    mask = ai*aj

    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps)+1)
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin = ((pti <= ptj) * pti + (pti > ptj) * ptj) if for_onnx else torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps)+1)
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps)+1)
        outputs = [lnkt, lnz, lndelta]

    if num_outputs > 3:
        xij = xi + xj
        lnm2 = torch.log(to_m2(xij, eps=eps)+1)
        outputs.append(lnm2)

    if num_outputs > 6:
        ei, ej = xi[:, 3:4], xj[:, 3:4]
        emin = ((ei <= ej) * ei + (ei > ej) * ej) if for_onnx else torch.minimum(ei, ej)
        lnet = torch.log((emin * delta).clamp(min=eps))
        lnze = torch.log((emin / (ei + ej).clamp(min=eps)).clamp(min=eps))
        outputs += [lnet, lnze]

    if num_outputs > 8:
        costheta = (p3_norm(xi, eps=eps) * p3_norm(xj, eps=eps)).sum(dim=1, keepdim=True)
        sintheta = (1 - costheta**2).clamp(min=0, max=1).sqrt()
        outputs += [costheta, sintheta]

    assert(len(outputs) == num_outputs)
    o = torch.cat(outputs, dim=1)*mask
    return o

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # From https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation='gelu'):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend([
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)
    def forward(self, x):
        if self.input_bn is not None:
            x = self.input_bn(x)
            x = x.permute(2, 0, 1).contiguous()
        return self.embed(x)
    
def tril_indices(x, seq_len, offset = True):
    if offset:
        a, b = [], []
        for i in range(seq_len):
            for j in range(i):
                a.append(i)
                b.append(j)
    else:
        a, b = [], []
        for i in range(seq_len):
            for j in range(i+1):
                a.append(i)
                b.append(j)
    i = torch.tensor(a)
    j = torch.tensor(b)
    
    return i, j

def tril_indicesNEW(rows, cols, offset=0):
    return torch.ones(rows, cols).tril(offset).nonzero().t()

class PairEmbed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation='gelu', eps=1e-8, for_onnx=False):
        super().__init__()

        self.for_onnx = for_onnx
        self.pairwise_lv_fts = partial(pairwise_lv_fts, num_outputs=4, eps=eps, for_onnx=for_onnx)

        module_list = []
        for dim in dims:
            module_list.extend([
                nn.BatchNorm1d(input_dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Conv1d(input_dim, dim, 1),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

        self.out_dim = dims[-1]

    def forward(self, x):

        batch_size, _, seq_len = x.size()
        if not self.for_onnx:
            i, j = torch.tril_indices(seq_len, seq_len, offset = -1, device=x.device)
            x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
            xi = x[:, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2)
            xj = x[:, :, j, i]
            x = self.pairwise_lv_fts(xi, xj)
        else:
            i, j = tril_indicesNEW(seq_len, seq_len, offset = -1) # old
            x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
            xi = x[:, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2)
            xj = x[:, :, j, i]
            x = self.pairwise_lv_fts(xi, xj)
        elements = self.embed(x)  # (batch, embed_dim, num_elements
        
        if not self.for_onnx:
            y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=x.device)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=x.device)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        return y

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.cuda.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

class InputConv(nn.Module):

    def __init__(self, in_chn, out_chn, dropout_rate = 0.1, **kwargs):
        super(InputConv, self).__init__(**kwargs)
        
        self.lin = torch.nn.Conv1d(in_chn, out_chn, kernel_size=1)
        self.bn1 = torch.nn.BatchNorm1d(out_chn, eps = 0.001, momentum = 0.1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sc, skip: bool = True):
        
        x2 = self.dropout(self.bn1(self.act(self.lin(x))))
        if skip:
            x = sc + x2
        else:
            x = x2
        return x
    
class LinLayer(nn.Module):

    def __init__(self, in_chn, out_chn, dropout_rate = 0.1, **kwargs):
        super(LinLayer, self).__init__(**kwargs)
        
        self.lin = torch.nn.Linear(in_chn, out_chn)
        self.bn1 = torch.nn.BatchNorm1d(out_chn, eps = 0.001, momentum = 0.1)
        self.bn2 = torch.nn.BatchNorm1d(out_chn, eps = 0.001, momentum = 0.1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sc, skip = True):
        
        x2 = self.dropout(self.bn1(self.act(self.lin(x))))
        if skip:
            x = self.bn2(sc + x2)
        else:
            x = self.bn2(x2)
        return x

class LinLayer2(nn.Module):

    def __init__(self, in_chn, out_chn, dropout_rate = 0.1, **kwargs):
        super(LinLayer2, self).__init__(**kwargs)

        self.lin = torch.nn.Linear(in_chn, out_chn)
        self.ln = torch.nn.LayerNorm(out_chn, eps = 0.001)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        x = self.dropout(self.ln(self.act(self.lin(x))))
        return x

class InputProcess(nn.Module):

    def __init__(self, cpf_dim, npf_dim, vtx_dim, embed_dim, **kwargs):
        super(InputProcess, self).__init__(**kwargs)
        
        self.cpf_bn0 = torch.nn.BatchNorm1d(cpf_dim, eps = 0.001, momentum = 0.1)
        self.cpf_conv1 = InputConv(cpf_dim,embed_dim)
#        self.cpf_conv2 = InputConv(embed_dim,embed_dim*4)
        self.cpf_conv3 = InputConv(embed_dim*1,embed_dim)

        self.npf_bn0 = torch.nn.BatchNorm1d(npf_dim, eps = 0.001, momentum = 0.1)
        self.npf_conv1 = InputConv(npf_dim,embed_dim)
 #       self.npf_conv2 = InputConv(embed_dim,embed_dim*4)
        self.npf_conv3 = InputConv(embed_dim*1,embed_dim)

        self.vtx_bn0 = torch.nn.BatchNorm1d(vtx_dim, eps = 0.001, momentum = 0.1)
        self.vtx_conv1 = InputConv(vtx_dim,embed_dim)
  #      self.vtx_conv2 = InputConv(embed_dim,embed_dim*4)
        self.vtx_conv3 = InputConv(embed_dim*1,embed_dim)

#        self.meta_conv = InputConv(8*16,8*16)

    def forward(self, cpf, npf, vtx):
                
        cpf = self.cpf_bn0(torch.transpose(cpf, 1, 2))
        cpf = self.cpf_conv1(cpf, cpf, skip = False)
#        cpf = self.cpf_conv2(cpf, cpf, skip = False)
        cpf = self.cpf_conv3(cpf, cpf, skip = False)

        npf = self.npf_bn0(torch.transpose(npf, 1, 2))
        npf = self.npf_conv1(npf, npf, skip = False)
 #       npf = self.npf_conv2(npf, npf, skip = False)
        npf = self.npf_conv3(npf, npf, skip = False)

        vtx = self.vtx_bn0(torch.transpose(vtx, 1, 2))
        vtx = self.vtx_conv1(vtx, vtx, skip = False)
  #      vtx = self.vtx_conv2(vtx, vtx, skip = False)
        vtx = self.vtx_conv3(vtx, vtx, skip = False)

        out = torch.cat((cpf,npf,vtx), dim = 2)
        out = torch.transpose(out, 1, 2)
        
        return out
    
class DenseClassifier(nn.Module):

    def __init__(self, **kwargs):
        super(DenseClassifier, self).__init__(**kwargs)
             
        self.LinLayer1 = LinLayer(128,128)

    def forward(self, x):
        
        x = self.LinLayer1(x, x, skip = True)
        
        return x
    
class AttentionPooling(nn.Module):

    def __init__(self, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)

        self.ConvLayer = torch.nn.Conv1d(128, 1, kernel_size=1)
        self.Softmax = nn.Softmax(dim=-1)
        self.bn = torch.nn.BatchNorm1d(128, eps = 0.001, momentum = 0.1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        
        a = self.ConvLayer(torch.transpose(x, 1, 2))
        a = self.Softmax(a)
        
        y = torch.matmul(a,x)
        y = torch.squeeze(y, dim = 1)
        y = self.dropout(self.bn(self.act(y)))
        
        return y
    
class HF_TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
       >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dropout=0.1, activation="relu"):
        super(HF_TransformerEncoderLayer, self).__init__()
        #MultiheadAttention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first = True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_model*4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model*4, d_model)

        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model*4)
        self.dropout0 = nn.Dropout(dropout)

        self.activation = nn.GELU() #_get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = nn.GELU()
        super(HF_TransformerEncoderLayer, self).__setstate__(state)

    def forward(self,src,mask,padding_mask):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.norm0(src)
        merged_mask = self.self_attn.merge_masks(mask, padding_mask, src2)[0]
        src2 = self.self_attn(src2,src2,src2,attn_mask = merged_mask.reshape(-1, merged_mask.shape[2], merged_mask.shape[2]))[0]
        src = src + src2
        src = self.norm1(src)
        
        src2 = self.dropout0(self.linear2(self.norm2(self.activation(self.linear1(src)))))
        src = src + src2
        return src
    
class HF_TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers):
        super(HF_TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self,src, mask, padding_mask):
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src
        mask = mask
        padding_mask = padding_mask

        for mod in self.layers:
            output = mod(output, mask, padding_mask)

        return output
    
class CLS_TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dropout=0.1, activation="relu"):
        super(CLS_TransformerEncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first = True)

        self.linear1 = nn.Linear(d_model, d_model*4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model*4, d_model)

        self.norm0a = nn.LayerNorm(d_model)
        self.norm0b = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model*4)
        self.dropout0 = nn.Dropout(dropout)

        self.activation = nn.GELU() #_get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = nn.GELU()
        super(CLS_TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, cls_token, x, padding_mask):
        r"""Pass the input through the encoder layer.                                                
        Args:                                                                                        
            src: the sequence to the encoder layer (required).                                       
            src_mask: the mask for the src sequence (optional).                                      
            src_key_padding_mask: the mask for the src keys per batch (optional).                    
        Shape:                                                                                       
            see the docs in Transformer class.                                                       
        """
        src = torch.cat((cls_token, x), dim = 1)
        padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)

        enc2 = self.norm0a(cls_token)
        src2 = self.norm0b(src)
        src2 = self.self_attn(enc2, src2, src2, key_padding_mask = padding_mask)[0]
        src = cls_token + src2
        src = self.norm1(src)

        src2 = self.dropout0(self.linear2(self.norm2(self.activation(self.linear1(src)))))
        src = src + src2
        return src
    
class CLS_TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers):
        super(CLS_TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self,cls_token, src):
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = cls_token
        mask = src

        for mod in self.layers:
            output = mod(output, mask)

        return output
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.ReLU()

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

def build_E_p(tensor, is_cpf = False): #pt, eta, phi, e (, coords)
    out = torch.zeros(tensor.shape[0], tensor.shape[1], 4, device = tensor.device)
    out[:,:,0] = tensor[:,:,0]*torch.cos(tensor[:,:,2]) #Get px
    out[:,:,1] = tensor[:,:,0]*torch.sin(tensor[:,:,2]) #Get py
    out[:,:,2] = tensor[:,:,0]*(0.5*(torch.exp(tensor[:,:,1]) - torch.exp(-tensor[:,:,1]))) #torch.sinh(tensor[:,:,1]) #Get pz
    out[:,:,3] = tensor[:,:,3] #Get E
    if is_cpf == True:
        out[:,:,4:] = tensor[:,:,4:]

    return out

def get_mass(x, eps=1e-8):
    m2 = x[:, :, 3:4].square() - x[:, :, :3].square().sum(dim=2, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return torch.sqrt(m2)
    
class ParticleTransformer(nn.Module):

    def __init__(self,
                 num_classes = 6,
                 num_enc = 8,
                 num_head = 8,
                 embed_dim = 128,
                 cpf_dim = 17,
                 npf_dim = 8,
                 vtx_dim = 12,
                 for_inference = False,
                 build_4v = True,
                 feature_edges = None,
                 **kwargs):
        super(ParticleTransformer, self).__init__(**kwargs)

        #self.feature_edges = torch.Tensor(feature_edges).int()
        self.for_inference = for_inference
        self.build_4v = build_4v
        self.num_enc_layers = num_enc
        self.cpf_fts = cpf_dim
        self.npf_fts = npf_dim
        self.vtx_fts = vtx_dim
        self.InputProcess = InputProcess(cpf_dim, npf_dim, vtx_dim, embed_dim)
        self.Linear = nn.Linear(embed_dim, num_classes)

        self.pair_embed = PairEmbed(4, [64,64] + [num_head], for_onnx=for_inference)
        self.cls_norm = torch.nn.LayerNorm(embed_dim)

        self.EncoderLayer = HF_TransformerEncoderLayer(d_model=embed_dim, nhead=num_head, dropout = 0.1)
        self.Encoder = HF_TransformerEncoder(self.EncoderLayer, num_layers=num_enc)

        self.CLS_EncoderLayer1 = CLS_TransformerEncoderLayer(d_model=embed_dim, nhead=num_head, dropout = 0.1)
        if(self.num_enc_layers > 3):
            self.CLS_EncoderLayer2 = CLS_TransformerEncoderLayer(d_model=embed_dim, nhead=num_head, dropout = 0.1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):

        cpf, npf, vtx, cpf_4v, npf_4v, vtx_4v = x[0],x[1],x[2],x[3],x[4],x[5]
        
        padding_mask = torch.cat((cpf_4v[:,:,:1],npf_4v[:,:,:1],vtx_4v[:,:,:1]), dim = 1)
        padding_mask = torch.eq(padding_mask[:,:,0], 0.0)

        if self.build_4v:
            cpf_4v = build_E_p(cpf_4v)
            npf_4v = build_E_p(npf_4v)
            vtx_4v = build_E_p(vtx_4v)

        cpf = cpf[:,:,:self.cpf_fts]
        npf = npf[:,:,:self.npf_fts]
        vtx = vtx[:,:,:self.vtx_fts]
            
        enc = self.InputProcess(cpf, npf, vtx)

        lorentz_vectors = torch.cat((cpf_4v,npf_4v,vtx_4v), dim = 1)
        v = lorentz_vectors.transpose(1, 2)
        attn_mask = self.pair_embed(v).view(-1, v.size(-1), v.size(-1))

        enc = self.Encoder(enc, attn_mask, padding_mask)

        cls_tokens = self.cls_token.expand(enc.size(0), 1, -1)
        cls_tokens = self.CLS_EncoderLayer1(cls_tokens, enc, padding_mask)
        if(self.num_enc_layers > 3):
            cls_tokens = self.CLS_EncoderLayer2(cls_tokens, enc, padding_mask)

        x = torch.squeeze(cls_tokens, dim = 1)
        output = self.Linear(self.cls_norm(x))

        if self.for_inference:
            output = torch.softmax(output, dim=1)

        return output
