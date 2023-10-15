import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial
import numpy as np
from typing import List

import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm

def MultiwayWrapper(args, module, dim=1):
    #if args.multiway:
     #   return MultiwayNetwork(module, dim=dim)
    return module


def set_split_position(position):
    def apply_fn(module):
        if hasattr(module, "split_position"):
            module.split_position = position

    return apply_fn


class MultiwayNetwork(nn.Module):
    def __init__(self, module, dim=1):
        super().__init__()
        self.dim = dim
        self.A = module
        self.B = copy.deepcopy(module)
        self.B.reset_parameters()
        self.split_position = -1

    def forward(self, x, **kwargs):
        if self.split_position == -1:
            return self.A(x, **kwargs)
        if self.split_position == 0:
            return self.B(x, **kwargs)
        x1, x2 = torch.split(
            x,
            [self.split_position, x.size(self.dim) - self.split_position],
            dim=self.dim,
        )
        # x1, x2 = x[:self.split_position], x[self.split_position:]
        y1, y2 = self.A(x1, **kwargs), self.B(x2, **kwargs)
        return torch.cat([y1, y2], dim=self.dim)


class MutliwayEmbedding(MultiwayNetwork):
    def __init__(self, modules, dim=1):
        super(MultiwayNetwork, self).__init__()
        self.dim = dim
        assert len(modules) == 2
        self.A = modules[0]
        self.B = modules[1]
        self.split_position = -1

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

def get_activation_fn(activation):
    if activation == "swish":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    else:
        raise NotImplementedError
    
class MultiScaleRetention(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        num_heads,
        value_factor=1,
        gate_fn="swish",
    ):
        super().__init__()
        self.args = args
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        
        self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.q_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.k_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        #self.m_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.v_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim * self.factor, bias=True))
        self.g_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim * self.factor, bias=True))
        
        self.out_proj = MultiwayWrapper(args, nn.Linear(embed_dim * self.factor, embed_dim, bias=True))

        self.group_norm = MultiwayWrapper(args, LayerNorm(self.head_dim, eps=1e-6, elementwise_affine=False))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        #nn.init.xavier_uniform_(self.m_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def parallel_forward(self, qr, kr, v, mask, pair_mask):
        bsz, tgt_len, embed_dim = v.size()

        vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        qk_mat = qr @ kr.transpose(-1, -2) # bsz * m * tgt_len * tgt_len
        qk_mat = qk_mat + pair_mask #) * mask
        # invariant after normalization
        qk_mat = qk_mat / qk_mat.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1)
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2)
        return output

    def recurrent_forward(
        self,
        qr, kr, v,
        decay,
        incremental_state
    ):
        bsz = v.size(0)

        v = v.view(bsz, self.num_heads, self.head_dim, 1)
        kv = kr * v
        if "prev_key_value" in incremental_state:
            prev_kv = incremental_state["prev_key_value"]
            prev_scale = incremental_state["scale"]
            scale = prev_scale * decay + 1
            kv = prev_kv * (prev_scale.sqrt() * decay / scale.sqrt()).view(self.num_heads, 1, 1) + kv / scale.sqrt().view(self.num_heads, 1, 1)
            # kv = prev_kv * decay.view(self.num_heads, 1, 1) + kv
        else:
            scale = torch.ones_like(decay)

        incremental_state["prev_key_value"] = kv
        incremental_state["scale"] = scale

        output = torch.sum(qr * kv, dim=3)
        return output
    
    def chunk_recurrent_forward(
        self,
        qr, kr, v,
        inner_mask
    ):
        mask, cross_decay, inner_decay = inner_mask
        bsz, tgt_len, embed_dim = v.size()
        chunk_len = mask.size(1)
        num_chunks = tgt_len // chunk_len

        assert tgt_len % chunk_len == 0

        qr = qr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        kr = kr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        v = v.view(bsz, num_chunks, chunk_len, self.num_heads, self.head_dim).transpose(2, 3)

        kr_t = kr.transpose(-1, -2)

        qk_mat = qr @ kr_t # bsz * num_heads * chunk_len * chunk_len
        qk_mat = qk_mat * mask
        inner_scale = qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1)
        qk_mat = qk_mat / inner_scale
        inner_output = torch.matmul(qk_mat, v) # bsz * num_heads * num_value_heads * chunk_len * head_dim
        
        # reduce kv in one chunk
        kv = kr_t @ (v * mask[:, -1, :, None])
        kv = kv.view(bsz, num_chunks, self.num_heads, self.key_dim, self.head_dim)

        kv_recurrent = []
        cross_scale = []
        kv_state = torch.zeros(bsz, self.num_heads, self.key_dim, self.head_dim).to(v)
        kv_scale = torch.ones(bsz, self.num_heads, 1, 1).to(v)
        
        # accumulate kv by loop
        for i in range(num_chunks):
            kv_recurrent.append(kv_state / kv_scale)
            cross_scale.append(kv_scale)
            kv_state = kv_state * cross_decay + kv[:, i]
            kv_scale = kv_state.detach().abs().sum(dim=-2, keepdim=True).max(dim=-1, keepdim=True).values.clamp(min=1)
            
        kv_recurrent = torch.stack(kv_recurrent, dim=1)
        cross_scale = torch.stack(cross_scale, dim=1)
        
        all_scale = torch.maximum(inner_scale, cross_scale)
        align_inner_scale = all_scale / inner_scale
        align_cross_scale = all_scale / cross_scale

        cross_output = (qr * inner_decay) @ kv_recurrent
        output = inner_output / align_inner_scale + cross_output / align_cross_scale
        # output = inner_output / cross_scale + cross_output / inner_scale

        output = output.transpose(2, 3)
        return output
    
    def forward(
        self,
        x,
        rel_pos,
        chunkwise_recurrent=False,
        incremental_state=None
    ):
        bsz, tgt_len, _ = x.size()
        (sin, cos), inner_mask = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        m = 1 #self.m_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        k *= self.scaling
        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        #m = m.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)

        #m = torch.matmul(m,m.transpose(2, 3))

        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        if incremental_state is not None:
            output = self.recurrent_forward(qr, kr, v, inner_mask, incremental_state)
        elif chunkwise_recurrent:
            output = self.chunk_recurrent_forward(qr, kr, v, inner_mask)
        else:
            output = self.parallel_forward(q, k, v, m, inner_mask)
        
        output = self.group_norm(output).reshape(bsz, tgt_len, self.head_dim * self.num_heads)

        output = self.gate_fn(g) * output

        output = self.out_proj(output)

        return output
        
class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn,
        dropout,
        activation_dropout,
        layernorm_eps,
        subln=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = get_activation_fn(activation=str(activation_fn))
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
        self.ffn_layernorm = LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        x = self.fc1(x)
        x = self.activation_fn(x).type_as(x)
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = x.view(x_shape)
        x = self.dropout_module(x)
        return x

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

    def __init__(self, d_model, nhead, dropout=0.1, activation="gelu"):
        super(HF_TransformerEncoderLayer, self).__init__()
        #MultiheadAttention
        self.self_attn = MultiScaleRetention(1,d_model, nhead) #nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.dropout = nn.Dropout(dropout)
        self.ffn = FeedForwardNetwork(d_model, d_model*4, activation, dropout, dropout, 1e-6, subln=True)
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.angle = 1.0 / (10000 ** torch.linspace(0, 1, d_model // 8 // 2))
        self.angle = self.angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.decay = torch.log(1 - 2 ** (-5 - torch.arange(8, dtype=torch.float)))
        
        self.alpha = 1
        self.activation = get_activation_fn(activation)        

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
        index = torch.arange(src.shape[1]).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :]).to(src.device)
        cos = torch.cos(index[:, None] * self.angle[None, :]).to(src.device)
        retention_rel_pos = ((sin, cos), mask)
        
        src2 = src
        src2 = self.dropout(self.self_attn(src2,retention_rel_pos))
        src = src*self.alpha + src2
        src = self.norm0(src)
        
        src2 = self.ffn(src)
        src = src*self.alpha + src2
        src = self.norm1(src)
        
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
    
class ParticleRetention(nn.Module):

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
        super(ParticleRetention, self).__init__(**kwargs)

        self.for_inference = for_inference
        self.build_4v = build_4v
        self.num_enc_layers = num_enc
        self.cpf_fts = cpf_dim
        self.npf_fts = npf_dim
        self.vtx_fts = vtx_dim
        self.InputProcess = InputProcess(cpf_dim, npf_dim, vtx_dim, embed_dim)
        self.Linear = nn.Linear(embed_dim, num_classes)

        self.pair_embed = PairEmbed(4, [64,64] + [num_head], for_onnx=for_inference)

        self.EncoderLayer = HF_TransformerEncoderLayer(d_model=embed_dim, nhead=num_head, dropout = 0.1)
        self.Encoder = HF_TransformerEncoder(self.EncoderLayer, num_layers=num_enc)

        self.pooling = AttentionPooling()

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
        attn_mask = self.pair_embed(v)

        enc = self.Encoder(enc, attn_mask, padding_mask)

        x = self.pooling(enc)
        output = self.Linear(x)

        if self.for_inference:
            output = torch.softmax(output, dim=1)

        return output
