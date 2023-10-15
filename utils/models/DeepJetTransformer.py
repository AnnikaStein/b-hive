import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial
import numpy as np
from typing import List

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
        self.cpf_conv3 = InputConv(embed_dim*1,embed_dim)

        self.npf_bn0 = torch.nn.BatchNorm1d(npf_dim, eps = 0.001, momentum = 0.1)
        self.npf_conv1 = InputConv(npf_dim,embed_dim)
        self.npf_conv3 = InputConv(embed_dim*1,embed_dim)

        self.vtx_bn0 = torch.nn.BatchNorm1d(vtx_dim, eps = 0.001, momentum = 0.1)
        self.vtx_conv1 = InputConv(vtx_dim,embed_dim)
        self.vtx_conv3 = InputConv(embed_dim*1,embed_dim)


    def forward(self, cpf, npf, vtx):
                
        cpf = self.cpf_bn0(torch.transpose(cpf, 1, 2))
        cpf = self.cpf_conv1(cpf, cpf, skip = False)
        cpf = self.cpf_conv3(cpf, cpf, skip = False)

        npf = self.npf_bn0(torch.transpose(npf, 1, 2))
        npf = self.npf_conv1(npf, npf, skip = False)
        npf = self.npf_conv3(npf, npf, skip = False)

        vtx = self.vtx_bn0(torch.transpose(vtx, 1, 2))
        vtx = self.vtx_conv1(vtx, vtx, skip = False)
        vtx = self.vtx_conv3(vtx, vtx, skip = False)

        out = torch.cat((cpf,npf,vtx), dim = 2)
        out = torch.transpose(out, 1, 2)
        
        return out
    
class DenseClassifier(nn.Module):

    def __init__(self, dim = 128, **kwargs):
        super(DenseClassifier, self).__init__(**kwargs)
             
        self.LinLayer1 = LinLayer(128,128)

    def forward(self, x):
        
        x = self.LinLayer1(x, x, skip = True)
        
        return x
    
class AttentionPooling(nn.Module):

    def __init__(self, dim = 128, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)

        self.ConvLayer = torch.nn.Conv1d(dim, 1, kernel_size=1)
        self.Softmax = nn.Softmax(dim=-1)
        self.bn = torch.nn.BatchNorm1d(dim, eps = 0.001, momentum = 0.1)
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

        self.activation = nn.GELU()

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = nn.GELU()
        super(HF_TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, padding_mask):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.norm0(src)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            src2 = self.self_attn(src2,src2,src2)[0] #, key_padding_mask = padding_mask)[0]
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

    def forward(self, src, padding_mask):
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src
        padding_mask = padding_mask

        for mod in self.layers:
            output = mod(output, padding_mask)

        return output
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.ReLU()

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class DeepJetTransformer(nn.Module):

    def __init__(self,
                 feature_edges,
                 num_classes = 6,
                 num_enc = 3,
                 num_head = 8,
                 embed_dim = 128,
                 cpf_dim = 16,
                 npf_dim = 6,
                 vtx_dim = 12,
                 for_inference = False,
                 **kwargs):
        super(DeepJetTransformer, self).__init__(**kwargs)
        
        self.feature_edges = torch.Tensor(feature_edges).int()
        self.InputProcess = InputProcess(cpf_dim, npf_dim, vtx_dim, embed_dim)
        self.Linear = nn.Linear(embed_dim, num_classes)
        self.DenseClassifier = DenseClassifier()
        self.pooling = AttentionPooling()
        self.for_inference = for_inference

        self.EncoderLayer = HF_TransformerEncoderLayer(d_model=embed_dim, nhead=num_head, dropout = 0.1)
        self.Encoder = HF_TransformerEncoder(self.EncoderLayer, num_layers=num_enc)

    def forward(self, x):

        _, cpf, npf, vtx, cpf_4v, npf_4v, vtx_4v = x[0],x[1],x[2],x[3],x[4],x[5],x[6]
       
        padding_mask = torch.cat((cpf[:,:,:1],npf[:,:,:1],vtx[:,:,:1]), dim = 1)
        padding_mask = torch.eq(padding_mask[:,:,0], 0.0)
        enc = self.InputProcess(cpf, npf, vtx)

        enc = self.Encoder(enc, padding_mask)
        enc = self.pooling(enc)
        
        x = self.DenseClassifier(enc)
        output = self.Linear(x)
        
        if self.for_inference:
            output = torch.softmax(output, dim=1)

        return output
