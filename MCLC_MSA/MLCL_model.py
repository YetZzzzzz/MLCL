import logging
from typing import Optional, Tuple
from modules.transformer import TransformerEncoder
import torch
import torch.nn as nn
import torch.utils.checkpoint

from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange, repeat
from einops.layers.torch import Reduce
from torch.nn import L1Loss, MSELoss
from torch.autograd import Function
from math import pi, log
from functools import wraps
from torch import nn, einsum
import torch.nn.functional as F
import os

from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers.activations import gelu, gelu_new
from transformers import BertConfig
import numpy as np 

import torch.optim as optim
from itertools import chain

from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
DEVICE = torch.device("cuda:2")

# MOSI SETTING
#ACOUSTIC_DIM = 74
#VISUAL_DIM = 47
#TEXT_DIM = 768
# MOSI SETTING
ACOUSTIC_DIM = 74
VISUAL_DIM = 35
TEXT_DIM = 768
logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"




BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-base-cased",
]


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))
BertLayerNorm = torch.nn.LayerNorm

ACT2FN = {
    "gelu": gelu,
    "relu": torch.nn.functional.relu,
    "gelu_new": gelu_new,
    "mish": mish,
}

class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.# not

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor. #
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor. # may need more 
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128 # 
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size) # 
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys) #
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True) # 

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations    
            negative_logits = query @ transpose(negative_keys) # (N,D) * (D,N)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)# (N,1,D)
            negative_logits = query @ transpose(negative_keys) # (N,1,D) * (N,D,M)
            negative_logits = negative_logits.squeeze(1) # 

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key) # cosine similarity

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)# [0,1,2,...,len(query)-1] 

    return F.cross_entropy(logits / temperature, labels, reduction=reduction) # 


def transpose(x):
    return x.transpose(-2, -1) 

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs] # 


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


 # perceiver pre-layernorm       
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)
        
# perceiver activation class        
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

# perceiver ff
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):# mult
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# perceiver cross-vit attention
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 2, dim_head = 64, dropout = 0.):#
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)



## perceiver cross-transformer

class CrossTransformer(nn.Module):
    def __init__(self, latent_dim, input_dim, depth, heads, dim_head, dropout=0.):
        super().__init__()
        self.cross_heads = heads
        self.cross_dim_head = dim_head
        self.depth = depth
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            self.layers.append(nn.ModuleList([
                PreNorm(latent_dim, Attention(latent_dim, input_dim, self.cross_heads, self.cross_dim_head, dropout = dropout), context_dim = input_dim),# attn (x1,x2), FC, 
                PreNorm(latent_dim, FeedForward(latent_dim, dropout = dropout))
            ]))
        self.to_embedds = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim))
    def forward(self, latent_tokens, input_tokens,mask = None):
        latent_tokens = rearrange(latent_tokens, 'b ... d -> b (...) d')
        input_tokens = rearrange(input_tokens, 'b ... d -> b (...) d')
        for cross_attn, cross_ff in self.layers:
            latent_tokens = cross_attn(latent_tokens, context = input_tokens, mask = mask) + latent_tokens
            latent_tokens = cross_ff(latent_tokens) + latent_tokens
        return self.to_embedds(latent_tokens)



class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=3):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class MLCL_BertModel(BertPreTrainedModel):
    def __init__(self, config, multimodal_config, d_l):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.d_l = d_l
        self.proj_l = nn.Conv1d(TEXT_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        # visual,
        # acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        fused_embedding = embedding_output

        encoder_outputs = self.encoder(
            fused_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
       # print('sequence_output_shape', sequence_output.shape)

        outputs = sequence_output.transpose(1, 2)
        outputs = self.proj_l(outputs)
        pooled_output = outputs[:, :, -1]

        return pooled_output



class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=3):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum
        

class MLCL(BertPreTrainedModel):
    def __init__(self, config, multimodal_config, args = None):
        super().__init__(config)
        self.num_labels = config.num_labels# here is 1

        self.d_l = args.d_l
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma

        self.bert = MLCL_BertModel(config, multimodal_config, self.d_l)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.infonce = InfoNCE()
        self.activation = nn.ReLU()
        self.rdrop_rate = args.rdrop 
        self.ctdepths = args.ct_depths

        # private mapping to catch the modality-specific and task-relevant information
        self.private_t1 = nn.Sequential()
        self.private_t1.add_module('private_t_1', nn.Linear(in_features=self.d_l, out_features=self.d_l))
        self.private_t1.add_module('private_t_1_dropout', nn.Dropout(self.rdrop_rate))
        self.private_t1.add_module('private_t_activation_1', nn.ReLU())
        
        self.private_v1 = nn.Sequential()
        self.private_v1.add_module('private_v_1', nn.Linear(in_features=self.d_l, out_features=self.d_l))
        self.private_v1.add_module('private_v_1_dropout', nn.Dropout(self.rdrop_rate))
        self.private_v1.add_module('private_v_activation_1', nn.ReLU())
        
        self.private_a1 = nn.Sequential()
        self.private_a1.add_module('private_a_1', nn.Linear(in_features=self.d_l, out_features=self.d_l))
        self.private_a1.add_module('private_a_1_dropout', nn.Dropout(self.rdrop_rate))
        self.private_a1.add_module('private_a_activation_1', nn.ReLU())
        
        self.private_t2 = nn.Sequential()
        self.private_t2.add_module('private_t_2', nn.Linear(in_features=self.d_l, out_features=self.d_l))
        self.private_t2.add_module('private_t_2_dropout', nn.Dropout(self.rdrop_rate))
        self.private_t2.add_module('private_t_activation_2', nn.ReLU())
        
        self.private_v2 = nn.Sequential()
        self.private_v2.add_module('private_v_2', nn.Linear(in_features=self.d_l, out_features=self.d_l))
        self.private_v2.add_module('private_v_2_dropout', nn.Dropout(self.rdrop_rate))
        self.private_v2.add_module('private_v_activation_2', nn.ReLU())
        
        self.private_a2 = nn.Sequential()
        self.private_a2.add_module('private_a_2', nn.Linear(in_features=self.d_l, out_features=self.d_l))
        self.private_a2.add_module('private_a_2_dropout', nn.Dropout(self.rdrop_rate))
        self.private_a2.add_module('private_a_activation_2', nn.ReLU())

        self.attn_dropout = args.attn_dropout  
        self.num_heads = args.num_heads
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.embed_dropout = args.embed_dropout
        self.clip = args.clip

        self.proj_a = nn.Conv1d(ACOUSTIC_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False) # acoustic
        self.proj_v = nn.Conv1d(VISUAL_DIM, self.d_l, kernel_size=3, stride=1, padding=1, bias=False) # visual
        
        self.transa = self.get_network(self_type='a', layers=args.layers) ###3
        self.transv = self.get_network(self_type='v', layers=args.layers)  ###3
                
        self.cross_ta = CrossTransformer(
            latent_dim = self.d_l,
            input_dim = self.d_l,
            depth = self.ctdepths,# 2
            heads = self.num_heads,
            dim_head = 64,
            dropout = self.attn_dropout)
        self.cross_tv = CrossTransformer(
            latent_dim = self.d_l,
            input_dim = self.d_l,
            depth = self.ctdepths,# 2
            heads = self.num_heads,
            dim_head = 64,
            dropout = self.attn_dropout)
        self.cross_av = CrossTransformer(
            latent_dim = self.d_l,
            input_dim = self.d_l,
            depth = self.ctdepths,# 2
            heads = self.num_heads,
            dim_head = 64,
            dropout = self.attn_dropout)
        
        self.cross_at = CrossTransformer(
            latent_dim = self.d_l,
            input_dim = self.d_l,
            depth = self.ctdepths,# 2
            heads = self.num_heads,
            dim_head = 64,
            dropout = self.attn_dropout)
        self.cross_vt = CrossTransformer(
            latent_dim = self.d_l,
            input_dim = self.d_l,
            depth = self.ctdepths,# 2
            heads = self.num_heads,
            dim_head = 64,
            dropout = self.attn_dropout)
        self.cross_va = CrossTransformer(
            latent_dim = self.d_l,
            input_dim = self.d_l,
            depth = self.ctdepths,# 2
            heads = self.num_heads,
            dim_head = 64,
            dropout = self.attn_dropout)
            
            
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_l, nhead=self.num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2) # num_layers 
        # multimodal fusion classification 
        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.d_l*6, out_features=self.d_l*3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(self.attn_dropout))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.d_l*3, out_features= self.num_labels))
        
        self.lr = args.learning_rate     #main task learning rate
        self.model_all_parameters = chain(self.bert.parameters(), self.transa.parameters(), self.transv.parameters(), self.proj_a.parameters(), self.proj_v.parameters(), self.private_t1.parameters(), self.private_v1.parameters(), self.private_a1.parameters(), self.private_t2.parameters(), self.private_v2.parameters(), self.private_a2.parameters(), self.cross_ta.parameters(), self.cross_tv.parameters(), self.cross_av.parameters(), self.cross_at.parameters(), self.cross_vt.parameters(), self.cross_va.parameters(), self.transformer_encoder.parameters(), self.fusion.parameters())
        
        self.optimizer_all = getattr(optim, 'Adam')(self.model_all_parameters, lr=self.lr)
        
        self.loss = L1Loss() # for classification task, use the CrossEntropyLoss 
        self.uncertainloss = AutomaticWeightedLoss(10)
        self.init_weights()       

    def get_network(self, self_type='l', layers=5):
        if self_type in ['a', 'v']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads= self.num_heads,
                                  layers=layers,
                                  attn_dropout= attn_dropout,
                                  relu_dropout=self.relu_dropout,   
                                  res_dropout= self.res_dropout,    
                                  embed_dropout=self.embed_dropout,  
                                  attn_mask= False)


    def compute_kl_loss(self, p, q):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

        p_loss = p_loss.mean()
        q_loss = q_loss.mean()
        loss = (p_loss + q_loss) / 2
        return loss


    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        label_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
  
        outputs_t = outputs

        acoustic = acoustic.transpose(1, 2)
        visual = visual.transpose(1, 2)

        acoustic = self.proj_a(acoustic)
        visual = self.proj_v(visual)

        acoustic = acoustic.permute(2, 0, 1)
        visual = visual.permute(2, 0, 1)

        outputa = self.transa(acoustic)
        outputv = self.transv(visual)
        outputs_a = outputa[-1]  
        outputs_v = outputv[-1]
        
        outputs_t1 = self.private_t1(outputs_t)
        outputs_a1 = self.private_a1(outputs_a)
        outputs_v1 = self.private_v1(outputs_v)
        
        outputs_t2 = self.private_t2(outputs_t)
        outputs_a2 = self.private_a2(outputs_a)
        outputs_v2 = self.private_v2(outputs_v)


        infonce_t = self.infonce(outputs_t1, outputs_t2)
        infonce_a = self.infonce(outputs_a1, outputs_a2)
        infonce_v = self.infonce(outputs_v1, outputs_v2)
        contra_loss0 = infonce_t + infonce_a + infonce_v

        ################### Contrastive_learning###############################
        cross_ta = self.cross_ta(outputs_t1, outputs_a1)  # [batch_size, hidden_states]
        cross_tv = self.cross_tv(outputs_t1, outputs_v1)   #
        cross_av = self.cross_av(outputs_a1, outputs_v1) 
        
        cross_at = self.cross_at(outputs_a1, outputs_t1)  # [batch_size, hidden_states]
        cross_vt = self.cross_av(outputs_v1, outputs_t1)   #
        cross_va = self.cross_va(outputs_v1, outputs_a1)
        
        infonce_ta = self.infonce(outputs_v1, cross_ta)
        infonce_tv = self.infonce(outputs_a1, cross_tv)
        infonce_av = self.infonce(outputs_t1, cross_av)
        
        infonce_tat = self.infonce(cross_ta, cross_at)
        infonce_tvt = self.infonce(cross_tv, cross_vt)
        infonce_ava = self.infonce(cross_av, cross_va)

        contra_loss1 = infonce_ta + infonce_tv + infonce_av
        contra_loss2 = infonce_tat + infonce_tvt + infonce_ava
        # print('contra_loss / 10.0', contra_loss)
        
        ################### fusion##############################################
        h = torch.stack((outputs_t1, outputs_a1, outputs_v1, cross_ta, cross_tv, cross_av), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        output = self.fusion(h)
        
        self.optimizer_all.zero_grad()
        loss_all = self.loss(output.view(-1), label_ids.view(-1)) 
        loss_sum = loss_all + contra_loss0 / 3 * self.alpha + contra_loss1 / 3 * self.beta + contra_loss2 / 3 * self.gamma # 
        
        loss_sum.backward() #
        self.optimizer_all.step()

        return output,h



    def test(self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,):

        outputs_t = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,)


        acoustic = acoustic.transpose(1, 2)
        visual = visual.transpose(1, 2)

        acoustic = self.proj_a(acoustic)
        visual = self.proj_v(visual)

        acoustic = acoustic.permute(2, 0, 1)
        visual = visual.permute(2, 0, 1)

        outputa = self.transa(acoustic)
        outputv = self.transv(visual)
        outputs_a = outputa[-1]  # 48 50
        outputs_v = outputv[-1]
        
        outputs_t1 = self.private_t1(outputs_t)
        outputs_a1 = self.private_a1(outputs_a)
        outputs_v1 = self.private_v1(outputs_v)
        
        cross_ta = self.cross_ta(outputs_t1, outputs_a1)  # [batch_size, hidden_states]
        cross_tv = self.cross_tv(outputs_t1, outputs_v1)   #
        cross_av = self.cross_av(outputs_a1, outputs_v1)
        
        h = torch.stack((outputs_t1, outputs_a1, outputs_v1, cross_ta, cross_tv, cross_av), dim=0)
        #print('h0',h0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        output = self.fusion(h)

        return output, h
  
