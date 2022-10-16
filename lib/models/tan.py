from lib.core.config import config
import lib.models.frame_modules as frame_modules
import lib.models.prop_modules as prop_modules
import lib.models.map_modules as map_modules
import lib.models.fusion_modules as fusion_modules 
from  transformers import RobertaModel, RobertaTokenizerFast
from typing import List, Optional
from torch import Tensor, nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
import torch


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        # self.proj2 = nn.Linear(dim*4, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask):
        B, N, C = x.shape
        # for einops
        # qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        #masking
        mask = mask[:, None, None, :]
        if mask != None :
            maskVal = mask.type_as(attn).masked_fill(mask==0, -100000.0) - 1
        
    
            # [B, num_heads, query_seq_len, key_seq_len] + [B, 1, 1, key_seq_len] <----- broadCasting
            attn += maskVal
            
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj2(x)
        x = self.proj_drop(x)
        return x
    
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, attn_drop=dropout, proj_drop=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):

        # q = k = src = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            src, src_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    
class TAN(nn.Module):
    def __init__(self):
        super(TAN, self).__init__()

        self.frame_layer = getattr(frame_modules, config.TAN.FRAME_MODULE.NAME)(config.TAN.FRAME_MODULE.PARAMS)
        self.frame_norm = nn.LayerNorm(config.TAN.FRAME_MODULE.PARAMS.HIDDEN_SIZE)
        self.prop_layer = getattr(prop_modules, config.TAN.PROP_MODULE.NAME)(config.TAN.PROP_MODULE.PARAMS)
        self.fusion_layer = getattr(fusion_modules, config.TAN.FUSION_MODULE.NAME)(config.TAN.FUSION_MODULE.PARAMS)
        self.map_layer = getattr(map_modules, config.TAN.MAP_MODULE.NAME)(config.TAN.MAP_MODULE.PARAMS)
        self.pred_layer = nn.Conv2d(config.TAN.PRED_INPUT_SIZE, 1, 1, 1)
       
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.text_embed_layer = RobertaModel.from_pretrained('roberta-base')
        self.text_embed_down = nn.Linear(768, config.TAN.FRAME_MODULE.PARAMS.HIDDEN_SIZE)
        self.text_embed_norm = nn.LayerNorm(config.TAN.FRAME_MODULE.PARAMS.HIDDEN_SIZE)
        
        
        self.transformer = TransformerEncoderLayer(config.TAN.FRAME_MODULE.PARAMS.HIDDEN_SIZE, 8)
        
        self.positional_sum = Summer(PositionalEncoding1D(config.TAN.FRAME_MODULE.PARAMS.HIDDEN_SIZE))

    def forward(self, textual_input, textual_mask, visual_input, visual_mask):
        
        textual_input = self.text_embed_layer(textual_input, textual_mask)
        textual_input = textual_input.last_hidden_state
        textual_input = self.text_embed_down(textual_input)
        textual_input = self.text_embed_norm(textual_input)
        
        vis_h = self.frame_layer(visual_input.transpose(1, 2)) #[B, hidden_dim, NUM_SAMPLE_CLIPS/TARGET_STRIDE], avg per stride 
        vis_h = self.frame_norm(vis_h.transpose(1, 2))
        vis_h = self.positional_sum(vis_h)
        visual_mask = torch.ones((vis_h.size(0), vis_h.size(1)), device='cuda')
        output = self.transformer(torch.concat([vis_h, textual_input], dim=1),  torch.concat([visual_mask, textual_mask], dim=1))
        
        vis_len = vis_h.size(1)
        vis_h = output[:, :vis_len, :].transpose(-1,-2)
        fused_h, map_mask = self.prop_layer(vis_h)
        # fused_h = self.fusion_layer(textual_input, textual_mask, map_h, map_mask)
        # fused_h = self.map_layer(fused_h, map_mask)
        prediction = self.pred_layer(fused_h) * map_mask

        return prediction, map_mask

    def extract_features(self, textual_input, textual_mask, visual_input):
        vis_h = self.frame_layer(visual_input.transpose(1, 2))
        map_h, map_mask = self.prop_layer(vis_h)

        fused_h = self.fusion_layer(textual_input, textual_mask, map_h, map_mask)
        fused_h = self.map_layer(fused_h, map_mask)
        prediction = self.pred_layer(fused_h) * map_mask

        return fused_h, prediction, map_mask
