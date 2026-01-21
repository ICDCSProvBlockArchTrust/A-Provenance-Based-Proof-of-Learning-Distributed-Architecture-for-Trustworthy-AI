
import torch
from torch import nn

class ImageLinearAttention(nn.Module):
    def __init__(self, chan, chan_out = None, kernel_size = 1, padding = 0, stride = 1, key_dim = 64, value_dim = 64, heads = 8, norm_queries = True):
        super().__init__()
        self.chan = chan
        chan_out = chan if chan_out is None else chan_out

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.heads = heads

        self.norm_queries = norm_queries

        conv_kwargs = {'padding': padding, 'stride': stride}
        self.to_q = nn.Conv2d(chan, key_dim * heads, kernel_size, **conv_kwargs)
        self.to_k = nn.Conv2d(chan, key_dim * heads, kernel_size, **conv_kwargs)
        self.to_v = nn.Conv2d(chan, value_dim * heads, kernel_size, **conv_kwargs)

        out_conv_kwargs = {'padding': padding}
        self.to_out = nn.Conv2d(value_dim * heads, chan_out, kernel_size, **out_conv_kwargs)

    def forward(self, x, context = None):
        b, c, h, w, k_dim, heads = *x.shape, self.key_dim, self.heads

        q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))

        q, k, v = map(lambda t: t.reshape(b, heads, -1, h * w), (q, k, v))

        q, k = map(lambda x: x * (self.key_dim ** -0.25), (q, k))

        if context is not None:
            context = context.reshape(b, c, 1, -1)
            ck, cv = self.to_k(context), self.to_v(context)
            ck, cv = map(lambda t: t.reshape(b, heads, k_dim, -1), (ck, cv))
            k = torch.cat((k, ck), dim=3)
            v = torch.cat((v, cv), dim=3)

        k = k.softmax(dim=-1)

        if self.norm_queries:
            q = q.softmax(dim=-2)

        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhdn,bhde->bhen', q, context)
        out = out.reshape(b, -1, h, w)
        out = self.to_out(out)
        return out

from fightingcv_attention.attention.MobileViTv2Attention import *
from fightingcv_attention.attention.A2Atttention import *
from fightingcv_attention.attention.ACmixAttention import *
from fightingcv_attention.attention.AFT import *
from fightingcv_attention.attention.Axial_attention import *
from fightingcv_attention.attention.BAM import *
from fightingcv_attention.attention.CBAM import *
from fightingcv_attention.attention.CoAtNet import *
from fightingcv_attention.attention.CoTAttention import *
from fightingcv_attention.attention.CoordAttention import *
from fightingcv_attention.attention.CrissCrossAttention import *
from fightingcv_attention.attention.Crossformer import *	
from fightingcv_attention.attention.DANet import *
from fightingcv_attention.attention.CoAtNet import *
from fightingcv_attention.attention.DAT import *
from fightingcv_attention.attention.ECAAttention import *
from fightingcv_attention.attention.EMSA import *
from fightingcv_attention.attention.ExternalAttention import *
from fightingcv_attention.attention.HaloAttention import *
from fightingcv_attention.attention.MOATransformer import *
from fightingcv_attention.attention.MUSEAttention import *
from fightingcv_attention.attention.MobileViTAttention import *
from fightingcv_attention.attention.MobileViTv2Attention import *
from fightingcv_attention.attention.OutlookAttention import *
from fightingcv_attention.attention.PSA import *
from fightingcv_attention.attention.ParNetAttention import *
from fightingcv_attention.attention.PolarizedSelfAttention import *
from fightingcv_attention.attention.ResidualAttention import *
from fightingcv_attention.attention.S2Attention import *
from fightingcv_attention.attention.SEAttention import *
from fightingcv_attention.attention.SGE import *
from fightingcv_attention.attention.SKAttention import *	
from fightingcv_attention.attention.SelfAttention import *
from fightingcv_attention.attention.ShuffleAttention import *	
from fightingcv_attention.attention.SimAM import *
from fightingcv_attention.attention.SimplifiedSelfAttention import *	
from fightingcv_attention.attention.TripletAttention import *


def create_attn_module(AttnCls, IMG_SIZE, CHANNELS): 
    if AttnCls is ImageLinearAttention: 
        attn = ImageLinearAttention(chan=32) 
    elif AttnCls is DoubleAttention: 
        attn = AttnCls(32,IMG_SIZE,IMG_SIZE,True)
    elif AttnCls is BAMBlock: 
        attn = AttnCls(channel=CHANNELS,reduction=CHANNELS-1,dia_val=1)
    elif AttnCls in [PSA]: 
        attn = AttnCls(channel=CHANNELS,reduction=3)
    elif AttnCls in [SKAttention]: 
        attn = AttnCls(channel=32,reduction=2)
    elif AttnCls in [SequentialPolarizedSelfAttention, SEAttention, SimAM, OutlookAttention]: 
        attn = AttnCls(CHANNELS)
    elif AttnCls == CrissCrossAttention: 
        attn = AttnCls(32)
    elif AttnCls is CoTAttention: 
        attn = AttnCls(CHANNELS, 32)
    elif AttnCls is ECAAttention: 
        attn = AttnCls(3)
    elif AttnCls is SpatialGroupEnhance: 
        attn = AttnCls(8)
    elif AttnCls is AxialAttention: 
        attn = AttnCls(3, heads = 3)
    elif AttnCls is ShuffleAttention: 
        attn = AttnCls(channel=CHANNELS, G=2)
    elif AttnCls in [ACmix, AFT_FULL]: 
        attn = AttnCls(CHANNELS, CHANNELS)
    elif AttnCls == CoordAtt: 
        attn = AttnCls(32, 32)
    elif AttnCls is DAModule:
        attn = AttnCls(CHANNELS, 3, IMG_SIZE, IMG_SIZE)
    elif AttnCls is EMSA:
        attn = AttnCls(d_model=IMG_SIZE, d_k=128, d_v=128, h=8,H=8,W=8,ratio=2,apply_transform=True)
    elif AttnCls in [MUSEAttention]:
        attn = AttnCls(d_model=IMG_SIZE, d_k=128, d_v=128, h=8)
    elif AttnCls is TripletAttention:
        attn = AttnCls()
    elif AttnCls in [MobileViTv2Attention, ExternalAttention]:
        attn = AttnCls(d_model=IMG_SIZE//2)
    else: 
        attn = AttnCls(d_model=IMG_SIZE)

    return attn

attn_classes = {
    str(k.__name__): k
    for k in [
        # ShuffleAttention, 
        ImageLinearAttention, 
        MobileViTv2Attention, 
        DoubleAttention, 
        # BAMBlock, 
        CoordAtt, 
        CrissCrossAttention, 
        # ECAAttention, 
        # ExternalAttention, 
        # SequentialPolarizedSelfAttention, 
        # SEAttention, 
        SKAttention, 
        SimAM, 
        TripletAttention
    ]
}


        # AxialAttention, 
        # CoTAttention, 
        # PSA, 