"""
UniTS
"""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from functools import partial

from timm.layers import DropPath
from timm.layers.helpers import to_2tuple

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

def calculate_unfold_output_length(input_length, size, step):
    # Calculate the number of windows
    num_windows = (input_length - size) // step + 1
    return num_windows


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, query):
        B, N, C = x.shape
        q = self.q(query).reshape(
            B, query.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        var_num = query.shape[1]
        kv = self.kv(x).reshape(B, N, 2, self.num_heads,
                                self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        k = self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, var_num, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :, :x.size(2)]


class SeqAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(
            q, k, v,  # attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VarAttention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, P, C = x.shape

        qkv = self.qkv(x).reshape(B, N, P, 3, self.num_heads,
                                  self.head_dim).permute(3, 0, 2, 4, 1, 5) # [3, B, P, num_heads, N, head_dim]
        q, k, v = qkv.unbind(0) # [B, P, num_heads, N, head_dim]
        q, k = self.q_norm(q), self.k_norm(k)

        q = q.mean(dim=1, keepdim=False) # [B, num_heads, N, head_dim]
        k = k.mean(dim=1, keepdim=False) # [B, num_heads, N, head_dim]
        # [B, num_heads, N, head_dim, P] -> [B, num_heads, N, head_dim * P]
        v = v.permute(0, 2, 3, 4, 1).reshape(B, self.num_heads, N, -1)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = x.view(B, self.num_heads, N, -1, P).permute(0,
                                                        2, 4, 1, 3).reshape(B, N, P, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GateLayer(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gate = nn.Linear(dim, 1)

    def forward(self, x):
        gate_value = self.gate(x)
        return gate_value.sigmoid() * x


class SeqAttBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_seq = SeqAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.ls1 = GateLayer(dim, init_values=init_values)
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask):
        x_input = x
        x = self.norm1(x)
        n_vars, n_seqs = x.shape[1], x.shape[2]
        x = torch.reshape(
            x, (-1, x.shape[-2], x.shape[-1]))
        x = self.attn_seq(x, attn_mask)
        x = torch.reshape(
            x, (-1, n_vars, n_seqs, x.shape[-1]))
        x = x_input + self.drop_path1(self.ls1(x))
        return x


class VarAttBlock(nn.Module):

    def __init__(
            self,
            dim,
            num_heads, 
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_var = VarAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = GateLayer(dim, init_values=init_values)
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn_var(self.norm1(x))))
        return x


class MLPBlock(nn.Module):

    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            proj_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = GateLayer(dim, init_values=init_values)
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class BasicBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=8.,
            qkv_bias=False,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.seq_att_block = SeqAttBlock(dim=dim, num_heads=num_heads,
                                         qkv_bias=qkv_bias, qk_norm=qk_norm,
                                         attn_drop=attn_drop, init_values=init_values, proj_drop=proj_drop,
                                         drop_path=drop_path, norm_layer=norm_layer)

        self.var_att_block = VarAttBlock(dim=dim, num_heads=num_heads,
                                         qkv_bias=qkv_bias, qk_norm=qk_norm,
                                         attn_drop=attn_drop, init_values=init_values, proj_drop=proj_drop,
                                         drop_path=drop_path, norm_layer=norm_layer)

        self.mlp = MLPBlock(dim=dim, mlp_ratio=mlp_ratio,
                                    proj_drop=proj_drop, init_values=init_values, drop_path=drop_path,
                                    act_layer=act_layer, norm_layer=norm_layer,)

    def forward(self, x):
        x = self.seq_att_block(x, attn_mask=None)
        x = self.var_att_block(x)
        x = self.mlp(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        assert self.patch_len == self.stride, "non-overlap"
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x)
        return self.dropout(x), n_vars


class CLSHead(nn.Module):
    def __init__(self, d_model, head_dropout=0):
        super().__init__()
        d_mid = d_model
        self.proj_in = nn.Linear(d_model, d_mid)
        self.cross_att = CrossAttention(d_mid)

        self.mlp = MLPBlock(dim=d_mid, mlp_ratio=8,
                            proj_drop=head_dropout, init_values=None, drop_path=0.0,
                            act_layer=nn.GELU, norm_layer=nn.LayerNorm)

    def forward(self, x, category_token=None, return_feature=False):
        x = self.proj_in(x)
        B, V, L, C = x.shape
        x = x.view(-1, L, C) # [B*V, L, C]
        cls_token = x[:, -1:] # [B*V, 1, C]
        cls_token = self.cross_att(x, query=cls_token) # [B*V, 1, C]
        cls_token = cls_token.reshape(B, V, -1, C) # [B, V, 1, C]

        cls_token = self.mlp(cls_token)
        if return_feature:
            return cls_token # [B, V, 1, C]
        m = category_token.shape[2] # num_class
        cls_token = cls_token.expand(B, V, m, C)
        distance = torch.einsum('nvkc,nvmc->nvm', cls_token, category_token) # sum(dot_prod(kc, mc)) -> m, where k = m

        distance = distance.mean(dim=1) # [B, m]
        return distance


def random_rotation(tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply a random rotation to each sample in the batch.
    Input tensor shape: [B, L, 6] or [B, L, 9]
    Each sample gets a unique rotation matrix applied to each 3-channel part.
    Assumes tensor is on GPU.
    """
    B, L, C = tensor.shape
    assert C in [6, 9], "Expected last dimension of size 6 or 9"

    # Generate random rotation matrices for each sample
    quaternions = torch.randn(B, 4, device=tensor.device)
    quaternions = quaternions / quaternions.norm(dim=1, keepdim=True)
    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]

    rot_mats = torch.zeros(B, 3, 3, device=tensor.device)
    rot_mats[:, 0, 0] = 1 - 2*(y**2 + z**2)
    rot_mats[:, 0, 1] = 2*(x*y - z*w)
    rot_mats[:, 0, 2] = 2*(x*z + y*w)
    rot_mats[:, 1, 0] = 2*(x*y + z*w)
    rot_mats[:, 1, 1] = 1 - 2*(x**2 + z**2)
    rot_mats[:, 1, 2] = 2*(y*z - x*w)
    rot_mats[:, 2, 0] = 2*(x*z - y*w)
    rot_mats[:, 2, 1] = 2*(y*z + x*w)
    rot_mats[:, 2, 2] = 1 - 2*(x**2 + y**2)

    new_tensor = tensor.clone()
    # Expanded rotation application:
    # For input with 6 channels: apply rotation to parts [0:3] and [3:6]
    if C == 6:
        part1 = new_tensor[:, :, 0:3]
        part1_rot = torch.bmm(part1, rot_mats.transpose(1, 2))
        new_tensor[:, :, 0:3] = part1_rot

        part2 = new_tensor[:, :, 3:6]
        part2_rot = torch.bmm(part2, rot_mats.transpose(1, 2))
        new_tensor[:, :, 3:6] = part2_rot
    # For input with 9 channels: apply rotation to parts [0:3], [3:6] and [6:9]
    elif C == 9:
        part1 = new_tensor[:, :, 0:3]
        part1_rot = torch.bmm(part1, rot_mats.transpose(1, 2))
        new_tensor[:, :, 0:3] = part1_rot

        part2 = new_tensor[:, :, 3:6]
        part2_rot = torch.bmm(part2, rot_mats.transpose(1, 2))
        new_tensor[:, :, 3:6] = part2_rot

        part3 = new_tensor[:, :, 6:9]
        part3_rot = torch.bmm(part3, rot_mats.transpose(1, 2))
        new_tensor[:, :, 6:9] = part3_rot

    return new_tensor

class Model(nn.Module):
    """
    UniTS: Building a Unified Time Series Model
    d_model=256, stride=8, 
    patch_len=8, dropout=0.1, 
    e_layers=3, n_heads=8,
    min_mask_ratio=0.7, max_mask_ratio=0.8
    """

    def __init__(self, 
                 num_class,
                 args,
                ):
        super().__init__()

        enc_in = 6
        # Tokens settings
        self.mask_token = nn.Parameter(torch.zeros(1, enc_in, 1, args.d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, enc_in, 1, args.d_model))
        self.category_tokens = nn.Parameter(torch.zeros(1, enc_in, num_class, args.d_model))

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.category_tokens, std=0.02)

        ### model settings ###
        self.stride = args.stride
        self.pad = args.stride
        self.patch_len = args.patch_len

        # input processing
        self.patch_embeddings = PatchEmbedding(
            args.d_model, args.patch_len, args.stride, args.stride, args.dropout)
        self.position_embedding = PositionalEmbedding(args.d_model)

        # basic blocks
        self.block_num = args.e_layers
        self.blocks = nn.ModuleList(
            [BasicBlock(dim=args.d_model, num_heads=args.n_heads, qkv_bias=False, qk_norm=False,
                        mlp_ratio=8., proj_drop=args.dropout, attn_drop=0., drop_path=0.,
                        init_values=None) for l in range(args.e_layers)]
        )

        # output processing
        self.cls_head = CLSHead(args.d_model, head_dropout=args.dropout)

    def tokenize(self, x, mask=None):
        x = x.permute(0, 2, 1) # [B, V, L]
        remainder = x.shape[2] % self.patch_len
        if remainder != 0:
            padding = self.patch_len - remainder
            x = F.pad(x, (0, padding))
        else:
            padding = 0
        x, n_vars = self.patch_embeddings(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1])) # [B, V, L, C]
        x = x + self.position_embedding(x)
        return x, n_vars, padding

    def backbone(self, x):
        attn_mask = None
        for block in self.blocks:
            x = block(x)
        return x
    
    def classify(self, x):
        x_cls_prompt = self.cls_token.repeat(x.shape[0], 1, 1, 1)
        category_token = self.category_tokens.repeat(x.shape[0], 1, 1, 1)

        x = torch.cat((x, x_cls_prompt), dim=2)
        x = self.backbone(x) # [B, V, L, C]

        out = self.cls_head(x, category_token)

        return out
    
    def forward(self, x):
        # x: [B, L, V]
        x, _, _ = self.tokenize(x)
        return self.classify(x)

@dataclass
class ModelArgs:
    d_model: int
    n_heads: int
    e_layers: int
    patch_len: int
    stride: int
    dropout: float
    
    @classmethod
    def from_args(cls, args):
        # ignore args that are not in the dataclass
        return cls(**{k: v for k, v in vars(args).items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_dict(cls, args):
        return cls(**args)
    
    def to_dict(self):
        return vars(self)

if __name__ == '__main__':
    args = ModelArgs(64, 8, 3, 8, 8, 0.1)
    model = Model(6, args)
    x = torch.randn(1, 120, 9)
    y = model(x)
    print(y.shape)
    # print the para. number
    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_parameters}")