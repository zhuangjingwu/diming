from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from .utils import ChannelPool
from timm.models.layers import DropPath, Mlp, trunc_normal_
from timm.models.helpers import named_apply
from functools import partial
from models.pos_utils import get_2d_sincos_pos_embed


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_mem = None

    def forward(self, x, t_h, t_w, s_h, s_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        q_mt, q_s = torch.split(q, [t_h * t_w, s_h * s_w], dim=2)
        k_mt, k_s = torch.split(k, [t_h * t_w, s_h * s_w], dim=2)
        v_mt, v_s = torch.split(v, [t_h * t_w, s_h * s_w], dim=2)

        # asymmetric mixed attention
        attn = (q_mt @ k_mt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_mt = (attn @ v_mt).transpose(1, 2).reshape(B, t_h*t_w, C)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s = (attn @ v).transpose(1, 2).reshape(B, s_h*s_w, C)

        x = torch.cat([x_mt, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, t_h, t_w, s_h, s_w):
        x = x + self.drop_path1(self.attn(self.norm1(x), t_h, t_w, s_h, s_w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class AttentionFusionBlock(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        img_size_uav = opt.data_config["UAVhw"][0]
        img_size_satellite = opt.data_config["Satellitehw"][0]
        opt.data_config["UAVhw"][0]
        patch_size = opt.model["head"]["patch_size"]
        dropout_rate = opt.model["head"]["dropout_rate"]
        input_ndim = opt.model["head"]["input_ndim"]
        mid_ndim = opt.model["head"]["mid_ndim"]
        attention_layer_num = opt.model["head"]["attention_layer_num"]

        dpr = [x.item() for x in torch.linspace(0, dropout_rate, attention_layer_num)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=input_ndim, num_heads=8, mlp_ratio=4, qkv_bias=True,
                drop=dropout_rate, attn_drop=dropout_rate, drop_path=dpr[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6)) for i in range(attention_layer_num)])

        self.grid_size_uav = img_size_uav // patch_size
        self.grid_size_satellite = img_size_satellite // patch_size
        self.num_patches_uav = self.grid_size_uav ** 2
        self.num_patches_satellite = self.grid_size_satellite ** 2

        self.pos_drop = nn.Dropout(p=dropout_rate)

        self.out_linears = nn.ModuleList()
        last_ndim = input_ndim
        for module_index, channel in enumerate(mid_ndim):
            self.out_linears.append(nn.Linear(last_ndim, channel))
            # setattr(self, "output_linear_{}".format(module_index), nn.Linear(last_ndim, channel))
            last_ndim = channel

        named_apply(self.init_weights_vit_timm, self)
        # self._reset_parameters()

        self.pos_embed_uav = nn.Parameter(
            torch.zeros(1, self.num_patches_uav, input_ndim),
            requires_grad=False)
        self.pos_embed_satellite = nn.Parameter(
            torch.zeros(1, self.num_patches_satellite, input_ndim),
            requires_grad=False)

        self.init_pos_embed()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_weights_vit_timm(self, module: nn.Module, name: str):
        """ ViT weight initialization, original timm impl (for reproducibility) """
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif hasattr(module, 'init_weights'):
            module.init_weights()

    def init_pos_embed(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_uav = get_2d_sincos_pos_embed(
            self.pos_embed_uav.shape[-1],
            int(self.num_patches_uav ** .5),
            cls_token=False)
        self.pos_embed_uav.data.copy_(torch.from_numpy(pos_embed_uav).float().unsqueeze(0))

        pos_embed_satellite = get_2d_sincos_pos_embed(
            self.pos_embed_satellite.shape[-1],
            int(self.num_patches_satellite ** .5),
            cls_token=False)
        self.pos_embed_satellite.data.copy_(
            torch.from_numpy(pos_embed_satellite).float().unsqueeze(0))

    def forward_single(self, z, x):
        B, C, _, _ = z.shape
        H_uav = W_uav = self.grid_size_uav
        H_satellite = W_satellite = self.grid_size_satellite

        # convert B,C,H,W->B,N,C
        z = z.flatten(2).transpose(1, 2).contiguous()
        x = x.flatten(2).transpose(1, 2).contiguous()
        # position embedding
        x_uav = z + self.pos_embed_uav
        x_satellite = x + self.pos_embed_satellite
        x = torch.cat([x_uav, x_satellite], dim=1)
        x = self.pos_drop(x)
        # attention block
        for blk in self.blocks:
            x = blk(x, H_uav, W_uav, H_satellite, W_satellite)

        x_uav, x_satellite = torch.split(x, [H_uav*W_uav, H_satellite*W_satellite], dim=1)

        for ind in range(len(self.out_linears)):
            x_satellite = self.out_linears[ind](x_satellite)

        # x_uav_2d = x_uav.transpose(1, 2).reshape(B, C, H_uav, W_uav)
        x_satellite_2d = x_satellite.transpose(1, 2).reshape(B, 1, H_satellite, W_satellite)

        return x_satellite_2d, None
    
    def forward(self, z, x):
        return self.forward_single(z[0], x)
        

class CrossAttentionFusion(nn.Module):
    def __init__(self, opt):
        super(CrossAttentionFusion, self).__init__()
        dropout_rate = opt.model["head"]["dropout_rate"]
        input_ndim = opt.model["head"]["input_ndim"]
        mid_ndim = opt.model["head"]["mid_ndim"]
        attention_layer_num = opt.model["head"]["attention_layer_num"]
        linear_layer_num = opt.model["head"]["linear_layer_num"]

        self.CrossAttention = nn.MultiheadAttention(input_ndim, 8, dropout=dropout_rate)
        self.CrossAttentionNorm = nn.LayerNorm(input_ndim)
        # attention module init
        self.SelfAttentionModuleList = nn.ModuleList(
            [nn.MultiheadAttention(input_ndim, 8, dropout=dropout_rate)
             for _ in range(attention_layer_num)])
        # norm module init
        self.NormModuleList = nn.ModuleList([nn.LayerNorm(input_ndim)
                                            for _ in range(attention_layer_num)])
        # Linear module init
        self.LinearModuleList = nn.ModuleList()
        for _ in range(linear_layer_num):
            self.LinearModuleList.append(
                nn.Sequential(
                    nn.Linear(input_ndim, mid_ndim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate)
                )
            )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, z, x):
        if len(z.shape) == 4:
            z_ = z.reshape(z.shape[0], z.shape[1], z.shape[2]*z.shape[3])
            x_ = x.reshape(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
        k = v = z_.permute(2, 0, 1).contiguous()
        q = x_.permute(2, 0, 1).contiguous()

        s1 = self.CrossAttentionNorm(self.CrossAttention(query=q, key=k, value=v)[0])

        for layer_ind in range(len(self.SelfAttentionModuleList)):
            s1 = self.NormModuleList[layer_ind](
                self.SelfAttentionModuleList[layer_ind](query=s1, key=s1, value=s1)[0])

        res = s1.permute(1, 0, 2).contiguous()

        for layer_ind in range(len(self.LinearModuleList)):
            res = self.LinearModuleList[layer_ind](res)

        res = s1.permute(0, 2, 1).contiguous()

        return res.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])


class MultiCrossAttentionFusion(nn.Module):
    def __init__(self, opt):
        super(MultiCrossAttentionFusion, self).__init__()
        level_nums = len(opt.model["neck"]["UAV_output_index"])
        self.HeadModuleList = nn.ModuleList()
        for _ in range(level_nums):
            self.HeadModuleList.append(CrossAttentionFusion(opt))

        head_pool = opt.model["head"]["head_pool"]
        # pool method init
        self.pool = ChannelPool(mode=head_pool)

    def forward(self, z, x):
        res = []
        for ind, z_part in enumerate(z):
            res.append(self.HeadModuleList[ind](z_part, x))
        res = torch.concat(res, dim=1)
        res = self.pool(res)
        return res


class SwinTrack(nn.Module):
    def __init__(self, opt):
        super(SwinTrack, self).__init__()
        original_dim = 384
        opt.dim = 48
        self.linear1 = nn.Linear(original_dim, opt.dim)
        self.linear2 = nn.Linear(original_dim, opt.dim)
        self.z_patches = opt.UAVhw[0] // 16 * opt.UAVhw[1] // 16
        self.x_patches = opt.Satellitehw[0] // 16 * opt.Satellitehw[1] // 16
        num_patches = self.z_patches+self.x_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, opt.dim))
        self.attnBlock = nn.Sequential(Block(dim=opt.dim, num_heads=12))
        self.out_norm = nn.LayerNorm(opt.dim)
        # cls and loc
        self.cls_linear = nn.Linear(opt.dim, 1)
        self.loc_linear = nn.Linear(opt.dim, 2)

    def forward(self, z, x):
        z_feat = self.linear1(z)
        x_feat = self.linear2(x)
        concat_feature = torch.concat((z_feat, x_feat), dim=1)
        concat_feature += self.pos_embed
        out_feature = self.attnBlock(concat_feature)[:, self.z_patches:, :]
        out_feature = self.out_norm(out_feature)
        cls_feat = self.cls_linear(out_feature)
        # loc_feat = self.loc_linear(decoder_feat)

        return cls_feat  # B*1*25*25 B*2*25*25


class CrossViewCrossAttention(nn.Module):
    """
    Cross-View Cross-Attention (CVCA) mechanism
    - UAV token作为Query
    - 卫星feature map作为Key/Value
    """
    def __init__(self, opt):
        super(CrossViewCrossAttention, self).__init__()
        dropout_rate = opt.model["head"]["dropout_rate"]
        input_ndim = opt.model["head"]["input_ndim"]
        attention_layer_num = opt.model["head"]["attention_layer_num"]
        self.mid_ndim = opt.model["head"]["mid_ndim"]
        
        # 跨视角注意力层：卫星特征作为Query，UAV特征作为Key/Value
        # 这样输出长度会是卫星特征的长度（N_x）
        self.cross_attention = nn.MultiheadAttention(input_ndim, 8, dropout=dropout_rate)
        self.cross_attention_norm = nn.LayerNorm(input_ndim)
        
        # 自注意力模块列表
        self.self_attention_modules = nn.ModuleList(
            [nn.MultiheadAttention(input_ndim, 8, dropout=dropout_rate)
             for _ in range(attention_layer_num)]
        )
        
        # 归一化模块列表
        self.norm_modules = nn.ModuleList(
            [nn.LayerNorm(input_ndim) for _ in range(attention_layer_num)]
        )
        
        # 线性层用于精匹配
        self.fine_match_linear = nn.Linear(input_ndim, self.mid_ndim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # 粗匹配：卫星图上的注意力热图
        # 精匹配：经纬度偏移回归
        self.fine_reg_layer = nn.Linear(self.mid_ndim, 2)  # 输出经纬度偏移
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, z, x):
        """
        z: UAV特征 (B, C, H, W)
        x: 卫星特征 (B, C, H, W)
        """
        if isinstance(z, list):
            z = z[0]  # UAV特征应该是单个特征图
        
        B, C, H_z, W_z = z.shape
        _, _, H_x, W_x = x.shape
        N_z = H_z * W_z
        N_x = H_x * W_x
        
        # 将UAV特征转换为Key/Value (B, N_z, C)
        z_flat = z.flatten(2).transpose(1, 2).contiguous()  # (B, N_z, C)
        
        # 将卫星特征转换为Query (B, N_x, C)
        x_flat = x.flatten(2).transpose(1, 2).contiguous()  # (B, N_x, C)
        
        # 跨视角注意力：卫星token作为Query，UAV feature map作为Key/Value
        # 这样输出长度会是卫星特征的长度（N_x）
        q = x_flat.permute(1, 0, 2).contiguous()  # (N_x, B, C)
        k = v = z_flat.permute(1, 0, 2).contiguous()  # (N_z, B, C)
        
        # 计算注意力
        attn_output, attn_weights = self.cross_attention(query=q, key=k, value=v)
        attn_output = self.cross_attention_norm(attn_output)
        
        # 自注意力融合
        for layer_ind in range(len(self.self_attention_modules)):
            attn_output = self.norm_modules[layer_ind](
                self.self_attention_modules[layer_ind](query=attn_output, key=attn_output, value=attn_output)[0]
            )
        
        # 将注意力输出转换为特征 (B, N_x, C)
        satellite_features = attn_output.permute(1, 0, 2).contiguous()  # (B, N_x, C)
        
        # 粗匹配：在卫星图上生成注意力热图
        # 注意力权重形状为 (B, N_x, N_z)，取UAV中心token的注意力权重作为粗匹配结果
        center_idx = N_z // 2
        coarse_cls = attn_weights[:, :, center_idx].view(B, 1, H_x, W_x)  # (B, 1, H_x, W_x)
        
        # 精匹配：对每个卫星网格点进行经纬度偏移回归
        # 对每个卫星网格点的特征进行处理
        fine_features = self.fine_match_linear(satellite_features)  # (B, N_x, mid_ndim)
        fine_features = self.gelu(fine_features)
        fine_features = self.dropout(fine_features)
        
        # 输出每个网格点的经纬度偏移 (B, N_x, 2)
        fine_reg = self.fine_reg_layer(fine_features)  # (B, N_x, 2)
        
        # 重塑为 (B, 2, H_x, W_x) 格式，符合验证阶段的使用要求
        loc_bias = fine_reg.view(B, H_x, W_x, 2).permute(0, 3, 1, 2).contiguous()  # (B, 2, H_x, W_x)
        
        return coarse_cls, loc_bias
