'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
This file is modified from:
https://github.com/rstrudel/segmenter/blob/master/segm/model/decoder.py
'''
import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_

from models.vision_transformer.vision_transformer import Block



class MaskTransformer(nn.Module):
    def __init__(
        self,
        n_cls=150,
        patch_size=16,
        d_encoder=384,
        n_layers=2,
        n_heads=6,
        d_model=384,
        d_ff=1536,
        drop_path_rate=0.0,
        dropout=0.1,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls
        self.d_model = d_model
        # self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, drop=dropout, attn_drop=dropout, drop_path=dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(self._init_weights)
        # trunc_normal_(self.cls_emb, std=0.02)
        nn.init.orthogonal(self.cls_emb)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, ret_cls_emb=False):
        '''
        x:  vit attention feature, B x h*w x d
        '''
        # B, hw, embed_dim
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        # B, (hw+N_cls), embed_dim
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # patches:      B x hw x embed_dim
        # cls_seg_feat: B x n_cls x embed_dim
        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)

        if ret_cls_emb:
            return masks, cls_seg_feat

        return masks

