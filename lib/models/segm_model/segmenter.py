import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import padding, unpadding
from timm.models.layers import trunc_normal_
from einops import rearrange

import pdb

class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x, attns = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        preds, patch_feats, cls_feats, decode_attns = self.decoder(x, (H, W))  # preds-[bz,21,32,32], patch_feats-[bz,1024,768], cls_feats-[bz,21,768]
        masks = F.interpolate(preds, size=(H, W), mode="bilinear")  # [bz, cls_num, 512, 512]
        masks = unpadding(masks, (H_ori, W_ori))

        outs = []
        outs.append(masks)

        if self.training:
            outs.append(preds)
            patch_feats = rearrange(patch_feats, "b (h w) n -> b n h w", h = preds.size()[-1])  # [bz, 768, 32, 32]
            outs.append(patch_feats)   
            outs.append(attns)
            outs.append(decode_attns)
        return outs  # [bz, cls_num, 512, 512],  [bz,21,32,32],  [bz, 768, 32, 32]


    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
