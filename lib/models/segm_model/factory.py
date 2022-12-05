from pathlib import Path
import yaml
import torch
import math
import os
import torch.nn as nn

from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer

from .vit import VisionTransformer
from .utils import checkpoint_filter_fn
from .decoder import DecoderLinear
from .decoder import MaskTransformer
from .segmenter import Segmenter
#import segm.utils.torch as ptu

#import numpy 
#import cv2
import torch.nn.functional as F

import pdb

@register_model
def vit_base_patch8_384(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch8_384",
        pretrained=pretrained,
        default_cfg=dict(
            url="",
            input_size=(3, 384, 384),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            num_classes=1000,
        ),
        **model_kwargs,
    )
    return model


def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    backbone = model_cfg.pop("backbone")

    normalization = model_cfg.pop("normalization")
    model_cfg["n_cls"] = 1000
    mlp_expansion_ratio = 4
    model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]

    if backbone in default_cfgs:
        default_cfg = default_cfgs[backbone]
    else:
        default_cfg = dict(
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )

    default_cfg["input_size"] = (
        3,
        model_cfg["image_size"][0],
        model_cfg["image_size"][1],
    )
    model = VisionTransformer(**model_cfg)  # vit_base_patch16_384

    weights = torch.load("./pretrained/deit_base_patch16_384-8de9b5d1.pth",map_location='cpu')['model']
    if model.pos_embed.size() != weights['pos_embed'].size():
        cls_pos_embed = weights['pos_embed'][:,0,:].unsqueeze(dim=1)
        ori_wh = int((weights['pos_embed'].size()[1]-1)**0.5)
        tar_wh = int((model.pos_embed.size()[1]-1)**0.5)
        ori_pos_embed = weights['pos_embed'][:,1:,:].view(1, ori_wh, ori_wh, -1).permute(0, 3, 1, 2)

        tar_pos_embed = F.interpolate(ori_pos_embed, size=(tar_wh, tar_wh), mode='bilinear', align_corners=True)
        tar_pos_embed = tar_pos_embed.view(1, -1, tar_wh*tar_wh).permute(0,2,1)
        weights['pos_embed'] = torch.cat([cls_pos_embed, tar_pos_embed], dim=1)
    
    model.load_state_dict(weights)

    
    # if backbone == "vit_base_patch8_384":
    #     path = os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth")
    #     state_dict = torch.load(path, map_location="cpu")
    #     filtered_dict = checkpoint_filter_fn(state_dict, model)
    #     model.load_state_dict(filtered_dict, strict=True)
    # elif "deit" in backbone:
    #     load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn)
    # else:
    #     load_custom_pretrained(model, default_cfg)

    return model


def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["patch_size"] = encoder.patch_size

    if "linear" in name:
        decoder = DecoderLinear(**decoder_cfg)
    elif name == "mask_transformer":
        dim = encoder.d_model
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder


def create_segmenter(model_cfg):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    decoder_cfg["n_cls"] = model_cfg["n_cls"]
    encoder = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)
    model = Segmenter(encoder, decoder, n_cls=model_cfg["n_cls"])

    return model


def load_model(model_path):
    variant_path = Path(model_path).parent / "variant.yml"
    with open(variant_path, "r") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    net_kwargs = variant["net_kwargs"]

    model = create_segmenter(net_kwargs)
    #data = torch.load(model_path, map_location=ptu.device)
    data = torch.load(model_path, map_location='cpu')
    checkpoint = data["model"]

    model.load_state_dict(checkpoint, strict=True)

    return model, variant
