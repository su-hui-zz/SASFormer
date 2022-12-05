import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.modules.segformer_head import SegFormerHead
from lib.models.nets import mix_transformer
import numpy as np
import os
import pdb


class Segformer(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained="", pooling=None,):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        ## initilize encoder
        if pretrained:
            print("\n\n\npretrained!\n\n\n")
            state_dict = torch.load(os.path.join(pretrained, backbone+'.pth'))
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict,)

        if pooling=="gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling=="gap":
            self.pooling = F.adaptive_avg_pool2d

        self.dropout = torch.nn.Dropout2d(0.5)
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels, embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        

    def get_param_groups(self):

        param_groups = [[], [], [], []] # backbone; backbone_norm; cls_head; seg_head;
        
        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[2].append(param)

        return param_groups


    def forward(self, x, cam_only=False, seg_detach=True,):
        _x, _attns = self.encoder(x)  # _attns - a list of 8, [bz, head_num, 16*16, 16*16]
        _x1, _x2, _x3, _x4 = _x       # [bz,64,128,128], [bz,128,64,64], [bz,320,32,32], [bz,512,32,32]

        seg, seg_16, seg_32, seg_64= self.decoder(_x)   # [bz,21,128,128]

        # attn_cat = torch.cat(_attns[-2:], dim=1)#.detach()  # [bz, 16, 32*32, 32*32]
        # attn_cat = attn_cat + attn_cat.permute(0, 1, 3, 2)
        # attn_pred = self.attn_proj(attn_cat)           # [bz, 1, 32*32, 32*32]
        # attn_pred = torch.sigmoid(attn_pred)[:,0,...]  # [bz, 32*32, 32*32]

        # [bz,21,512,512], [bz,21,16,16], [bz,21,32,32], seg-[bz,21,128,128], _attns- a list of 8, [bz,1,256,256]
        outs = []
        #if self.training:
            #outs.append(F.interpolate(seg, size=(seg.size()[-2]*4,  seg.size()[-1]*4) , mode='bilinear', align_corners=True))
            #outs.append(F.interpolate(seg, size=(seg.size()[-2]//8, seg.size()[-1]//8), mode='bilinear', align_corners=True))
            #outs.append(F.interpolate(seg, size=(seg.size()[-2]//4, seg.size()[-1]//4), mode='bilinear', align_corners=True))
        outs.append(seg)
        if self.training:
            outs.append(seg_16)
            outs.append(seg_32)
            outs.append(seg_64)
            outs.append(_attns)
        return outs