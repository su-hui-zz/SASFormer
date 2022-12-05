##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Copyright (c) 2022 megvii-model. All Rights Reserved.
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Microsoft Research
## Author: RainbowSecret, LangHuang, JingyiXie, JianyuanGuo
## Copyright (c) 2019
## yuyua@microsoft.com
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Our approaches including FCN baseline, HRNet, OCNet, ISA, OCR
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FCN baseline 
from lib.models.nets.fcnet import FcnNet

# OCR
from lib.models.nets.ocrnet import SpatialOCRNet, ASPOCRNet
from lib.models.nets.ideal_ocrnet import IdealSpatialOCRNet, IdealSpatialOCRNetB, IdealSpatialOCRNetC, IdealGatherOCRNet, IdealDistributeOCRNet

# HRNet
from lib.models.nets.hrnet import HRNet_W48
from lib.models.nets.hrnet import HRNet_W48_OCR, HRNet_W48_ASPOCR, HRNet_W48_OCR_B

# OCNet
from lib.models.nets.ocnet import BaseOCNet, AspOCNet

# ISA Net
from lib.models.nets.isanet import ISANet

# CE2P
from lib.models.nets.ce2pnet import CE2P_OCRNet, CE2P_IdealOCRNet, CE2P_ASPOCR

# SegFix
from lib.models.nets.segfix import SegFix_HRNet

# segmenter
from lib.models.segm_model.factory import create_segmenter

# segformer
from lib.models.nets.segformer import Segformer

# DeeplabV3
# from lib.models.nets.deeplab import DeepLabV3, DeepLabV2
from lib.models.nets.deeplabv3plus import DeepLabV3Plus
from lib.models.nets.treefcn import TreeFCN

from lib.utils.tools.logger import Logger as Log

import pdb

SEG_MODEL_DICT = {
    # SegFix
    'segfix_hrnet': SegFix_HRNet,
    # OCNet series
    'base_ocnet': BaseOCNet,
    'asp_ocnet': AspOCNet,
    # ISA Net
    'isanet': ISANet,
    # OCR series
    'spatial_ocrnet': SpatialOCRNet,
    'spatial_asp_ocrnet': ASPOCRNet,
    # OCR series with ground-truth   
    'ideal_spatial_ocrnet': IdealSpatialOCRNet,
    'ideal_spatial_ocrnet_b': IdealSpatialOCRNetB,
    'ideal_spatial_ocrnet_c': IdealSpatialOCRNetC, 
    'ideal_gather_ocrnet': IdealGatherOCRNet,
    'ideal_distribute_ocrnet': IdealDistributeOCRNet,
    # HRNet series
    'hrnet_w48': HRNet_W48,
    'hrnet_w48_ocr': HRNet_W48_OCR,
    'hrnet_w48_ocr_b': HRNet_W48_OCR_B,
    'hrnet_w48_asp_ocr': HRNet_W48_ASPOCR,
    # CE2P series
    'ce2p_asp_ocrnet': CE2P_ASPOCR,
    'ce2p_ocrnet': CE2P_OCRNet,
    'ce2p_ideal_ocrnet': CE2P_IdealOCRNet, 
    # baseline series
    'fcnet': FcnNet,
    'deeplabv3p': DeepLabV3Plus,
    'treefcn': TreeFCN,

    # segmenter
    'segmenter': create_segmenter,

    # segformer
    'segformer':  Segformer
}


class ModelManager(object):
    def __init__(self, configer):
        self.configer = configer

    def semantic_segmentor(self):
        model_name = self.configer.get('network', 'model_name')

        if model_name not in SEG_MODEL_DICT:
            Log.error('Model: {} not valid!'.format(model_name))
            exit(1)

        if model_name == 'segmenter':
            config_net = self.configer.get('segmenter')
            model = SEG_MODEL_DICT[model_name](config_net)
        elif model_name == 'segformer':
            model = SEG_MODEL_DICT[model_name](backbone=self.configer.get('network','config'), 
                                               stride=self.configer.get('network','stride'),
                                               num_classes=self.configer.get('data','num_classes'),
                                               embedding_dim=256,
                                               pretrained=self.configer.get('network','pretrained'))
        else:
            model = SEG_MODEL_DICT[model_name](self.configer)

        return model
