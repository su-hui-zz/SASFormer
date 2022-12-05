##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: DonnyYou, RainbowSecret, JingyiXie, JianyuanGuo
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.loss.loss_helper import FSAuxOhemCELoss, FSOhemCELoss
from lib.loss.loss_helper import FSCELoss, FSAuxCELoss, FSAuxRMILoss
from lib.loss.loss_helper import SegFixLoss, TreeEnergyLoss, AffinityEnergyLoss, SegformerAffinityEnergyLoss
from lib.loss.rmi_loss import RMILoss

from lib.utils.tools.logger import Logger as Log
from lib.utils.distributed import is_distributed

import pdb

SEG_LOSS_DICT = {
    'fs_ce_loss': FSCELoss,
    'fs_ohemce_loss': FSOhemCELoss,
    'fs_auxce_loss': FSAuxCELoss,
    'fs_aux_rmi_loss': FSAuxRMILoss,
    'fs_auxohemce_loss': FSAuxOhemCELoss,
    'segfix_loss': SegFixLoss,
    'rmi_loss': RMILoss,
    'tree_loss': TreeEnergyLoss,
    'affinity_loss': AffinityEnergyLoss,
    'segformer_affinity_loss': SegformerAffinityEnergyLoss,
}


class LossManager(object):
    def __init__(self, configer):
        self.configer = configer

    def _parallel(self, loss):
        if is_distributed():
            Log.info('use distributed loss')
            return loss
            
        if self.configer.get('network', 'loss_balance') and len(self.configer.get('gpu')) > 1:
            Log.info('use DataParallelCriterion loss')
            from lib.extensions.parallel.data_parallel import DataParallelCriterion
            loss = DataParallelCriterion(loss)

        return loss

    def get_seg_loss(self, loss_type=None):
        key = self.configer.get('loss', 'loss_type') if loss_type is None else loss_type
        if key not in SEG_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)
        Log.info('use loss: {}.'.format(key))
        loss = SEG_LOSS_DICT[key](self.configer)
        return self._parallel(loss)
        
    def get_tree_loss(self, loss_type=None):
        key = self.configer.get('tree_loss', 'loss_type') if loss_type is None else loss_type
        if key not in SEG_LOSS_DICT:
            Log.error('Loss: {} not valid!'.format(key))
            exit(1)
        Log.info('use loss: {}.'.format(key))
        loss = SEG_LOSS_DICT[key](self.configer)
        return self._parallel(loss)

    def get_affinity_loss(self, loss_type=None):
        key = self.configer.get('affinity_loss', 'loss_type') if loss_type is None else loss_type
        loss = SEG_LOSS_DICT[key](self.configer)
        return self._parallel(loss)
