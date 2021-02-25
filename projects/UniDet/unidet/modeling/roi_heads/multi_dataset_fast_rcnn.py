import logging
import math
from typing import Dict, Union
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from .custom_fast_rcnn import CustomFastRCNNOutputLayers, CustomFastRCNNOutputs

class MultiDatasetFastRCNNOutputLayers(CustomFastRCNNOutputLayers):
    def __init__(
        self,
        cfg,
        num_classes_list,
        input_shape: ShapeSpec,
        **kwargs
    ):
        super().__init__(cfg, input_shape, **kwargs)
        del self.cls_score
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        prior_prob = cfg.MODEL.ROI_BOX_HEAD.PRIOR_PROB
        if cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE:
            bias_value = -math.log((1 - prior_prob) / prior_prob)
        else:
            bias_value = 0
        self.openimage_index = cfg.MULTI_DATASET.DATASETS.index('oid')
        self.num_datasets = len(num_classes_list)
        self.cls_score = nn.ModuleList()
        for num_classes in num_classes_list:
            self.cls_score.append(nn.Linear(input_size, num_classes + 1))
            nn.init.normal_(self.cls_score[-1].weight, std=0.01)
            nn.init.constant_(self.cls_score[-1].bias, bias_value)

    def forward(self, x, dataset_source=-1):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        proposal_deltas = self.bbox_pred(x)
        if dataset_source >= 0:
            scores = self.cls_score[dataset_source](x)
        else:
            scores = [self.cls_score[d](x) for d in range(self.num_datasets)]
        return scores, proposal_deltas

    def losses(self, predictions, proposals, dataset_source):
        is_open_image = (dataset_source == self.openimage_index)
        scores, proposal_deltas = predictions
        losses = CustomFastRCNNOutputs(
            self.cfg,
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            self.freq_weight if is_open_image else None, 
            self.hierarchy_weight if is_open_image else None,
        ).losses()
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

