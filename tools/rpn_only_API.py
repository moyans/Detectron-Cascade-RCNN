#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
import os
import sys
import yaml

import numpy as np
from caffe2.python import workspace
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import load_cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.logging import setup_logging
import detectron.core.rpn_generator as rpn_engine
import detectron.core.test_engine as model_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.boxes as box_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

class RPN_INFERENCE(object):
    def __init__(self, cfg_file, weights, gpu_id=0):
        self.NMS = 0.1
        self.thresh = 0.6
        self.classs = '3477'
        self.class_num = 2
        self.DETECTIONS_PER_IM = 1000

        cfg.immutable(False)
        merge_cfg_from_file(cfg_file)
        cfg.NUM_GPUS = 1
        cfg.MODEL.RPN_ONLY = True
        # cfg.TEST.RPN_PRE_NMS_TOP_N = 10000
        # cfg.TEST.RPN_POST_NMS_TOP_N = 2000
        cfg.TEST.RPN_PRE_NMS_TOP_N = 500
        cfg.TEST.RPN_POST_NMS_TOP_N = 300
        assert_and_infer_cfg(cache_urls=False)

        self.model = model_engine.initialize_model_from_cfg(weights, gpu_id=gpu_id)

    def detect(self, img):
        cls_boxes = [[] for _ in range(self.class_num)]
        im = cv2.imread(img)
        with c2_utils.NamedCudaScope(0):
            proposal_boxes, _proposal_scores = rpn_engine.im_proposals(self.model, im)
            # workspace.ResetWorkspace()
        # stack bbox and score like [x1, y1, x2, y2, score]
        dets_j = np.hstack((proposal_boxes, _proposal_scores[:, np.newaxis])).astype(np.float32, copy=False)
        # bbox NMS
        keep = box_utils.nms(dets_j, self.NMS)
        cls_boxes[1] = dets_j[keep, :]
        # Limit to max_per_image detections **over all classes**
        if self.DETECTIONS_PER_IM > 0:
            image_scores = np.hstack([cls_boxes[j][:, -1] for j in range(1, self.class_num)])
            if len(image_scores) > self.DETECTIONS_PER_IM:
                image_thresh = np.sort(image_scores)[-self.DETECTIONS_PER_IM]
                for j in range(1, self.class_num):
                    keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                    cls_boxes[j] = cls_boxes[j][keep, :]
        boxes = np.vstack([cls_boxes[j] for j in range(1, self.class_num)])

        if (boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < self.thresh) and not False:
            return
        if boxes is None:
            sorted_inds = []  # avoid crash when 'boxes' is None
        else:
            # Display in largest to smallest order to reduce occlusion
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            sorted_inds = np.argsort(-areas)
        box_dict = []
        pic_struct = {}
        pic_struct['width'] = str(im.shape[1])
        pic_struct['height'] = str(im.shape[0])
        pic_struct['depth'] = str(im.shape[2])
        box_dict.append(pic_struct)
        for i in sorted_inds:
            obj_struct = {}
            obj_struct['bbox'] = boxes[i, :4]
            score = boxes[i, -1]
            if score < self.thresh:
                continue
            obj_struct['name'] = self.classs
            obj_struct['score'] = score
            box_dict.append(obj_struct)
        return box_dict


def main():

    # # for one image
    # img = '/home/train/桌面/heiren/vis/4403172.jpg'
    # cfg_file = '/data/code/Detectron-Cascade-RCNN/OUTPUT_DIR/skuDethj/skuDet_rpn_R50_FPN_2x_190425/skuDet_rpn_R50_FPN_2x_190425.yaml'
    # weights = '/data/code/Detectron-Cascade-RCNN/OUTPUT_DIR/skuDethj/skuDet_rpn_R50_FPN_2x_190425/model_final.pkl'
    # de = RPN_INFERENCE(cfg_file,weights)
    # print(de.detect(img))

    cfg_file = '/data/code/Detectron-Cascade-RCNN/OUTPUT_DIR/skuDethj/skuDet_rpn_R50_FPN_2x_190425/skuDet_rpn_R50_FPN_2x_190425.yaml'
    weights = '/data/code/Detectron-Cascade-RCNN/OUTPUT_DIR/skuDethj/skuDet_rpn_R50_FPN_2x_190425/model_final.pkl'
    RPN_Model = RPN_INFERENCE(cfg_file,weights)

    rootDir = '/home/train/桌面/heiren/JPEGImages'
    rootDirList = os.listdir(rootDir)

    for idx, nname in enumerate(rootDirList):
        imgPath = os.path.join(rootDir, nname)

        import time
        sart_ = time.time()
        print(RPN_Model.detect(imgPath))
        print('inference time: {}'.format(time.time()-sart_))

if __name__ == '__main__':
    main()