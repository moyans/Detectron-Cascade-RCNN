#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 18-11-27 下午7:40
# @Author : Moyan
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import cv2
import sys
import __init__
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
from caffe2.python import workspace
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
import detectron.core.test_engine as infer_engine
import detectron.utils.c2 as c2_utils

from pricetag_api import getPriceApi
from BASE_TOOLS import walkDir2RealPathList, writeXml, pathExit

c2_utils.import_detectron_ops()
cv2.ocl.setUseOpenCL(False)




def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes

class mycaffe2(object):
    def __init__(self, cfg_file, weights, gpu_id=0):
        self.gpu_id = gpu_id
        self.thresh = 0.5
        self.classs = '1'
        workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
        merge_cfg_from_file(cfg_file)
        cfg.NUM_GPUS = 1
        assert_and_infer_cfg(cache_urls=False)
        self.model = infer_engine.initialize_model_from_cfg(weights, gpu_id)

    def detect(self,img):
        im = cv2.imread(img)
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(self.model, im, None)

        if isinstance(cls_boxes, list):
            boxes, segms, keypoints, classes = convert_from_cls_format(
                cls_boxes, cls_segms, cls_keyps)
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
            bbox = boxes[i, :4]
            bbox_int = bbox.astype(np.int)

            score = boxes[i, -1]
            if score < self.thresh:
                continue
            obj_struct['name'] = self.classs  # class： 3477
            obj_struct['score'] = score

            pricetag_area = im[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2], :]
            # from priceApi get price

            # import time
            # st = time.time()

            pre_price = getPriceApi(pricetag_area)

            # print('time cost: {}'.format(time.time()-st))
            # print("####========>pre_price:", pre_price)

            if pre_price == '':
                pre_price = -1
            obj_struct['price'] = str(pre_price)

            box_dict.append(obj_struct)

        return box_dict

if __name__ == '__main__':


    # # singe u image
    # img = '/home/train/桌面/黑人/testimg/0aeefa824658bd32318471f12d24b682.jpg'
    # cfg_file = '/data/code/Detectron-Cascade-RCNN/OUTPUT_DIR/ONLINE/26/config.yaml'
    # weights = '/data/code/Detectron-Cascade-RCNN/OUTPUT_DIR/ONLINE/26/model_iter29999.pkl'
    # de = mycaffe2(cfg_file,weights)
    # detlist = de.detect(img)
    # print(de.detect(img))


    # det with images

    testDir = '/data/Data/百事/pricetag/Data191112/train/'
    test_imgDir = os.path.join(testDir, 'JPEGImages')
    outputDir = os.path.join(testDir, 'Annotations')
    pathExit(outputDir)

    # # old
    # cfg_file = '/data/code/Detectron-Cascade-RCNN/OUTPUT_DIR/priceTagDet/priceTagDet_faster_rcnn_R50_FPN_2x_Data02_190306_G4train15/priceTagDet_faster_rcnn_R50_FPN_2x_Data02_190306.yaml'
    # weights = '/data/code/Detectron-Cascade-RCNN/OUTPUT_DIR/priceTagDet/priceTagDet_faster_rcnn_R50_FPN_2x_Data02_190306_G4train15/model_final.pkl'

    # new1
    cfg_file = '/data/code/Detectron-Cascade-RCNN/OUTPUT_DIR/priceTagDet/priceTagDet_faster_rcnn_R50_FPN_2x_4g_ms_190410/priceTagDet_faster_rcnn_R50_FPN_2x_4g_ms_190410.yaml'
    weights = '/data/code/Detectron-Cascade-RCNN/OUTPUT_DIR/priceTagDet/priceTagDet_faster_rcnn_R50_FPN_2x_4g_ms_190410/model_final.pkl'


    de = mycaffe2(cfg_file, weights)
    imgs = walkDir2RealPathList(test_imgDir)
    for idx, img_path in enumerate(imgs):
        print(idx, img_path)
        imgname = img_path.strip().split('/')[-1]
        outxml = os.path.join(outputDir, imgname.strip().replace('.jpg', '.xml'))
        detlist = de.detect(img_path)
        if detlist is not  None:
            # print(detlist)
            writeXml(detlist, imgname, outxml, write_price=True)