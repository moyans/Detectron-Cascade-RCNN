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
import time

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
from featurematchingfindobjects import FMFO
c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

# draw colors
color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

class RPN_INFERENCE(object):
    def __init__(self, cfg_file, weights, gpu_id=0, crop_area=None):
        self.NMS = 0.1
        self.thresh = 0.998
        self.classs = '3477'
        self.class_num = 2
        self.DETECTIONS_PER_IM = 1000
        self.crop_area = crop_area

        cfg.immutable(False)
        merge_cfg_from_file(cfg_file)
        cfg.NUM_GPUS = 1
        cfg.MODEL.RPN_ONLY = True
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

    def crop_detect_image(self, image, crop_area=None):
        if not crop_area:
            crop_area = self.crop_area
        print("crop_area: {}".format(crop_area))
        xmin = crop_area[0]
        ymin = crop_area[1]
        xmax = crop_area[2]
        ymax = crop_area[3]
        return image[ymin:ymax, xmin:xmax]

    def detect_im(self, im, crop_area=None):

        if crop_area is not None:
            self.crop_area = crop_area
        if self.crop_area is not None:
            im = self.crop_detect_image(im, crop_area=crop_area)

        cls_boxes = [[] for _ in range(self.class_num)]
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

cc = (0, 255, 0)
def draw_detect_rect(frame, out_list, shelf_area=None):
    if not out_list:
       return frame




    for ind, out_infos in enumerate(out_list):
        if ind == 0:
            continue
        bbox = out_infos['bbox']
        if shelf_area is not None:
            [x1, y1, x2, y2] = bbox
            x1 = x1 + shelf_area[0]
            y1 = y1 + shelf_area[1]
            x2 = x2 + shelf_area[0]
            y2 = y2 + shelf_area[1]
            bbox = [x1, y1, x2, y2]

        [x1, y1, x2, y2] = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        # cv2.rectangle(frame, (x1, y1), (x2, y2), cc, 2)
        box_h = int((y2-y1)/10)
        box_w = int((x2-x1)/5)
        cv2.line(frame, (x1, y1), (x1+box_w, y1), cc, thickness=2, lineType=8, shift=0)
        cv2.line(frame, (x1, y1), (x1, y1+box_h), cc, thickness=2, lineType=8, shift=0)
        cv2.line(frame, (x2, y1), (x2-box_w, y1), cc, thickness=2, lineType=8, shift=0)
        cv2.line(frame, (x2, y1), (x2, y1+box_h), cc, thickness=2, lineType=8, shift=0)
        cv2.line(frame, (x1, y2), (x1+box_w, y2), cc, thickness=2, lineType=8, shift=0)
        cv2.line(frame, (x1, y2), (x1, y2-box_h), cc, thickness=2, lineType=8, shift=0)
        cv2.line(frame, (x2, y2), (x2-box_w, y2), cc, thickness=2, lineType=8, shift=0)
        cv2.line(frame, (x2, y2), (x2, y2-box_h), cc, thickness=2, lineType=8, shift=0)
    return frame

def main():

    cfg_file = '/data/code/Detectron-Cascade-RCNN/OUTPUT_DIR/skuDethj/skuDet_rpn_R50_FPN_2x_190425/skuDet_rpn_R50_FPN_2x_190425.yaml'
    weights = '/data/code/Detectron-Cascade-RCNN/OUTPUT_DIR/skuDethj/skuDet_rpn_R50_FPN_2x_190425/model_final.pkl'
    RPN_Model = RPN_INFERENCE(cfg_file,weights)


    MJPG = 1196444237.0
    capture = cv2.VideoCapture('/data/code/AI_Congress_brk/idt-candies-realtime-recognize/video/2018-09-10-183454.webm')
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    capture.set(cv2.CAP_PROP_FOURCC, MJPG)
    while not capture.isOpened():
        print('Waiting for camera to exist...')
        time.sleep(10)
    print(capture.isOpened())
    success, frame = capture.read()


    rset_xmin, rset_ymin, rset_xmax, rset_ymax = 400, 0, 1250, 1080

    shelf_area = [rset_xmin, rset_ymin, rset_xmax, rset_ymax]



    template_img = cv2.imread('./shelf_template.png')
    #初始化
    fmfo = FMFO(height=320, min_feature_num=20, min_match_num=10, method='orb')
    fmfo.set_general_parameters(height=320, min_feature_num=20, min_match_num=10, method='sift')
    fmfo.set_sift_parameters(nfeatures=300, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    #设置模板图片
    fmfo.set_template_pic(template_img)

    clock = 0
    while success:
        success, frame = capture.read()
        print(frame.shape)
        clock += 1
        # 缓冲10帧q
        if clock < 10:
            continue
        if clock % 50 == 0:
            corners = fmfo.find_object(frame=frame, display=False)
            if corners is not None:
                corners = np.squeeze(corners)
                xmin = np.min(corners[:, 0])
                ymin = np.min(corners[:, 1])
                xmax = np.max(corners[:, 0])
                ymax = np.max(corners[:, 1])
                # shelf_area = map(int, [xmin,ymin,xmax,ymax])
                shelf_area = map(int, [xmin, 0, xmin + 830, 1080])
                # detecter.crop_area = shelf_area
                print('#' * 20, shelf_area)

        out_list = RPN_Model.detect_im(frame, crop_area=shelf_area)

        frame = draw_detect_rect(frame, out_list, shelf_area)
        # print(out_list)

        cv2.rectangle(frame, (shelf_area[0],shelf_area[1]), (shelf_area[2],shelf_area[3]), color=(255,0,0), thickness=3)

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()
        success, frame = capture.read()
    capture.release()







if __name__ == '__main__':
    main()