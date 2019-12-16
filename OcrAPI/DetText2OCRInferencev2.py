#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 19-07-25

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import sys
import os
import math
import mmcv
import numpy as np
from caffe2.python import workspace
import pycocotools.mask as maskUtils
import detectron.utils.c2 as c2_utils
from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import merge_cfg_from_file
import detectron.core.test_engine as infer_engine
from detectron.core.config import get_cfg, merge_cfg_from_cfg, get_clean_cfg

from ocrAPI import getRecApi
import imagedt

reload(sys)
sys.setdefaultencoding('utf-8')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import __init__
from BASE_TOOLS import walkDir2RealPathList, writeXml, pathExit

c2_utils.import_detectron_ops()
cv2.ocl.setUseOpenCL(False)


def prep_im(im, target_size=1000, max_size=1333):
    im = im.astype(np.float32, copy=False)
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)

    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(
        im,
        None,
        None,
        fx=im_scale,
        fy=im_scale,
        interpolation=cv2.INTER_LINEAR
    )
    return im, im_scale

# @imagedt.decorator.time_cost
def sort_rectangle(poly, support_Vertical=True):
    # sort the four coordinates of the polygon, points in poly should be sorted clockwise
    # First find the lowest point
    # Angle is in [-90, 90]. #support_Vertical=True [-45, 45] for MLT dataset
    # If h is close to w, we can't determine truth_angle, ignore it during training.
    angle_train = True
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(abs(poly[:, 1]-poly[p_lowest, 1]) <= 0.001) == 2:
        # 底边平行于X轴, 那么p0为左上角
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4

        # 竖直 而且很长， 认为是90’.
        if (poly[p3_index, 1] - poly[p0_index, 1]) / (poly[p1_index, 0] - poly[p0_index, 0]) > 3 and support_Vertical == False:
            angle = np.pi/2
            return poly[[p3_index, p0_index, p1_index, p2_index]], angle
        else:
            return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        # 找到最低点右边的点
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left = (p_lowest + 1) % 4
        angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right]
                            [1])/(poly[p_lowest][0] - poly[p_lowest_right][0]))
        right_edge = np.linalg.norm(poly[p_lowest] - poly[p_lowest_right])
        left_edge = np.linalg.norm(poly[p_lowest] - poly[p_lowest_left])
        right_left_ratio = right_edge / (left_edge + 1e-4)

        # assert angle > 0
        if angle <= 0:
         #       print(angle, poly)
            angle = 0
        if (support_Vertical == True) or (right_left_ratio < 3 and right_left_ratio > 1.0/3.0):
            # Text region isn't very long, hard to determine its orientation. Trust that most text regions have a small angle(-45, 45).
            if angle/np.pi * 180 > 45:
                # 这个点为p2
                p2_index = p_lowest
                p1_index = (p2_index - 1) % 4
                p0_index = (p2_index - 2) % 4
                p3_index = (p2_index + 1) % 4
                return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi/2 - angle)
            else:
                # 这个点为p3
                p3_index = p_lowest
                p0_index = (p3_index + 1) % 4
                p1_index = (p3_index + 2) % 4
                p2_index = (p3_index + 3) % 4
                return poly[[p0_index, p1_index, p2_index, p3_index]], angle
        else:
            if right_edge > left_edge:
                p3_index = p_lowest
                p0_index = (p3_index + 1) % 4
                p1_index = (p3_index + 2) % 4
                p2_index = (p3_index + 3) % 4
                # , angle_train
                return poly[[p0_index, p1_index, p2_index, p3_index]], angle
            if right_edge < left_edge:
                angle = - (np.pi / 2 - angle)
                p2_index = p_lowest
                p1_index = (p2_index - 1) % 4
                p0_index = (p2_index - 2) % 4
                p3_index = (p2_index + 1) % 4
                # , angle_train
                return poly[[p0_index, p1_index, p2_index, p3_index]], angle

# @imagedt.decorator.time_cost
def extract_image_from_corners(img_in, minRect):
    rect = minRect.reshape((4, 2))
    rectange, rotate_angle = sort_rectangle(rect, False)
    p0_rect, p1_rect, p2_rect, p3_rect = rectange
    lt = p0_rect
    rt = p1_rect
    lb = p3_rect
    rb = p2_rect

    template_width = (rt[1] - lt[1]) * (rt[1] - lt[1]) + \
        (rt[0] - lt[0]) * (rt[0] - lt[0])
    template_height = (rb[1] - rt[1]) * (rb[1] - rt[1]) + \
        (rt[0] - rb[0]) * (rt[0] - rb[0])
    template_width = int(math.sqrt(template_width))
    template_height = int(math.sqrt(template_height))

    pts1 = np.float32([lt, rt, rb, lb])
    template_pts = np.float32([[0, 0], [template_width, 0], [
                              template_width, template_height], [0, template_height]])

    M = cv2.getPerspectiveTransform(pts1, template_pts)
    res = cv2.warpPerspective(img_in, M, (template_width, template_height))
    return res

@imagedt.decorator.time_cost
def genarate_rorateRect(img, bbox_result, segm_result, dataset='coco', score_thr=0.5):
    #class_names = get_classes(dataset)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    bboxes = np.vstack(bbox_result)
    # segm_result = [seg['counts'] for seg in  segm_result]
    segms = mmcv.concat_list(segm_result)
    inds = np.where(bboxes[:, -1] > score_thr)[0]

    rorate_obj = np.zeros((9,), np.float64)
    rorate_rect = []
    new_im = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # import time
    # st = time.time()
    for i in inds:
        mask = maskUtils.decode(segms[i]).astype(np.bool)
        img_mask = (maskUtils.decode(segms[i])).astype(np.uint8)
        img_mask = cv2.erode(img_mask, element)
        contours, _ = cv2.findContours(
            img_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area = []
        for cnt in contours:
            area.append(cv2.contourArea(cnt))
        if len(area) == 0:
            continue

        idx = area.index(max(area))
        # merge result
        new_im = cv2.drawContours(
            new_im, [contours[idx]], 0, 255, thickness=-1)
    
    # et = time.time()
    # print("part1 use time: {:.3f}".format(et-st))

    # st = time.time()
    minAreaRect = []
    contours, _ = cv2.findContours(
        new_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        new_im = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        new_im = cv2.drawContours(new_im, [cnt], 0, 255, thickness=-1)
        new_im = cv2.dilate(new_im, element)
        contours, _ = cv2.findContours(
            new_im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(contours[0])

        # expand and filter small box
        w, h = rect[1]
        expand = 3
        w, h = w + expand, h + expand
        if h < 9:
            continue
        cx, cy = rect[0]
        angle = rect[2]
        new_rect = ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(new_rect)
        minAreaRect.append(box)

    # et = time.time()
    # print("part2 use time: {:.3f}".format(et-st))
    return minAreaRect

# @imagedt.decorator.time_cost
def countors2rect(minRect):
    ''' 轮廓转矩形'''
    if isinstance(minRect, list):
        bboxlist = []
        for box in minRect:
            x, y, w, h = cv2.boundingRect(box)
            bboxlist.append(np.array([x, y, w, h]))
        return bboxlist
    elif isinstance(minRect, np.ndarray):
        x, y, w, h = cv2.boundingRect(minRect)
        return np.array([x, y, w, h])
    else:
        raise TypeError('minRect only accept list or np.ndarray')


class mycaffe2(object):

    def __init__(self, cfg_file, weights, classs='99999', thresh=0.5, gpu_id=0):
        self.infer_engine = infer_engine
        self.gpu_id = gpu_id
        self.thresh = thresh
        self.classs = classs
        self.clean_cfg = get_clean_cfg()
        merge_cfg_from_cfg(self.clean_cfg)
        merge_cfg_from_file(cfg_file)
        assert_and_infer_cfg(cache_urls=False, make_immutable=False)
        self.model = infer_engine.initialize_model_from_cfg(weights, gpu_id)
        self.cfg = get_cfg()
        # self.detbbox = []
        self.maskmat = []

    @imagedt.decorator.time_cost
    def ocrdet(self, im):
        # im = cv2.imread(img)
        # im = 128 * np.ones((300, 400, 3), dtype=np.uint8) # test 
        h, w, c = im.shape
        with c2_utils.NamedCudaScope(self.gpu_id):
            merge_cfg_from_cfg(self.cfg)
            cls_boxes, cls_segms, cls_keyps = self.infer_engine.im_detect_all(
                self.model, im, None)

        if cls_segms is None:
            return

        minAreaRect = genarate_rorateRect(
            im, cls_boxes[1:], cls_segms[1:], score_thr=self.thresh)
        num_crop = len(minAreaRect)
        new_min_area_rect = []
        crop_img_list = []
        for ci in range(num_crop):
            crop_img = extract_image_from_corners(im, minAreaRect[ci])
            # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
            if crop_img.shape[1] < 10 or crop_img.shape[0] < 10:
                continue
            new_min_area_rect.append(minAreaRect[ci])
            crop_img_list.append(crop_img)
        bboxs_list = countors2rect(new_min_area_rect)
        assert len(bboxs_list) == len(crop_img_list)

        # self.bbox = bboxs_list
        self.maskmat = crop_img_list

        box_dict = []
        pic_struct = {}
        pic_struct['width'] = str(w)
        pic_struct['height'] = str(h)
        pic_struct['depth'] = str(c)
        box_dict.append(pic_struct)
        for ind in range(len(bboxs_list)):
            obj_struct = {}
            x1, y1, w, h = bboxs_list[ind]
            x2 = int(x1) + int(w)
            y2 = int(y1) + int(h)
            obj_struct['bbox'] = [x1, y1, x2, y2]
            obj_struct['name'] = self.classs
            box_dict.append(obj_struct)

        return box_dict

    @imagedt.decorator.time_cost
    def ocrrec(self, sort_keys=[]):
        orcrec = []
        if sort_keys:
            for i in sort_keys:
                orcrec.append(getRecApi(self.maskmat[i]))
        else:
            for i in range(len(self.maskmat)):
                orcrec.append(getRecApi(self.maskmat[i]))
        return orcrec

if __name__ == '__main__':

    cfg_file = '/data/code/sunchao/Detectron-Cascade-RCNN/OUTPUT_DIR/text/text_mask_cascade_rcnn_R-101-FPN_multiscale_2x_191128/config.yaml'
    weights = '/data/code/sunchao/Detectron-Cascade-RCNN/OUTPUT_DIR/text/text_mask_cascade_rcnn_R-101-FPN_multiscale_2x_191128/model_final.pkl'
    models = mycaffe2(cfg_file, weights, gpu_id=1)

    import os
    rootDir = '/home/train/Desktop/demo/JPEGImages'
    outXmlDir = rootDir.strip().replace('/JPEGImages', '/Annotations')
    rootDirList = os.listdir(rootDir)

    for nname in rootDirList:
        print("loading {} :".format(nname))
        imgPath = os.path.join(rootDir, nname)
        xmlPath = os.path.join(outXmlDir, os.path.splitext(nname)[0]+'.xml')

        im = cv2.imread(imgPath)
        h, w, c = im.shape
        im, im_scale = prep_im(im)

        import time
        start = time.time()
        bbox_result = models.ocrdet(im)
        ocr_result = models.ocrrec()
        end = time.time()
        print("all use time: {}".format(end - start))

        print("len bbox: {}".format(len(bbox_result)))

        bbox_result[0]['width'] = str(w)
        bbox_result[0]['height'] = str(h)
        bbox_result[0]['depth'] = str(c)
        for ind in range(1, len(bbox_result)):
            bbox_result[ind]['bbox'] = np.array(bbox_result[ind]['bbox']) / im_scale
            bbox_result[ind]['text_ocr'] = ocr_result[ind-1]

        # print(bbox_result)
        # writeXml(bbox_result, nname, xmlPath, write_text=True)