#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import yaml
import cv2
import traceback
# sys.path.append('./utils/')
from utils.cython_bbox import bbox_overlaps

from detectron.core.config import cfg
import detectron.utils.c2 as c2_utils
import detectron.utils.boxes as box_utils
import detectron.core.test_engine as model_engine
import detectron.core.rpn_generator as rpn_engine
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import assert_and_infer_cfg
c2_utils.import_detectron_ops()
cv2.ocl.setUseOpenCL(False)
color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

class Detection_net(object):
    def __init__(self, cfg_file, caffemodel, GPUID, crop_area=None):

        self.NMS = 0.1
        self.class_num = 2
        self.DETECTIONS_PER_IM = 100
        self.crop_area = crop_area
        self.conf = 0.998

        cfg.immutable(False)
        merge_cfg_from_file(cfg_file)
        cfg.NUM_GPUS = 1
        assert_and_infer_cfg(cache_urls=False)

        self.model = model_engine.initialize_model_from_cfg(caffemodel, gpu_id=GPUID)

    def crop_detect_image(self, image, crop_area=None):
        if not crop_area:
            crop_area = self.crop_area
        print("crop_area: {}".format(crop_area))
        xmin = crop_area[0]
        ymin = crop_area[1]
        xmax = crop_area[2]
        ymax = crop_area[3]
        return image[ymin:ymax, xmin:xmax]


    def detect_inference(self, im):
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

        if (boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < self.conf) and not False:
            return
        if boxes is None:
            sorted_inds = []  # avoid crash when 'boxes' is None
        else:
            # Display in largest to smallest order to reduce occlusion
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            sorted_inds = np.argsort(-areas)

        box_dict = []
        for i in sorted_inds:
            obj_struct = {}
            obj_struct['bbox'] = boxes[i, :4]
            score = boxes[i, -1]
            if score < self.conf:
                continue
            obj_struct['score'] = score
            box_dict.append(obj_struct)
        print('this fram detect {} object'.format(len(box_dict)))
        return box_dict

    # from imagedt.decorator import time_cost
    # @time_cost
    def detect(self, frame, min_max_area, crop_area=None):

        print '*'*20, self.crop_area
        # 目标检测
        if crop_area is not None:
            self.crop_area = crop_area
        if self.crop_area is not None:
            frame = self.crop_detect_image(frame, crop_area=crop_area)
        try:
            dict_det = []
            h, w, c = frame.shape
            # TODO  is NoNE ?
            d_det = self.detect_inference(frame)

            for idx, _bbox in enumerate(d_det):
                _det = {}
                [xmin, ymin, xmax, ymax] = _bbox['bbox']
                # from crop_area Mapping back to the original img
                xmin = xmin + crop_area[0]
                xmax = xmax + crop_area[0]
                ymin = ymin + crop_area[1]
                ymax = ymax + crop_area[1]

                if min_max_area:
                    this_area = (xmax-xmin) * (ymax-ymin)
                    if this_area < min_max_area[0] or this_area > min_max_area[1]:
                        continue
                # 防止越界
                xmin = max(xmin, 0+self.crop_area[0])
                ymin = max(ymin, 0+self.crop_area[1])
                xmax = min(xmax, w+self.crop_area[0])
                ymax = min(ymax, h+self.crop_area[1])

                _det['bbox'] = [xmin, ymin, xmax, ymax]
                _det['take_tag'] = 0
                dict_det.append(_det)
            return self.sorted_out_Infos(dict_det)
        except:
            traceback.print_exc()

    def sorted_out_Infos(self, out):
        global SHELF_LAYER
        shelf_layer = SHELF_LAYER
        #  对out进行排序
        if out:
            by_min = min([item['bbox'][1] for item in out])
            by_max = max([item['bbox'][3] for item in out])
            per_layer = (float(by_max) - float(by_min)) / shelf_layer
            out_layer = [[] for i in range(shelf_layer)]
            for item in out:
                bbox = map(float, item['bbox'])
                y_mid = bbox[1] + (bbox[3] - bbox[1]) / 2
                for layer in range(shelf_layer):
                    this_layer = layer + 1
                    if y_mid > per_layer*(this_layer-1) + by_min and  y_mid < per_layer*this_layer + by_min:
                        item['layer'] = this_layer
                        out_layer[layer].append(item)

            # 横向排序
            for index, item in enumerate(out_layer):
                out_layer[index] = sorted(item, key=lambda x: float(x['bbox'][0]))
            out = [x for item in out_layer for x in item]
        return out

class Detecter(object):
    def __init__(self, model_name=None, crop_area=None, sku_num =24, shelf_layer=3, filter_threshold=0.5):
        super(Detecter, self).__init__()
        self._config_path = self._absolute_path('model_config/mod_config.yaml')
        self.model_name = model_name or 'detsku_rpn50'
        self.crop_area = crop_area # [xmin,ymin,xmax,ymax]
        self._load()
        self.get_model()
        self.template = []
        self.min_max_area = []
        self.template_list = []
        self.TEMPLATE_BBOX_NUM = sku_num
        self.FILTER_THRESHOLD = filter_threshold
        self.infos_image = {'image':[], 'infos':[]}
        global SHELF_LAYER
        SHELF_LAYER = shelf_layer

    def _absolute_path(self, path):
        return os.path.join(os.path.dirname(__file__), path)

    def _load(self):
        with open(self._config_path, 'r') as f:
            self.model_conf = yaml.load(f)[self.model_name]

    def get_model(self):
        model_root = self.model_conf['root_dir']
        prototxt = os.path.join(model_root, self.model_conf['prototxt'])
        caffemodel = os.path.join(model_root, self.model_conf['caffemodel'])
        GPUID = self.model_conf['GPUID']
        self.detect_net_runner = Detection_net(prototxt, caffemodel, GPUID, self.crop_area)

    def detect(self, img, crop_area=None):
        out= self.detect_net_runner.detect(img, self.min_max_area, crop_area=crop_area)
        return out

    def overlap(self, list1, list2):
        overlaps = bbox_overlaps(
            np.ascontiguousarray(list1, dtype=np.float),
            np.ascontiguousarray(list2, dtype=np.float))
        detected = []
        max_overlaps = np.zeros(len(list2))
        for ite in range(len(list1)):
            tem_ov = overlaps[ite, :]
            max_idx = np.argmax(tem_ov)
            # max_idx = max_idx if tem_ov[max_idx] == 0.0 else -1
            if max_idx not in detected:
                detected.append(max_idx)
                max_ov = np.max(tem_ov)
                max_overlaps[max_idx] = max_ov
            else:
                max_overlaps[max_idx] = 0.0
        return max_overlaps

    def origin_overlap(self, list1, list2):
        overlaps = bbox_overlaps(
            np.ascontiguousarray(list1, dtype=np.float),
            np.ascontiguousarray(list2, dtype=np.float))
        ov = overlaps.max(axis=0)
        return ov

    # calculate tempalte bbox area
    def calcu_area(self, list):
        list_area = []
        for ite in list:
            xmin, ymin, xmax, ymax = ite['bbox'][0], ite['bbox'][1], ite['bbox'][2], ite['bbox'][3]
            this_area = (ymax-ymin) * (xmax-xmin)
            list_area.append(this_area)

        min_area = min(list_area) * 0.9
        max_area = max(list_area) * 1.1
        self.min_max_area = [min_area, max_area]

    def filter_outlist(self, out_list):

        if self.template and out_list:
            template_list = [item['bbox'] for item in self.template]
            current_list = [item['bbox'] for item in out_list]
            overlaps = bbox_overlaps(
                np.ascontiguousarray(template_list, dtype=np.float),
                np.ascontiguousarray(current_list, dtype=np.float))
            ov = overlaps.max(axis=0)
            # ov = self.overlap(template_list, current_list)
            del_index_lists = np.where(ov < self.FILTER_THRESHOLD)[0]
            out_list = [ite for index, ite in enumerate(out_list) if index not in del_index_lists]

        return out_list

    # load ten frame, calculate frame template
    def calcu_template_frame(self, out_list):
        template = self.template
        template_list = self.template_list

        if len(template) != 1 and len(out_list):
            # TODO  模板帧平均
            # 如果模板list也为空，既是当前帧为第一帧
            if not len(template_list):
                template_list.append(out_list)
                self.template_list = template_list
                return

            # 接下来判断每一帧与前一帧的iou（过滤误检）
            if len(template_list) < 10:
                template_list.append(out_list)
                self.template_list = template_list

                # 如果模板list等于10帧，则开始计算平均帧
                if len(template_list) == 10:
                    # 求10帧和
                    max_list_len = 0
                    max_index = 0
                    for index, item_list in enumerate(template_list):
                        if len(item_list) > max_list_len:
                            max_list_len = len(item_list)
                            max_index = index

                    template = template_list[max_index]
                    tag_ind = np.zeros(len(template)) + 1
                    maxlen_temlist = [item['bbox'] for item in template]

                    for ind, item in enumerate(template_list):
                        if ind != max_index:
                            for bbox in item:
                                box = bbox['bbox']
                                ov = self.origin_overlap([box], maxlen_temlist)
                                # TODO iou 过滤误检,不考虑误检
                                tag = np.argmax(ov)
                                tag_ind[tag] += 1
                                template[tag]['bbox'] = np.array(template[tag]['bbox']) + np.array(box)

                    # 求10帧平均
                    for ind, ite in enumerate(template):
                        print '#' * 20, ite['bbox'], int(tag_ind[ind]), type(ite['bbox'])
                        # TODO type(ite['bbox']) 可能为list，无法进行:/
                        if isinstance(ite['bbox'], list):
                            ite['bbox'] = np.array(ite['bbox'])
                        ite['bbox'] /= int(tag_ind[ind])
                        ite['take_tag'] = 0  # [0:static , 1:take, 3:put back]
                    # template.append(template)
                    self.template = template
                    self.calcu_area(template)
                    self.template_list = []    # def calcu_take_tag

    def calcu_take_tag(self, out_list):
        template = self.template
        if not out_list or not template:
            return out_list

        #  如果当前帧等于模板帧，清空chang_index
        if len(out_list) == len(template):
            # out_list[0]['take']: [0:static , 1:take, 3:put_back]
            # TODO  静止状态和放回状态判断
            take_tag = [item['take_tag'] for item in template]
            # case 1: take --> put_back
            if 1 in take_tag:
                take_index = np.where(np.array(take_tag)==1)[0]
                for take_ite in take_index:
                    template[take_ite]['take_tag'] = 0

        # 如果当前帧长度与模板帧长度不同，进行拿走sku判断，记录拿走sku的chang_index
        else:
            # len(out_list) != len(template):
            out_list_infos = [item['bbox'] for item in out_list]
            template_infos = [item['bbox'] for item in template]
            ov = self.overlap(out_list_infos, template_infos)
            change_index = np.where(ov <= 0.01)[0].tolist()
            for change_ind in change_index:
                template[change_ind]['take_tag'] = 1
                inser_list = template[change_ind]
                out_list.insert(change_ind, inser_list)

        self.template = template
        return out_list
