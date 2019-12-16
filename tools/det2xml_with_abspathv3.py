#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 18-11-27 下午7:40
# @Author : Moyan
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#dummy_datasets_moyan.py

from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import argparse
import detectron.utils.c2 as c2_utils
import detectron.core.test_engine as infer_engine
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import cfg
from detectron.core.config import assert_and_infer_cfg
from caffe2.python import workspace
import numpy as np
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import os
import sys
import pexif
import traceback
import math
import detectron.datasets.dummy_datasets as dummy_datasets


#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

reload(sys)
sys.setdefaultencoding('utf-8')

c2_utils.import_detectron_ops()
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Detectron api, Input test_imgs Dir, Output xml')
    parser.add_argument(
        '--m',
        dest='model',
        help='detect model.pkl',
        default='model.pkl',
        type=str
    )
    parser.add_argument(
        '--c',
        dest='cfg_file',
        help='test config.yaml',
        default='config.yaml',
        type=str
    )
    parser.add_argument(
        '--t',
        dest='test_dir',
        help='test img root',
        default='demo',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='bbox thresh',
        default=0.5,
        type=float
    )

    parser.add_argument(
        '--gpu_id',
        dest='gpu_id',
        help='gpu_id',
        default=0,
        type=str
    )
    return parser.parse_args()

filter_postfix=[".jpg", ".JPG", "PNG", ".png", ".jpeg"]

def rotate(src, angle, min_edge=None):
    w = src.shape[1]
    h = src.shape[0]

    scale = 1.
    if min_edge is not None and min(w, h) > min_edge:
        scale = min_edge * 1.0 / min(w, h)

    rangle = np.deg2rad(angle)
    # now calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]

    img = cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

    return img

def rotateExif(input_file, min_edge=None):
    # if not cv2.__version__.startswith('2'):
    #     print 'incapable cv2 version!!!'
    #     sys.exit()
    orientation = 1

    try:
        img = pexif.JpegFile.fromFile(input_file)
        # Get the orientation if it exists
        orientation = img.exif.primary.Orientation[0]
    except:
        traceback.print_exc()

    # now rotate the image using the Python Image Library (PIL)
    if cv2.__version__.startswith('2'):
        img_ori = cv2.imread(input_file)
    else:
        img_ori = cv2.imread(input_file, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)

    img = img_ori.copy()
    # print("orientation: {}".format(orientation))
    if orientation is 6:
        img = rotate(img, -90, min_edge)
        cv2.imwrite(input_file, img)
    elif orientation is 8:
        img = rotate(img, 90, min_edge)
        cv2.imwrite(input_file, img)
    elif orientation is 3:
        img = rotate(img, 180, min_edge)
        cv2.imwrite(input_file, img)
    # save the result
    return img

def pathExit(path):
    if isinstance(path, list):
        for ipath in path:
            if not os.path.exists(ipath):
                os.makedirs(ipath)
    else:
        if not os.path.exists(path):
            os.makedirs(path)


# def walkDir2RealPathList(path, page):
#     root_lists = []
#     for fpathe, dirs, fs in os.walk(path):
#         # 返回的是一个三元tupple(dirpath, dirnames, filenames),
#         for k, f in enumerate(fs):
#             apath = os.path.join(fpathe, f)
#             if k == 0:
#                 lidpath = fpathe.strip().replace(page, 'Annotations/')
#                 pathExit(lidpath)
#             ext = os.path.splitext(apath)[1]
#             if ext in filter_postfix:
#                 root_lists.append(apath)
#     return root_lists

def walkDir2RealPathList(path, filter_postfix=[".jpg", ".JPG", "PNG", ".png", ".jpeg"]):
    root_lists = []
    filter_postfix = filter_postfix
    if filter_postfix:
        print("Files will be searched by the specified suffix, {}".format(filter_postfix))
    else:
        print("All files will be searched")

    for fpathe, dirs, fs in os.walk(path):
        # 返回的是一个三元tupple(dirpath, dirnames, filenames),
        for f in fs:
            # print(os.path.join(fpathe, f))
            apath = os.path.join(fpathe, f)
            ext = os.path.splitext(apath)[1]
            if filter_postfix:
                if ext in filter_postfix:
                    root_lists.append(apath)
            else:
                root_lists.append(apath)
    return root_lists


def get_class_string(class_index, dataset):
    class_text = dataset.classes[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text


def rewrite_xml_clean(bbox, img_name, xml_path):
    # [ { 'width': xx ; 'depth' : xx ; 'height': xx} ; {'name' : 'class_name' ; 'bbox' : [xmin ymin xmax ymax] }  ]
    node_root = Element('annotation')
    ####
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = img_name.decode('utf-8')
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = bbox[0]['width']
    node_height = SubElement(node_size, 'height')
    node_height.text = bbox[0]['height']
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    for i in range(1, len(bbox)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = str(bbox[i]['name'])
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_score = SubElement(node_object, 'score')
        node_score.text = str(float(bbox[i]['score']))
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(float(bbox[i]['bbox'][0]))
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(float(bbox[i]['bbox'][1]))
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(float(bbox[i]['bbox'][2]))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(float(bbox[i]['bbox'][3]))

    xml = tostring(node_root, pretty_print=False)  # 格式化显示，该换行的换行
    dom = parseString(xml)
    # print xml
    f = open(xml_path, 'w')
    dom.writexml(f, addindent='  ', newl='\n', encoding='utf-8')
    f.close()


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
    def __init__(self, cfg_file, weights, gpu_id=0, thresh_=0.5):
        self.gpu_id = gpu_id
        self.thresh = thresh_
        self.classs = dummy_datasets.get_idtSKU_dataset()
        workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
        merge_cfg_from_file(cfg_file)
        cfg.NUM_GPUS = 1
        assert_and_infer_cfg(cache_urls=False)
        self.model = infer_engine.initialize_model_from_cfg(weights, gpu_id)

    def detect(self, img):
        if cv2.imread(img) is None:
            return None
        else:
            im = rotateExif(img)

        with c2_utils.NamedCudaScope(self.gpu_id):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                self.model, im, None)

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
            score = boxes[i, -1]
            if score < self.thresh:
                continue
            obj_struct['name'] = get_class_string(classes[i], self.classs)
            obj_struct['score'] = score
            box_dict.append(obj_struct)

        return box_dict


if __name__ == '__main__':

    tag = '/gz-data/商品指纹数据/'
    pretag = '/project/商品指纹数据/'

    args = parse_args()
    GPU_ID = int(args.gpu_id)

    print("config file :{}".format(args.cfg_file))
    print("weights file :{}".format(args.model))
    print("det img path :{}".format(args.test_dir))

    assert os.path.exists(args.cfg_file) and os.path.exists(args.model) and os.path.exists(args.test_dir)

    model = mycaffe2(args.cfg_file, args.model, thresh_=args.thresh, gpu_id=GPU_ID)
    
    detList = walkDir2RealPathList(args.test_dir)
    print("{} imgs".format(len(detList)))
    for idx, imgPath in enumerate(detList):
        print('loading {},  name: {}'.format(idx, imgPath))
        postfix = os.path.splitext(imgPath)[1]
        name = imgPath.strip().split('/')[-1].replace(postfix, '')
        xmlPath = imgPath.strip().replace(tag, pretag).replace(postfix, '.xml')
        xmlDir = os.path.dirname(xmlPath)
        print(xmlDir)
        pathExit(xmlDir)
        print('output xml path {}'.format(xmlPath))

        predict_dict = model.detect(imgPath)
        if predict_dict:
            try:
                rewrite_xml_clean(predict_dict, name, xmlPath)
            except:
                pass

'''
export PYTHONIOENCODING=utf-8
'''
