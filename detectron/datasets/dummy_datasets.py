# coding:utf-8
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
"""Provide stub objects that can act as stand-in "dummy" datasets for simple use
cases, like getting all classes in a dataset. This exists so that demos can be
run without requiring users to download/install datasets first.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.utils.collections import AttrDict


def get_coco_dataset():
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_surveyPOSM_dataset():
    ds = AttrDict()
    classes = [
        '__background__', '17778', '17779', '17780'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_idtSKU_dataset():
    ds = AttrDict()
    classes = [
        '__background__', '3477'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_idtPrice_dataset():
    ds = AttrDict()
    classes = [
        '__background__', '1'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_ulposm_dataset():
    ds = AttrDict()
    classes = [
        '__background__', '9996', '15701', '15704', '9994', '9997', '9995', '15703', '15702', '3508'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_agposm191018_dataset():
    ds = AttrDict()
    classes = [
        '__background__', '19259', '19258', '19263', '19256', '19257', '19262', '19265', '19264', '19266', '19267'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

# def get_agposm191018_merge_dataset():
#     ds = AttrDict()
#     classes = [
#         '__background__', '19259', '19263', '19262', '19264', '19256', '19258', '19266', '19267', '19260'
#     ]
#     ds.classes = {i: name for i, name in enumerate(classes)}
#     return ds

# def get_agposm191018_merge_dataset():
#     ds = AttrDict()
#     classes = [
#         '__background__', '19259', '19263', '19262', '19264', '19256', '19258', '19266', '19267', '19260'
#     ]
#     ds.classes = {i: name for i, name in enumerate(classes)}
#     return ds

# def get_abbottposm191018_d1_dataset():
#     ds = AttrDict()
#     classes = [
#         # '__background__', '19294', '19279', '19286', '19285', '19272', '19273', '19274', '19275', '19277', '19297', '19276', '19298', '19293', '19296', '19283'
#         '__background__', '18669', '17913', '18666', '18666', '17903', '17904', '17904', '17904', '17907', '19268', '17907', '19270', '18669', '19268', '18658'
#     ]
#     ds.classes = {i: name for i, name in enumerate(classes)}
#     return ds
# def get_abbottposm191108_dataset():
#     ds = AttrDict()
#     classes = [
#         # '__background__', '19285', '19279', '19294', '19275', '19274', '19273', '19276', '19277', '19272', '19283', '19297', '19298', '19293', '19296'
#         '__background__', '18666', '17913', '18669', '17904', '17904', '17904', '17907', '17907', '17903', '18658', '19268', '19270', '18669', '19268'
#     ]
#     ds.classes = {i: name for i, name in enumerate(classes)}
#     return ds

def get_abbottposm191118_dataset():
    ds = AttrDict()
    classes = [
        # '__background__', '19285', '19279', '19294', '19275', '19274', '19273', '19276', '19277', '19272', '19283', '19297', '19298', '19293', '19296', '19299', '17278', '17277', '17315', '17314', '17274', '17273', '17275', '17276', '19287', '19290'
        '__background__', '18666', '17913', '18669', '17904', '17904', '17904', '17907', '17907', '17903', '18658', '19268', '19270', '18669', '19268', '19271', '17278', '17277', '17315', '17314', '17274', '17273', '17275', '17276', '18667', '18668'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds


def get_unilevelAgPosm191029_dataset():
    ds = AttrDict()
    classes = [
        '__background__', '18906', '18907', '18911', '18912', '18905', '18910', '18909', '19336', '18908', '18913', '18914'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds


def get_libyposm191024_dataset():
    ds = AttrDict()
    classes = [
        # '__background__', '18918', '19302', '18917', '19301', '19300'
         '__background__', '18918', '18918', '18917', '18919', '18919'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_gongniuposm_data():
    ds = AttrDict()
    classes = [
         '__background__', '19761', '19830', '19834', '19829', '19837', '19835', '19833', '19836', '19831', '19832'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_byhealthposm191212_data():
    ds = AttrDict()
    classes = [
        #  '__background__', '19916', '19915', '19914', '19917', '19913', '19918', '19920', '19925', '19922', '19924', '19919'
        #  '__background__', '19656', '19655', '19652', '19660', '19650', '19664', '19666', '19674', '19671', '19673', '19665'
        # '__background__', '19916', '19915', '19914', '19917', '19913', '19918', '19920', '19925', '19922', '19924', '19919', '19921', '19923'
        '__background__', '19656', '19655', '19652', '19660', '19650', '19664', '19666', '19674', '19671', '19673', '19665', '19669', '19672'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds   

def get_glicoposm191212_data():
    ds = AttrDict()
    classes = [
         '__background__', '19908', '19909', '19761_other', '19908_other', '19910', '19911', '19912'
        #  '__background__', '19908', '19909', '0', '0', '19910', '19911', '19912'    # online
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds 

def get_mzcposm191213_data():
    ds = AttrDict()
    classes = [
         '__background__','19309', '19311', '19310', '19312', '19308', '19303', '19316', '19327', '19313', '19317', '19326', '19318', '19304', '19315', '19314', '19333', '19332', '19320', '19305', '19329', '19306', '19331'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds  

def get_carlsbergposm200205_data():
    ds = AttrDict()
    classes = [
         '__background__','20378', '20379', '20326'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds  

# def get_tongy200224_data():
#     ds = AttrDict()
#     classes = [
#          '__background__', '19696', '19697', '19700', '19695', '19694', '19699', '21254', '21253'
#     ]
#     ds.classes = {i: name for i, name in enumerate(classes)}
#     return ds  

def get_tongy200306_data():
    ds = AttrDict()
    classes = [
        '__background__', '19696', '19695', '19697', '19700', '19694', '19698', '21253', '19699', '21254'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds  

# 多芬-posm data200323
def get_duofposm200323_data():
    ds = AttrDict()
    classes = [
        '__background__', '19761', '17353', '22781', '22783', '22782', '22780', '22784', '22787', '22788'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds  

# 达能data200325
def get_danoneposm200325_data():
    ds = AttrDict()
    classes = [
        # '__background__', '19761', '22773', '22776', '22770', '22771', '22774', '22766'
        '__background__', '0', '22773', '22776', '22770', '22771', '22774', '22766'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds  

def test():
    print(get_surveyPOSM_dataset())

# test()