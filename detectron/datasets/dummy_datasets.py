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

def get_agposm191018_merge_dataset():
    ds = AttrDict()
    classes = [
        '__background__', '19259', '19263', '19262', '19264', '19256', '19258', '19266', '19267', '19260'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_abbottposm191018_d1_dataset():
    ds = AttrDict()
    classes = [
        # '__background__', '19294', '19279', '19286', '19285', '19272', '19273', '19274', '19275', '19277', '19297', '19276', '19298', '19293', '19296'
        '__background__', '18669', '17913', '18666', '18666', '17903', '17904', '17904', '17904', '17907', '19268', '17907', '19270', '18669', '19268'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_agposm191018_merge_dataset():
    ds = AttrDict()
    classes = [
        '__background__', '19259', '19263', '19262', '19264', '19256', '19258', '19266', '19267', '19260'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds

def get_libyposm191024_dataset():
    ds = AttrDict()
    classes = [
        '__background__', '18918', '19302', '18917', '19301', '19300'
        #  '__background__', '18918', '18918', '18917', '18919', '18919'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds



def test():
    print(get_surveyPOSM_dataset())

# test()