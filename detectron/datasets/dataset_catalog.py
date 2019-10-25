# coding: utf-8
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

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os


# Path to data dir
_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Required dataset entry keys
_IM_DIR = 'image_directory'
_ANN_FN = 'annotation_file'

# Optional dataset entry keys
_IM_PREFIX = 'image_prefix'
_DEVKIT_DIR = 'devkit_directory'
_RAW_DIR = 'raw_dir'

# Available datasets
_DATASETS = {
    'cityscapes_fine_instanceonly_seg_train': {
        _IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        _ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_train.json',
        _RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_val': {
        _IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        # use filtered validation as there is an issue converting contours
        _ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        _RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_test': {
        _IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        _ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_test.json',
        _RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'coco_2014_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_train2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2014.json'
    },
    'coco_2014_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2014.json'
    },
    'coco_2014_minival': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_minival2014.json'
    },
    'coco_2014_valminusminival': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_valminusminival2014.json'
    },
    'coco_2015_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'coco_2015_test-dev': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'coco_2017_test': {  # 2017 test uses 2015 test images
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json',
        _IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_2017_test-dev': {  # 2017 test-dev uses 2015 test images
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
        _IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_stuff_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_train2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/coco_stuff_train.json'
    },
    'coco_stuff_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/coco_stuff_val.json'
    },
    'keypoints_coco_2014_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_train2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2014.json'
    },
    'keypoints_coco_2014_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2014.json'
    },
    'keypoints_coco_2014_minival': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_minival2014.json'
    },
    'keypoints_coco_2014_valminusminival': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_valminusminival2014.json'
    },
    'keypoints_coco_2015_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'keypoints_coco_2015_test-dev': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'voc_2007_train': {
        _IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_train.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2007_val': {
        _IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_val.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2007_test': {
        _IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_test.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2012_train': {
        _IM_DIR:
            _DATA_DIR + '/VOC2012/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/VOC2012/annotations/voc_2012_train.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/VOC2012/VOCdevkit2012'
    },
    'voc_2012_val': {
        _IM_DIR:
            _DATA_DIR + '/VOC2012/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/VOC2012/annotations/voc_2012_val.json',
        _DEVKIT_DIR:
            _DATA_DIR + '/VOC2012/VOCdevkit2012'
    },
    #  add data
    'coco_2019_HeapSeg0911Data' : {
        #  20190911_堆头分割
        _IM_DIR:
            _DATA_DIR + '/coco/coco_HeapSeg0911Data2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_HeapSeg0911Data2019.json'
    },
    'coco_2019_HeapSeg0911CheckData' : {
        #  20190911_堆头分割
        _IM_DIR:
            _DATA_DIR + '/coco/coco_HeapSeg0911Data2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_HeapSeg0911CheckData2019.json'
    },

    'coco_2019_SKU110kData' : {
        #  SKU110k trainset
        _IM_DIR:
            _DATA_DIR + '/coco/coco_SKU110kData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_SKU110kData2019.json'
    },
    'coco_2019_SKU110kCheckData' : {
        #  SKU110k testset
        _IM_DIR:
            _DATA_DIR + '/coco/coco_SKU110kData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_SKU110kCheckData2019.json'
    },

    'coco_2019_BagP1Data' : {
        #  bag data part1 trainset
        _IM_DIR:
            _DATA_DIR + '/coco/coco_BagP1Data2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_BagP1Data2019.json'
    },
    'coco_2019_BagP2Data' : {
        #  general sku test data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_BagP2Data2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_BagP2Data2019.json'
    },

    'coco_2019_SkuDetData' : {
        #  general sku test data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_SkuDet2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_SkuDetData2019.json'
    },
    'coco_2019_SkuDetTrainSetData' : {
        #  general sku train data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_SkuDetTrainSetData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_SkuDetTrainSetData2019.json'
    },

    'coco_2019_ULPOSMPart1TrainSetData' : {
        #  ul_posm part1_train data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_ULPOSMPart1TrainSetData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_ULPOSMPart1TrainSetData2019.json'
    },
    'coco_2019_ULPOSMPart1CheckSetData' : {
        #  ul_posm part1_test data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_ULPOSMPart1CheckSetData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_ULPOSMPart1CheckSetData2019.json'
    },
    'coco_2019_ULPOSMPart2TrainSetData' : {
        #  ul_posm part2_train data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_ULPOSMPart2TrainSetData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_ULPOSMPart2TrainSetData2019.json'
    },

    # AG POSM_191018
    'coco_2019_AGPOSM191018TrainSetData' : {
        #  ag_posm part12_train data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_AGPOSM191018TrainSetData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_AGPOSM191018TrainSetData2019.json'
    },
    'coco_2019_AGPOSM191018CheckData' : {
        #  ag_posm part12_test data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_AGPOSM191018CheckData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_AGPOSM191018CheckData2019.json'
    },
    'coco_2019_AGPOSM191018MergeTrainSetData' : {
        #  ag_posm part12_train Merge data 
        _IM_DIR:
            _DATA_DIR + '/coco/coco_AGPOSM191018TrainSetData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_AGPOSM191018MergeTrainSetData2019.json'
    },
    'coco_2019_AGPOSM191018MergeCheckData' : {
        #  ag_posm part12_test Merge data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_AGPOSM191018CheckData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_AGPOSM191018MergeCheckData2019.json'
    },

    # Abbott_posm
    'coco_2019_AbbottPOSMP1TrainSetData' : {
        #  Abbott_posm part1_train data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_AbbottPOSMP1TrainSetData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_AbbottPOSMP1TrainSetData2019.json'
    },
    # Abbott_posm
    'coco_2019_AbbottPOSMP2TrainSetData' : {
        #  Abbott_posm part2_train data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_AbbottPOSMP2TrainSetData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_AbbottPOSMP2TrainSetData2019.json'
    },

    # liby_posm 191024
    'coco_2019_LibyPOSMP1TrainSetData' : {
        #  liby_posm part1_train data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_LibyPOSMP1TrainSetData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_LibyPOSMP1TrainSetData2019.json'
    },
    'coco_2019_LibyPOSMP2TrainSetData' : {
        #  liby_posm part1_train data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_LibyPOSMP2TrainSetData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_LibyPOSMP2TrainSetData2019.json'
    },
    'coco_2019_LibyPOSMCheckData' : {
        #  liby_posm part1_train data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_LibyPOSMCheckData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_LibyPOSMCheckData2019.json'
    },

    
}


def datasets():
    """Retrieve the list of available dataset names."""
    return _DATASETS.keys()


def contains(name):
    """Determine if the dataset is in the catalog."""
    return name in _DATASETS.keys()


def get_im_dir(name):
    """Retrieve the image directory for the dataset."""
    return _DATASETS[name][_IM_DIR]


def get_ann_fn(name):
    """Retrieve the annotation file for the dataset."""
    return _DATASETS[name][_ANN_FN]


def get_im_prefix(name):
    """Retrieve the image prefix for the dataset."""
    return _DATASETS[name][_IM_PREFIX] if _IM_PREFIX in _DATASETS[name] else ''


def get_devkit_dir(name):
    """Retrieve the devkit dir for the dataset."""
    return _DATASETS[name][_DEVKIT_DIR]


def get_raw_dir(name):
    """Retrieve the raw dir for the dataset."""
    return _DATASETS[name][_RAW_DIR]
