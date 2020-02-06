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
    'coco_2019_SKUTrainDetData' : {
        #  20191203 general sku train data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_SKUDetData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_SKUTrainDetData2019.json'
    },
    'coco_2019_SKUTestDetData' : {
        #  20191203 general sku train data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_SKUDetData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_SKUTestDetData2019.json'
    },
    'coco_2019_SkuDetV1TrainData' : {
        #  general sku test data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_SkuDetV1Data2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_SkuDetV1TrainData2019.json'
    },
    'coco_2019_SkuDetV1TestData' : {
        #  general sku test data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_SkuDetV1Data2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_SkuDetV1TestData2019.json'
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

    # # AG POSM_191018 del
    # 'coco_2019_AGPOSM191018TrainSetData' : {
    #     #  ag_posm part12_train data
    #     _IM_DIR:
    #         _DATA_DIR + '/coco/coco_AGPOSM191018TrainSetData2019',
    #     _ANN_FN:
    #         _DATA_DIR + '/coco/annotations/instances_AGPOSM191018TrainSetData2019.json'
    # },
    # 'coco_2019_AGPOSM191018CheckData' : {
    #     #  ag_posm part12_test data
    #     _IM_DIR:
    #         _DATA_DIR + '/coco/coco_AGPOSM191018CheckData2019',
    #     _ANN_FN:
    #         _DATA_DIR + '/coco/annotations/instances_AGPOSM191018CheckData2019.json'
    # },
    # 'coco_2019_AGPOSM191018MergeTrainSetData' : {
    #     #  ag_posm part12_train Merge data 
    #     _IM_DIR:
    #         _DATA_DIR + '/coco/coco_AGPOSM191018TrainSetData2019',
    #     _ANN_FN:
    #         _DATA_DIR + '/coco/annotations/instances_AGPOSM191018MergeTrainSetData2019.json'
    # },
    # 'coco_2019_AGPOSM191018MergeCheckData' : {
    #     #  ag_posm part12_test Merge data
    #     _IM_DIR:
    #         _DATA_DIR + '/coco/coco_AGPOSM191018CheckData2019',
    #     _ANN_FN:
    #         _DATA_DIR + '/coco/annotations/instances_AGPOSM191018MergeCheckData2019.json'
    # },
    # agposm 191029
    'coco_2019_UnilevelAgPOSM191029TrainData' : {
        #  agposm 191029 train data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_UnilevelAgPOSM191029TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_UnilevelAgPOSM191029TrainData2019.json'
    },   
    'coco_2019_UnilevelAgPOSM191029CheckData' : {
        #  agposm 191029 test data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_UnilevelAgPOSM191029CheckData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_UnilevelAgPOSM191029CheckData2019.json'
    },  


    # # Abbott_posm del
    # 'coco_2019_AbbottPOSMP1TrainSetData' : {
    #     #  Abbott_posm part1_train data
    #     _IM_DIR:
    #         _DATA_DIR + '/coco/coco_AbbottPOSMP1TrainSetData2019',
    #     _ANN_FN:
    #         _DATA_DIR + '/coco/annotations/instances_AbbottPOSMP1TrainSetData2019.json'
    # },
    # # Abbott_posm del
    # 'coco_2019_AbbottPOSMP2TrainSetData' : {
    #     #  Abbott_posm part2_train data
    #     _IM_DIR:
    #         _DATA_DIR + '/coco/coco_AbbottPOSMP2TrainSetData2019',
    #     _ANN_FN:
    #         _DATA_DIR + '/coco/annotations/instances_AbbottPOSMP2TrainSetData2019.json'
    # },
    # # Abbott_posm del
    # 'coco_2019_AbbottPOSMP3TrainSetData' : {
    #     #  Abbott_posm part3_train data
    #     _IM_DIR:
    #         _DATA_DIR + '/coco/coco_AbbottPOSMP3TrainSetData2019',
    #     _ANN_FN:
    #         _DATA_DIR + '/coco/annotations/instances_AbbottPOSMP3TrainSetData2019.json'
    # },
    # Abbott_posm
    # 'coco_2019_AbbottPOSMP1108TrainData' : {
    #     #  Abbott_posm train 1108 data
    #     _IM_DIR:
    #         _DATA_DIR + '/coco/coco_AbbottPOSMP1108TrainData2019',
    #     _ANN_FN:
    #         _DATA_DIR + '/coco/annotations/instances_AbbottPOSMP1108TrainData2019.json'
    # },
    # 'coco_2019_AbbottPOSMP1108CheckData' : {
    #     #  Abbott_posm test 1108 data
    #     _IM_DIR:
    #         _DATA_DIR + '/coco/coco_AbbottPOSMP1108CheckData2019',
    #     _ANN_FN:
    #         _DATA_DIR + '/coco/annotations/instances_AbbottPOSMP1108CheckData2019.json'
    # },
    # 'coco_2019_AbbottPOSMP1120TrainData' : {
    #     #  Abbott_posm train 1120 data
    #     _IM_DIR:
    #         _DATA_DIR + '/coco/coco_AbbottPOSMP1120TrainData2019',
    #     _ANN_FN:
    #         _DATA_DIR + '/coco/annotations/instances_AbbottPOSMP1120TrainData2019.json'
    # },
    #  abbott 191204 
    'coco_2019_AbbottPOSMP1204TrainData' : {
        #  Abbott_posm train 191204 data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_AbbottPOSMP1204Data2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_AbbottPOSMP1204TrainData2019.json'
    },
    'coco_2019_AbbottPOSMP1204TestData' : {
        #  Abbott_posm test 191204 data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_AbbottPOSMP1204Data2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_AbbottPOSMP1204TestData2019.json'
    },
    'coco_2019_AbbottPOSMP1216TrainData' : {
        #  Abbott_posm train 191216 data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_AbbottPOSMP1216TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_AbbottPOSMP1216TrainData2019.json'
    },
    'coco_2019_AbbottPOSMP1209TestData' : {
        #  Abbott_posm test 191209 data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_AbbottPOSMP1209TestData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_AbbottPOSMP1209TestData2019.json'
    },


    # liby_posm 191107
    'coco_2019_LibyPOSM191107TrainData' : {
        #  liby_posm 191107 train data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_LibyPOSM191107TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_LibyPOSM191107TrainData2019.json'
    },
    'coco_2019_LibyPOSM191107CheckData' : {
        #  liby_posm 191107 test data
        _IM_DIR:
            _DATA_DIR + '/coco/coco_LibyPOSM191107CheckData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_LibyPOSM191107CheckData2019.json'
    },


    # invoice 发票
    'coco_2019_InvoiceDet191129TrainData' : {
        #  invoice train data191129
        _IM_DIR:
            _DATA_DIR + '/coco/coco_InvoiceDet191129Data2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_InvoiceDet191129TrainData2019.json'
    },
    'coco_2019_InvoiceDet191129CheckData' : {
        #  invoice val data191129
        _IM_DIR:
            _DATA_DIR + '/coco/coco_InvoiceDet191129Data2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_InvoiceDet191129CheckData2019.json'
    },

    # 立白sku
    'coco_2019_LibyTrainP1Data' : {
        #  liby train data191026
        _IM_DIR:
            _DATA_DIR + '/coco/coco_LibyTrainP1Data2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_LibyTrainP1Data2019.json'
    },
    'coco_2019_LibyCheckData' : {
        #  liby val data191026
        _IM_DIR:
            _DATA_DIR + '/coco/coco_LibyCheckData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_LibyCheck2019.json'
    },

    # 玛氏糖果sku 180802
    'coco_2019_MarsCandies180802Data' : {
        #  Mars sku train data 180802
        _IM_DIR:
            _DATA_DIR + '/coco/coco_MarsCandies180802Data2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_MarsCandies180802Data2019.json'
    },   

    # POSMDet 191101
    'coco_2019_POSMDet191101TrainData' : {
        #  POSMDet train data 191101
        _IM_DIR:
            _DATA_DIR + '/coco/coco_POSMDet191101TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_POSMDet191101TrainData2019.json'
    },       
    'coco_2019_POSMDet191101CheckData' : {
        #  POSMDet test data 191101
        _IM_DIR:
            _DATA_DIR + '/coco/coco_POSMDet191101CheckData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_POSMDet191101CheckData2019.json'
    }, 
    'coco_2019_GeneralPOSMDet191111TrainData' : {
        #  general General_MZCVol191111 train data 191111
        _IM_DIR:
            _DATA_DIR + '/coco/coco_GeneralPOSMDet191111TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_GeneralPOSMDet191111TrainData2019.json'
    }, 
    'coco_2019_GeneralulPOSMDet191115TrainData' : {
        #  general General_Unilevelul train data 191111
        _IM_DIR:
            _DATA_DIR + '/coco/coco_GeneralulPOSMDet191115Train2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_GeneralulPOSMDet191115Train2019.json'
    }, 
    'coco_2019_GeneralMZCPOSM191216TrainData' : {
        # trainset general MZC 3Class 191216 withSKU PriceTag
        _IM_DIR:
            _DATA_DIR + '/coco/coco_GeneralMZCPOSM191216TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_GeneralMZCPOSM191216TrainData2019.json'
    }, 
    'coco_2019_GeneralMZCPOSM191216TestData' : {
        # testset general MZC 3Class 191216 withSKU PriceTag
        _IM_DIR:
            _DATA_DIR + '/coco/coco_GeneralMZCPOSM191216TestData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_GeneralMZCPOSM191216TestData2019.json'
    },



    # 亿滋sku 191028
    'coco_2019_Mondelez191028TrainData' : {
        #  Mondelez sku train data 191028
        _IM_DIR:
            _DATA_DIR + '/coco/coco_Mondelez191028TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_Mondelez191028TrainData2019.json'
    },  
    'coco_2019_Mondelez191028CheckData' : {
        #  Mondelez sku test data 191028
        _IM_DIR:
            _DATA_DIR + '/coco/coco_Mondelez191028CheckData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_Mondelez191028CheckData2019.json'
    },  

    # 公牛sku 191107
    'coco_2019_Gongniu191107TrainData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_Gongniu191107TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_Gongniu191107TrainData2019.json'
    },  
    'coco_2019_Gongniu191125TrainData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_Gongniu191125TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_Gongniu191125TrainData2019.json'
    },  
    'coco_2019_Gongniou191107CheckData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_Gongniou191107CheckData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_Gongniou191107CheckData2019.json'
    },
    # 公牛posm 目标类别POSM 10class
    'coco_2019_GongniuPOSM191203TrainData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_GongniuPOSM191203Data2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_GongniuPOSM191203TrainData2019.json'
    },
    # 公牛posm 目标类别POSM 10class
    'coco_2019_GongniuPOSM191203TestData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_GongniuPOSM191203Data2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_GongniuPOSM191203TestData2019.json'
    },
    # 公牛posm 通用POSM
    'coco_2019_GongniuGeneralPOSM191203TrainData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_GongniuPOSM191203Data2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_GongniuGeneralPOSM191203TrainData2019.json'
    },
    # 公牛posm 通用POSM
    'coco_2019_GongniuGeneralPOSM191203TestData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_GongniuPOSM191203Data2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_GongniuGeneralPOSM191203TestData2019.json'
    },


    # 百事sku 191112
    'coco_2019_Pepsi191112TrainData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_Pepsi191112TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_Pepsi191112TrainData2019.json'
    },  
    'coco_2019_Pepsi191112CheckData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_Pepsi191112CheckData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_Pepsi191112CheckData2019.json'
    },  

    # 万宝路
    'coco_2014_Marlboro190627Data' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_Marlboro190627Data2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_Marlboro190627Data2014.json'
    },  
    'coco_2014_Marlboro190627CheckData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_Marlboro190627CheckData2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_Marlboro190627CheckData2014.json'
    }, 

    # 广东中烟
    'coco_2019_Tobacco191112TrainData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_Tobacco191112TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_Tobacco191112TrainData2019.json'
    },  
    'coco_2019_Tobacco191112CheckData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_Tobacco191112CheckData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_Tobacco191112CheckData2019.json'
    }, 

    # 斑布纸
    'coco_2019_Babo191124TrainData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_Babo191124TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_Babo191124TrainData2019.json'
    },  
    'coco_2019_Babo191124CheckData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_Babo191124CheckData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_Babo191124CheckData2019.json'
    }, 

    # 价格牌和价格区域
    'coco_2019_PriceTag2RegionTrainData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_PriceTag2RegionTrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_PriceTag2RegionTrainData2019.json'
    },  
    'coco_2019_PriceTag2RegionCheckData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_PriceTag2RegionCheckData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_PriceTag2RegionCheckData2019.json'
    },  

    # Art
    'coco_2019_TextArtTrainData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_TextArtTrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_TextArtTrainData2019.json'
    }, 
    # # DAS
    # 'coco_2019_TextDASTrainData' : {
    #     _IM_DIR:
    #         _DATA_DIR + '/coco/coco_TextDASTrainData2019',
    #     _ANN_FN:
    #         _DATA_DIR + '/coco/annotations/instances_TextDASTrainData2019.json'
    # }, 
    # LSVT
    'coco_2019_TextLSVTTrainData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_TextLSVTTrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_TextLSVTTrainData2019.json'
    }, 
    # MTWI
    'coco_2019_TextMTWITrainData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_TextMTWITrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_TextMTWITrainData2019.json'
    }, 
    # RCTW
    'coco_2019_TextRCTWTrainData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_TextRCTWTrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_TextRCTWTrainData2019.json'
    }, 
    # ReCTS
    'coco_2019_TextReCTSTrainData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_TextReCTSTrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_TextReCTSTrainData2019.json'
    }, 
    'coco_2019_TextReCTSCheckData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_TextReCTSTrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_TextReCTSCheckData2019.json'
    }, 

    'coco_2019_ByHealthPOS191209TrainData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_ByHealthPOS191209TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_ByHealthPOS191209TrainData2019.json'
    }, 
    'coco_2019_ByHealthPOS191209TestData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_ByHealthPOS191209TestData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_ByHealthPOS191209TestData2019.json'
    }, 
    'coco_2019_ByHealthPOS191230TrainData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_ByHealthPOS191230TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_ByHealthPOS191230TrainData2019.json'
    },

    # 格力高posm glico
    'coco_2019_GlicoPOS191212TrainData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_GlicoPOS191212TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_GlicoPOS191212TrainData2019.json'
    }, 
    'coco_2019_GlicoPOS191212TestData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_GlicoPOS191212TestData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_GlicoPOS191212TestData2019.json'
    }, 

    # 美赞臣posm data191213
    'coco_2019_MZCPOS191213TrainData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_MZCPOS191213TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_MZCPOS191213TrainData2019.json'
    }, 
    'coco_2019_MZCPOS191213TestData' : {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_MZCPOS191213TestData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_MZCPOS191213TestData2019.json'
    }, 



    # 黑人sku检测 dataset
    'coco_2019_HeirRen190919TrainData': {
        # HeirRen 190919 Train
        _IM_DIR:
            _DATA_DIR + '/coco/coco_HeirRen190919TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_HeirRen190919TrainData2019.json'
    },
    'coco_2019_HeirRen190919TestData': {
        # HeirRen 190919 test
        _IM_DIR:
            _DATA_DIR + '/coco/coco_HeirRen190919TestData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_HeirRen190919TestData2019.json'
    },
    'coco_2019_HeirRen191225TrainData': {
        # HeirRen 191225 Train
        _IM_DIR:
            _DATA_DIR + '/coco/coco_HeirRen191225TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_HeirRen191225TrainData2019.json'
    },
    'coco_2019_HeirRen191225TestData': {
        # HeirRen 191225 test
        _IM_DIR:
            _DATA_DIR + '/coco/coco_HeirRen191225TestData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_HeirRen191225TestData2019.json'
    },

    # 汤臣倍健 190312 sku dataset
    'coco_2019_TCBJ190312TrainData': {
        # Tcbj 190312 train
        _IM_DIR:
            _DATA_DIR + '/coco/coco_TCBJ190312TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_TCBJ190312TrainData2019.json'
    },
    # 汤臣倍健 190312 sku dataset
    'coco_2019_TCBJ190312TestData': {
        # Tcbj 190312 test
        _IM_DIR:
            _DATA_DIR + '/coco/coco_TCBJ190312TestData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_TCBJ190312TestData2019.json'
    },
    # 汤臣倍健 skudata 200105 
    'coco_2019_TCBJ200105TrainData': {
        # HeirRen 191225 test
        _IM_DIR:
            _DATA_DIR + '/coco/coco_TCBJ200105TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_TCBJ200105TrainData2019.json'
    },

    # 格力高 skudata 200103 
    'coco_2019_GlicoSku200103TrainData': {
        # Glico 200103 trainset
        _IM_DIR:
            _DATA_DIR + '/coco/coco_GlicoSku200103TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_GlicoSku200103TrainData2019.json'
    },
    'coco_2019_GlicoSku200103TestData': {
        # Glico 200103 testset
        _IM_DIR:
            _DATA_DIR + '/coco/coco_GlicoSku200103TestData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_GlicoSku200103TestData2019.json'
    },

    # 玛氏年会sku检测
    'coco_2019_MarsSku200119TrainData': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_MarsSku200119TrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_MarsSku200119TrainData2019.json'
    },
    'coco_2019_MarsSku200119TestData': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_MarsSku200119TestData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_MarsSku200119TestData2019.json'
    },

    # 2019之前价格牌检测数据整理
    'coco_2019_PricetagDetTrainData': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_PricetagDetTrainData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_PricetagDetTrainData2019.json'
    },
    # ------------------------------------------------------------------ #
    'coco_2014_PriceDet10wmzc0402Data': {
        # PriceDet 10w价格牌mzc_price0402(部分),大图5383
        _IM_DIR:
            _DATA_DIR + '/coco/coco_PriceDet10wmzc0402Data2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_PriceDet10wmzc0402Data2014.json'
    },
    # ------------------------------------------------------------------ #
    'coco_2019_PricetagDetTestData': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_PricetagDetTestData2019',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_PricetagDetTestData2019.json'
    },

    # 拜耳价格牌200101
    'coco_2020_BayerPricetagDet200101TrainData': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_BayerPricetagDet200101TrainData2020',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_BayerPricetagDet200101TrainData2020.json'
    },
    'coco_2020_BayerPricetagDet200101TestData': {
        _IM_DIR:
            _DATA_DIR + '/coco/coco_BayerPricetagDet200101TestData2020',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_BayerPricetagDet200101TestData2020.json'
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
