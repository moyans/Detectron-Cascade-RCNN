#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 18-8-29 下午4:48
import cv2, os
import time
import numpy as np
from init_detect import Detecter
# from init_detect_retina import Detecter
from init_tf_classifier import TF_Classifier
from init_window import Create_Show_Image
from utils.timer import Timer
from featurematchingfindobjects import FMFO

import sys
sys.path.append('./')

MJPG = 1196444237.0

def main():
    DEBUG = False
    shelf_layer = 4
    SAVE_TEMP_IMAGE = False

    classifier = TF_Classifier(model_name='mobilenetv1_808', graph_name='classify')

    # detect
    # xmin: 300, ymin: 0, xmax: 1380, ymax: 1080
    rset_xmin, rset_ymin, rset_xmax, rset_ymax = 400, 0, 1250, 1080  # 400, 0, 1300, 1080  #
    # shelf_area = [250, 0, 1154, 1080]
    shelf_area = [rset_xmin, rset_ymin, rset_xmax, rset_ymax]

    detecter = Detecter('detsku_rpn50', shelf_area, sku_num=32, shelf_layer=shelf_layer, filter_threshold=0.5)# detsku_vgg16 # detsku_retina  # detsku_rpn101

    capture = cv2.VideoCapture('/data/code/AI_Congress_brk/idt-candies-realtime-recognize/video/2018-09-10-183454.webm')
    # capture = cv2.VideoCapture('/data/git/video/02.webm')
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    capture.set(cv2.CAP_PROP_FOURCC, MJPG)
    # moyan debug
    conf_list = []
    while not capture.isOpened():
        print('Waiting for camera to exist...')
        time.sleep(10)
    print capture.isOpened()
    clock = 0
    success, frame = capture.read()
    create_showImage = Create_Show_Image(font_bg_color=(200, 200, 200))

    # cal bg_tem_feature
    tem_bg_img = '/data/code/AI_Congress_brk/idt-candies-realtime-recognize/tem_bg_img'
    # tem_bg_img = '/workspace/sunchao/idt-candies-realtime-recognize/tem_bg_img'

    classifier.cal_bg_feature(tem_bg_img, shelf_layer)

    # if not SAVE_TEMP_IMAGE:
    #     temp_mid_img = cv2.imread('/data/git/idt-candies-realtime-recognize/bg_temp.jpg') # 0
    #     temp_0_img = cv2.imread('/data/git/idt-candies-realtime-recognize/bg_temp.jpg')  # 0
    #     temp_7_img = cv2.imread('/data/git/idt-candies-realtime-recognize/bg_temp.jpg')  # 0
    #     temp_mid_img_features = classifier.extractor_one(temp_mid_img)
    #     temp_0_img_features = classifier.extractor_one(temp_0_img)
    #     temp_7_img_features = classifier.extractor_one(temp_7_img)

    template_img = cv2.imread('./shelf_template.png')
    # 初始化
    fmfo = FMFO(height=320, min_feature_num=20, min_match_num=10, method='orb')
    fmfo.set_general_parameters(height=320, min_feature_num=20, min_match_num=10, method='sift')
    fmfo.set_sift_parameters(nfeatures=300, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    # 设置模板图片
    fmfo.set_template_pic(template_img)

    template_list = []
    while success:
        success, frame = capture.read()
        print frame.shape
        clock += 1
        # 缓冲10帧q
        if clock < 10:
            continue

        # debug detect image size
        if DEBUG:
            out_list = detecter.detect(frame)
            frame = create_showImage.draw_detect_rect(frame, out_list)
            create_showImage.debug_detect_window(frame, rset_xmin, rset_ymin, rset_xmax, rset_ymax)
        else:
            timer = Timer()
            timer.tic()

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
                    detecter.template = []
                    detecter.template_list = []

            import time
            sart_ = time.time()
            out_list = detecter.detect(frame, crop_area=shelf_area)
            print('inference time: {}'.format(time.time() - sart_))

            # filter bbox according to iou
            out_list = detecter.filter_outlist(out_list)

            # get classify infos
            out_list = classifier.classify_multi_box(frame, out_list)

            # print out_list
            # 检测输出为空，过滤空检测
            if not detecter.template:
                detecter.calcu_template_frame(out_list)
                print 'detecter.template_list: {}'.format(len(detecter.template_list))
            else:
                out_list = detecter.calcu_take_tag(out_list)
                out_list = classifier.cal_bg_template(out_list, frame)

            rect_image = create_showImage.draw_detect_rect(frame, out_list)
            if detecter.template:
                infos_image = create_showImage.crete_infos_image(frame, out_list)
            else:
                infos_image = create_showImage.crete_infos_image(frame, [])

            frame = create_showImage.merge_show_image(rect_image, infos_image)
            # print ('Detection took {:.5f}s').format(timer.total_time)

        cv2.rectangle(frame, (shelf_area[0], shelf_area[1]), (shelf_area[2], shelf_area[3]), color=(255, 0, 0),
                      thickness=3)

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow('image', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()

        success, frame = capture.read()
    capture.release()


if __name__ == "__main__":
    main()
