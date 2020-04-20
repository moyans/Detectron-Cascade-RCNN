# coding=utf-8
import os
import sys
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import argparse
import time
import traceback

def imresize(src, height):
    ratio = src.shape[0] * 1.0 / height
    width = int(src.shape[1] * 1.0 / ratio)
    return cv.resize(src, (width, height))

def upsample_pnts_height(resize_h, ori_h, pnts):
    ratio_h = ori_h*1./resize_h
    pnts[0] = int(pnts[0] * ratio_h)
    pnts[1] = int(pnts[1] * ratio_h)
    return pnts

class FMFO:

    #sift:
    nfeatures = 500
    nOctaveLayers = 3
    contrastThreshold = 0.04
    edgeThreshold = 10
    sigma = 1.6

    #surf
    hessianThreshold = 100
    nOctaves = 4
    nOctaveLayers = 3
    extended = False
    upright = False

    #orb
    scaleFactor = 1.2
    nlevels = 8
    edgeThreshold = 31
    fastThreshold = 20

    def __init__(self, height, min_feature_num, min_match_num, method):
        self.height = height
        self.min_feature_num = min_feature_num
        self.min_match_num = min_match_num
        self.feature_method = method

    # 特征通用参数
    def set_general_parameters(self, height, min_feature_num, min_match_num, method):
        self.height = height
        self.min_feature_num = min_feature_num
        self.min_match_num = min_match_num
        self.feature_method = method

    # 特征计算参数
    def set_sift_parameters(self, nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6):
        self.nfeatures = nfeatures
        self.nOctaveLayers = nOctaveLayers
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold = edgeThreshold
        self.sigma = sigma

    def set_surf_parameters(self, hessianThreshold = 100, nOctaves = 4, nOctaveLayers = 3, extended = False, upright = False):
        self.hessianThreshold = hessianThreshold
        self.nOctaves = nOctaves
        self.nOctaveLayers = nOctaveLayers
        self.extended = extended
        self.upright = upright

    def set_orb_parameters(self, nfeatures=500, scaleFactor = 1.2, nlevels = 8, edgeThreshold = 31, fastThreshold=20):
        self.nfeatures = nfeatures
        self.scaleFactor = scaleFactor
        self.nlevels = nlevels
        self.edgeThreshold = edgeThreshold
        self.fastThreshold = fastThreshold

    # 设置模板图片
    def set_template_pic(self, img):
        self.img_object_color = img
        if img.shape[0] > self.height:
            self.img_object_resize = imresize(img, self.height)
            self.img_object = cv.cvtColor(self.img_object_resize, cv.COLOR_BGR2GRAY)
        else:
            self.img_object_resize = img
            self.img_object = cv.cvtColor(self.img_object_resize, cv.COLOR_BGR2GRAY)

        self.keypoints_obj, self.descriptors_obj = \
            self.feature_detect_compute(self.img_object, self.feature_method)

    def feature_detect_compute(self, img, method):
        if method == "sift":
            detector = cv.xfeatures2d.SIFT_create(self.nfeatures, self.nOctaveLayers, self.contrastThreshold,
                                                  self.edgeThreshold, self.sigma)
            keypoints, descriptors = detector.detectAndCompute(img, None)
        elif method == "surf":
            detector = cv.xfeatures2d.SURF_create(self.hessianThreshold, self.nOctaves, self.nOctaveLayers,
                                                  self.extended, self.upright)
            keypoints, descriptors = detector.detectAndCompute(img, None)
        elif method == "orb":
            detector = cv.ORB_create(self.nfeatures, self.scaleFactor, self.nlevels, self.edgeThreshold)
            detector.setFastThreshold(self.fastThreshold)
            keypoints, descriptors = detector.detectAndCompute(img, None)
            descriptors = descriptors.astype(np.float32)
        else:
            print "not ", method, "implemented."

        return keypoints, descriptors

    # 寻找目标，若失败返回Nono，成功返回4个点
    def find_object(self, frame, display=False):
        t = time.time()
        if frame is None:
            return None
        frame_mean = np.mean(frame)
        if frame_mean < 30 or frame_mean > 225:
            return None
        img_scene = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        img_scene_resize = imresize(img_scene, self.height)
        keypoints_scene, descriptors_scene = self.feature_detect_compute(img_scene_resize, self.feature_method)

        if len(keypoints_scene) < self.min_feature_num:
            if display is True:
                img_matches = np.zeros(
                    (max(self.img_object_resize.shape[0], img_scene.shape[0]),
                     self.img_object_resize.shape[1] + img_scene.shape[1], 3),
                    dtype=np.uint8)
                img_matches[0:self.img_object_resize.shape[0], 0:self.img_object_resize.shape[1], :] = self.img_object_resize
                img_matches[0:frame.shape[0], self.img_object_resize.shape[1]:img_matches.shape[1], :] = frame
                cv.namedWindow('Good Matches & Object detection')
                cv.imshow('Good Matches & Object detection', img_matches)
                cv.waitKey(33)
            return None

        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(self.descriptors_obj, descriptors_scene, 2)

        # -- Filter matches using the Lowe's ratio test
        ratio_thresh = 0.75
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        if len(good_matches) < self.min_match_num:
            if display is True:
                img_matches = np.zeros(
                    (max(self.img_object_resize.shape[0], img_scene.shape[0]),
                     self.img_object_resize.shape[1] + img_scene.shape[1], 3),
                    dtype=np.uint8)
                img_matches[0:self.img_object_resize.shape[0], 0:self.img_object_resize.shape[1], :] = self.img_object_resize
                img_matches[0:frame.shape[0], self.img_object_resize.shape[1]:img_matches.shape[1], :] = frame
                cv.namedWindow('Good Matches & Object detection')
                cv.imshow('Good Matches & Object detection', img_matches)
                cv.waitKey(33)
            return None

        # -- Localize the object
        obj = np.empty((len(good_matches), 2), dtype=np.float32)
        scene = np.empty((len(good_matches), 2), dtype=np.float32)
        for i in range(len(good_matches)):
            # -- Get the keypoints from the good matches
            obj[i, 0] = self.keypoints_obj[good_matches[i].queryIdx].pt[0]
            obj[i, 1] = self.keypoints_obj[good_matches[i].queryIdx].pt[1]
            scene[i, 0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
            scene[i, 1] = keypoints_scene[good_matches[i].trainIdx].pt[1]

        H, mask = cv.findHomography(obj, scene, cv.RANSAC)
        matchesMask = mask.ravel().tolist()

        if matchesMask.count(1) < self.min_match_num:
            if display is True:
                img_matches = np.zeros(
                    (max(self.img_object_resize.shape[0], img_scene.shape[0]),
                     self.img_object_resize.shape[1] + img_scene.shape[1], 3),
                    dtype=np.uint8)
                img_matches[0:self.img_object_resize.shape[0], 0:self.img_object_resize.shape[1], :] = self.img_object_resize
                img_matches[0:frame.shape[0], self.img_object_resize.shape[1]:img_matches.shape[1], :] = frame
                cv.namedWindow('Good Matches & Object detection')
                cv.imshow('Good Matches & Object detection', img_matches)
                cv.waitKey(33)
            return None

        # -- Get the corners from the image_1 ( the object to be "detected" )
        obj_corners = np.empty((4, 1, 2), dtype=np.float32)
        obj_corners[0, 0, 0] = 0
        obj_corners[0, 0, 1] = 0
        obj_corners[1, 0, 0] = self.img_object_resize.shape[1]
        obj_corners[1, 0, 1] = 0
        obj_corners[2, 0, 0] = self.img_object_resize.shape[1]
        obj_corners[2, 0, 1] = self.img_object_resize.shape[0]
        obj_corners[3, 0, 0] = 0
        obj_corners[3, 0, 1] = self.img_object_resize.shape[0]

        scene_corners = cv.perspectiveTransform(obj_corners, H)
        for sc_corner in scene_corners:
            scene_pnt = upsample_pnts_height(self.height, img_scene.shape[0], sc_corner[0])
            sc_corner[0] = scene_pnt

        if display is True:
            # -- Draw matches
            img_matches = np.zeros(
                (max(self.img_object_resize.shape[0], img_scene.shape[0]), self.img_object_resize.shape[1] + img_scene.shape[1], 3),
                dtype=np.uint8)
            for kp in keypoints_scene:
                kp_scene = np.array([kp.pt[0], kp.pt[1]])
                scene_pnt = upsample_pnts_height(self.height, img_scene.shape[0], kp_scene)
                kp.pt = (scene_pnt[0], scene_pnt[1])
            cv.drawMatches(self.img_object_resize, self.keypoints_obj, frame, keypoints_scene, good_matches, img_matches,
                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # -- Draw lines between the corners (the mapped object in the scene - image_2 )
            cv.line(img_matches, (int(scene_corners[0, 0, 0] + self.img_object_resize.shape[1]), int(scene_corners[0, 0, 1])), \
                    (int(scene_corners[1, 0, 0] + self.img_object_resize.shape[1]), int(scene_corners[1, 0, 1])), (0, 255, 0), 4)
            cv.line(img_matches, (int(scene_corners[1, 0, 0] + self.img_object_resize.shape[1]), int(scene_corners[1, 0, 1])), \
                    (int(scene_corners[2, 0, 0] + self.img_object_resize.shape[1]), int(scene_corners[2, 0, 1])), (0, 255, 0), 4)
            cv.line(img_matches, (int(scene_corners[2, 0, 0] + self.img_object_resize.shape[1]), int(scene_corners[2, 0, 1])), \
                    (int(scene_corners[3, 0, 0] + self.img_object_resize.shape[1]), int(scene_corners[3, 0, 1])), (0, 255, 0), 4)
            cv.line(img_matches, (int(scene_corners[3, 0, 0] + self.img_object_resize.shape[1]), int(scene_corners[3, 0, 1])), \
                    (int(scene_corners[0, 0, 0] + self.img_object_resize.shape[1]), int(scene_corners[0, 0, 1])), (0, 255, 0), 4)

            # -- Show detected matches
            cv.namedWindow('Good Matches & Object detection')
            cv.imshow('Good Matches & Object detection', img_matches)
            t = time.time() - t
            #print "run time: ", t
            cv.waitKey(33)

        if len(scene_corners) < 4:
            return None

        #print scene_corners

        return scene_corners


if __name__ == '__main__':

    template_img = cv.imread('./shelf_template.png')
    #初始化
    fmfo = FMFO(height=320, min_feature_num=20, min_match_num=10, method='orb')
    # #通用参数
    # fmfo.set_general_parameters(height=480, min_match_num=10, method='orb')
    # #特征计算参数
    # fmfo.set_orb_parameters(nfeatures=2000, scaleFactor = 1.2, nlevels = 8, edgeThreshold = 31, fastThreshold=20)

    fmfo.set_general_parameters(height=320, min_feature_num=20, min_match_num=20, method='sift')
    fmfo.set_sift_parameters(nfeatures=300, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

    # fmfo.set_general_parameters(height=320, min_match_num=10, method='surf')
    # fmfo.set_surf_parameters(hessianThreshold = 200, nOctaves = 4, nOctaveLayers = 3, extended = False, upright = False)

    #设置模板图片
    fmfo.set_template_pic(template_img)

    cap = cv.VideoCapture('/data/code/AI_Congress_brk/idt-candies-realtime-recognize/video/2018-09-10-183454.webm')
    # capture = cv2.VideoCapture('/workspace/sunchao/test.mp4')
    # capture = cv2.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    MJPG = 1196444237.0
    cap.set(cv.CAP_PROP_FOURCC, MJPG)


    # cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        try:
            #寻找目标，若失败返回Nono，成功返回4个点，测试时display为True
            corners = fmfo.find_object(frame=frame, display=True)
        except:
            print traceback.print_exc()
            continue

