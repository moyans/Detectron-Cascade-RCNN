# coding: utf-8

import os
import yaml
import numpy as np
from ast import literal_eval
import cv2
import imagedt
import imagedt.tensorflow.tools as tftools
from copy import  deepcopy


# RECOG_SKUS = [6529,6666,6539,6541,6528,6530,6527,6665,6538,6536,10270,10269,6540,6532,6919,6918,6920]

# mango : 6532
RECOG_SKUS = [6529,6541,6528,6527,10270,10269,6540,6919,6918,6920]


class TF_Classifier(object):
    def __init__(self, model_name=None, output_node=None, graph_name=''):
      super(TF_Classifier, self).__init__()
      self._config_path = self._absolute_path('model_config/mod_config.yaml')
      self.model_name = model_name or 'mobilenetv1_808'
      self.output_node = output_node
      self.graph_name = graph_name
      self._load()
      self.get_model()
      self._load_labels()
      self.Distance_threshold = 0.7

    def _absolute_path(self, path):
      return os.path.join(os.path.dirname(__file__), path)

    def _load(self):
      with open(self._config_path, 'r') as f:
        self.model_conf = yaml.load(f)[self.model_name]

    def _join_path(self, root_dir, add_path):
      return os.path.join(root_dir, add_path)

    def _load_labels(self):
      classesPath = self._join_path(self.model_root, self.model_conf['classes'])
      self.LABELS = {}
      with open(classesPath, 'r') as f:
        lines = f.readlines()
      for line in lines:
        self.LABELS[line.strip().split("\t")[1].strip()] = line.strip()

    def get_model(self):
        self.model_root = self.model_conf['root_dir']
        gpu_id = self.model_conf['GPUID']
        tfmodel = self._join_path(self.model_root, self.model_conf['tfmodel'])
        input_node = self.model_conf['input_node_name']
        output_node = self.model_conf['output_node_name'] if self.output_node == None else self.output_node

        self.predictor = tftools.TFmodel_Wrapper(tfmodel, 
                    input_nodename=input_node, 
                    output_nodename=output_node,
                    gpu_id=gpu_id,
                    graph_name=self.graph_name)

    # @imagedt.decorator.time_cost
    def classify(self, images):
      # surpport multi batchsize
      pre_infos = self.predictor.predict(images)
      infos = []
      for item in pre_infos:
          pre_cls = item['class']
          pre_conf = item['confidence']
          infos.append([pre_conf] + self.LABELS[str(pre_cls)].split('\t'))
      return infos

    def classify_multi_box(self, frame, dict_det, batchsize=24):
      # swap BGR to RGB
      frame = imagedt.image.swap_chanel_to_RGB(frame)
      # one frame: process multi batchsize
      if dict_det:
          multi_batches = self.package_multi_batch(frame, dict_det, batchsize=batchsize)
          for batch_times, multi_batche in enumerate(multi_batches):
            cls_infos = self.classify(multi_batche)
            for ite in range(len(cls_infos)):
              for index, info_name in enumerate(['confidence', 'object_id', 'cls_id','sku_name']):
                dict_det[batch_times*batchsize+ite][info_name] = cls_infos[ite][index]
          dict_det = self.filter_target_skus(dict_det)
      return dict_det

    def package_multi_batch(self, image_mat, dict_det, batchsize):
        pack_images = []
        num_batch = len(dict_det) / batchsize 
        num_batch = num_batch if len(dict_det) % batchsize == 0 else num_batch+1
        for ite_batch in range(num_batch):
            det_infos = dict_det[ite_batch*batchsize:(ite_batch+1)*batchsize]
            sub_imgs = self.splite_sub_images(image_mat, det_infos)
            pack_images.append(sub_imgs)
        return pack_images

    def splite_sub_images(self, image_mat, det_infos):
        sub_imgs = []
        for det_info in det_infos:

            # moyan_add
            det_info['bbox'] = [int(i) for i in det_info['bbox']]

            xmin, ymin, xmax, ymax = det_info['bbox']
            sub_img = image_mat[ymin:ymax, xmin:xmax]
            noise_img = imagedt.image.noise_padd(sub_img, edge_size=224, start_pixel_value=0)
            noise_img = imagedt.image.inception_preprocesing(noise_img)
            sub_imgs.append(noise_img)
        return sub_imgs

    #@imagedt.decorator.time_cost
    def splite_sub_images_extractor(self, image_mat, det_infos):
        sub_imgs = []
        for det_info in det_infos:
            xmin, ymin, xmax, ymax = det_info['bbox']
            sub_img = image_mat[ymin:ymax, xmin:xmax]
            noise_img = cv2.resize(sub_img, (224,224))
            noise_img = imagedt.image.inception_preprocesing(noise_img)
            sub_imgs.append(noise_img)
        return sub_imgs

    def splite_sub_images_gray(self, image_mat, det_infos):
        # image_mat_gray = cv2.CvtColor(image_mat, cv2.COLOR_BGR2GRAY)
        sub_imgs = []
        for det_info in det_infos:
            xmin, ymin, xmax, ymax = det_info['bbox']
            sub_img = image_mat[ymin:ymax, xmin:xmax]
            # sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
            backtorgb = cv2.cvtColor(sub_img, cv2.COLOR_GRAY2RGB)
            # temp_img = np.expand_dims(backtorgb, axis=2)
            # temp_img = np.concatenate((temp_img, temp_img, temp_img), axis=2)
            noise_img = imagedt.image.noise_padd(backtorgb, edge_size=224, start_pixel_value=0)
            noise_img = imagedt.image.inception_preprocesing(noise_img)
            sub_imgs.append(noise_img)
        return sub_imgs

    def filter_target_skus(self, outlist):
      for index, item in enumerate(outlist):
        object_id = int(item['object_id'])
        if object_id not in RECOG_SKUS:
          outlist[index]['sku_name'] =  u'其它商品'
          # outlist[index]['sku_name'] = ''
      return outlist

    # #@imagedt.decorator.time_cost
    # def extractor(self, frame, dict_det):
    #   features = []
    #   sub_imgs = self.splite_sub_images_extractor(frame, dict_det)
    #   # print len(sub_imgs)
    #   for sub_img in sub_imgs:
    #     features.append([self.predictor.extract([sub_img])])
    #   return features

    # @imagedt.decorator.time_cost
    def extractor(self, frame, dict_det):
        sub_imgs = self.splite_sub_images_extractor(frame, dict_det)
        # print len(sub_imgs)
        # for sub_img in sub_imgs:
        return self.predictor.extract(sub_imgs)


    def extractor_gray(self, frame, dict_det):
      features = []
      sub_imgs = self.splite_sub_images_gray(frame, dict_det)
      for sub_img in sub_imgs:
        features.append(self.predictor.extract([sub_img]))
      return features

    def extractor_one(self, img):
        features = []
        noise_img = cv2.resize(img, (224,224))
        # noise_img = imagedt.image.noise_padd(img, edge_size=224, start_pixel_value=255)
        noise_img = imagedt.image.inception_preprocesing(noise_img)
        features.append(self.predictor.extract([noise_img]))
        return features

    #@imagedt.decorator.time_cost
    def cosdistance(self, vector1, vector2):
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    def cal_bg_feature(self, feature_img_path, shelf_layer):
        status = False
        msg = ''
        self.bg_feature_list = [[] for i in range(shelf_layer)]
        list_imgs = imagedt.dir.loop(feature_img_path, ['.png', '.jpg'])

        for img in list_imgs:
            img_index = os.path.basename(img).split('_')[1]
            img_mat = cv2.imread(img)
            self.bg_feature_list[int(img_index)-1].append(self.extractor_one(img_mat))

        for layer, item in enumerate(self.bg_feature_list):
            if not item:
                raise ValueError("ERROR: this layer/{0} template is None ".format(layer+1))
                # exit()

    # @imagedt.decorator.time_cost
    def extrace_all_features(self, take_outlist, frame):
      ext_boxes = []
      H, W, C = frame.shape
      for items in take_outlist:
        ind, item = items
        dis_item = deepcopy(item)
        # print dis_item['bbox']
        x1, y1, x2, y2 = dis_item['bbox']
        w = x2 - x1
        h = y2 - y1
        # expand ten percent cal distance
        x1 = int(x1 - w * 0.1) if int(x1 - w * 0.1) > 0 else 0
        y1 = int(y1 - h * 0.1) if int(y1 - h * 0.1) > 0 else 0
        x2 = int(x2 + w * 0.1) if int(x2 + w * 0.1) < W else W
        y2 = int(y2 + h * 0.1) if int(y2 + h * 0.1) < H else H
        dis_item['bbox'] = x1, y1, x2, y2
        ext_boxes.append(dis_item)
      return self.extractor(frame, ext_boxes)


    #@imagedt.decorator.time_cost
    def cal_bg_template(self, outlist, frame):
        H, W, C = frame.shape
        take_outlist = [[i, item] for i, item in enumerate(outlist) if item['take_tag'] == 1]
        # select occlusion bbox according to the distance
        # have_feature_list = (np.where(np.array(self.feature_map) != 0)[0])
        left_take_outlist = []
        if take_outlist:
            features = self.extrace_all_features(take_outlist, frame)

        for f_ind, items in enumerate(take_outlist):
            ind, item = items
            dis_item_layer = item['layer']
            temp_features = self.bg_feature_list[dis_item_layer-1][0]

            dis = self.cosdistance(temp_features[0][0], features[f_ind])
            # print dis
            # import pdb
            # pdb.set_trace()
            if dis < self.Distance_threshold:
                outlist[ind]['take_tag'] = 2
            else:
                left_take_outlist.append(items)

        for item_bbox in left_take_outlist:
            ind, item = item_bbox
            item_layer = item['layer']
            if ind != 0 and ind != len(outlist)-1 :
                left_item = outlist[ind-1]
                left_item_layer = left_item['layer']
                right_item = outlist[ind+1]
                right_item_layer = right_item['layer']
                if item_layer and left_item_layer and right_item_layer:
                    if left_item['take_tag'] == right_item['take_tag'] \
                    and right_item['take_tag'] == 2 \
                    and item['take_tag'] != left_item['take_tag']:
                        outlist[ind]['take_tag'] = 2
        return outlist