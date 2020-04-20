#!/usr/bin/env python
# -*- coding: utf-8 -*-

from RPN_inference import RPN
from TWO_Stage_inference import model





class shelf_keeper():

    def __init__(self, model_name=None, crop_area=None, sku_num =24, shelf_layer=3, filter_threshold=0.5):
        super(Detecter, self).__init__()
        self._config_path = self._absolute_path('../model_config/mod_config.yaml')
        self.model_name = model_name or 'detsku_vgg16'
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
        labelmap = os.path.join(model_root, self.model_conf['labelmap'])
        GPUID = self.model_conf['GPUID']
        self.detect_net_runner = Detection_net(prototxt, caffemodel, labelmap, GPUID, self.crop_area)

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
                    for ind,ite in enumerate(template):
                        ite['bbox'] /= int(tag_ind[ind])
                        ite['take_tag'] = 0  # [0:static , 1:take, 3:put back]
                    # template.append(template)
                    self.template = template
                    self.calcu_area(template)
                    self.template_list = []

    # def calcu_take_tag
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










