# coding: utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import cv2
import numpy as np
from PIL import ImageDraw, ImageFont, Image

import time
from copy import deepcopy
import imagedt
from imagedt.decorator import time_cost
from imagedt.image.process import resize_with_scale


# cv font
"""
font types enum
  {
    FONT_HERSHEY_SIMPLEX = 0,
    FONT_HERSHEY_PLAIN = 1,
    FONT_HERSHEY_DUPLEX = 2,
    FONT_HERSHEY_COMPLEX = 3,
    FONT_HERSHEY_TRIPLEX = 4,
    FONT_HERSHEY_COMPLEX_SMALL = 5,
    FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
    FONT_HERSHEY_SCRIPT_COMPLEX = 7,
    FONT_ITALIC = 16
  };
"""


# logo parms
# LOG_PATH = './log.png'
# LOG_PATH = '/workspace/sunchao/idt-candies-realtime-recognize/model_config/logo.png'
LOG_PATH = '/data/code/AI_Congress_brk/idt-candies-realtime-recognize/model_config/logo.png'
LOG_IMG = cv2.imread(LOG_PATH, cv2.IMREAD_UNCHANGED)
LOG_IMG = imagedt.image.resize_with_scale(LOG_IMG, 400)
_, _, _, LOG_ALP = cv2.split(LOG_IMG)
LOGO_H, LOGO_W, logo_chanel = LOG_IMG.shape

LOG_ALP_IND = np.where((LOG_ALP/255)==1)
LOG_IMG = imagedt.image.resize_with_scale(cv2.imread(LOG_PATH), 400)

# draw colors
color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


class Create_Show_Image(object):
  """docstring for Create_Show_Image"""
  def __init__(self, font_bg_color=(200, 200, 200)):
    super(Create_Show_Image, self).__init__()
    self.font_size = 24
    self.font_bg_color = font_bg_color
    self.color_space =[(0, 0, 0), (0, 0, 255), (255, 0, 0)]
    self.color = self.color_space[0]
    # self.font_path = '/workspace/sunchao/idt-candies-realtime-recognize/model_config/wqy-zenhei.ttc' # '/workspace/sunchao/idt-candies-realtime-recognize/model_config/Microsoft Yahei.ttf'
    self.font_path = '/data/code/AI_Congress_brk/idt-candies-realtime-recognize/model_config/Microsoft Yahei.ttf'
    self.font = ImageFont.truetype(self.font_path, self.font_size)
    self.show_font_height = 30
    self.text_frame = None
    # self.font_type = cv2.freetype.createFreeType2()
    # self.font_type.loadFontData(fontFileName='./model_config/Microsoft Yahei.ttf', id=0)

    # init show window
    # self.init_window(window_size)

  def gen_font_space(self, width, height):
    blank_image = np.zeros((height, width/3, 3), np.uint8)
    blank_image[:, :] = self.font_bg_color  # (B, G, R)
    return blank_image

  def gen_font_space_PIL(self, width, height):
    font_space = Image.new('RGB', (width/4, height), (200,200,200))
    return font_space

  def get_show_text(self, out_info, ind, show_conf=False):
    # show_str = None
    # show_str = 'Id' +': ' + str(ind+1).zfill(2) + ' '
    # show_str +=  'name' + ': ' + str(out_info['sku_name'])
    sku_name = out_info.get('sku_name', None)
    if sku_name is not None:
      if show_conf:
        sku_conf = round(float(out_info['confidence']), 2)
        show_str = str(ind+1).zfill(2) + ': ' + str(sku_name) +'_'+str(sku_conf)
      else:
        show_str = str(ind+1).zfill(2) + ': ' + str(sku_name)
    else:
      show_str = str(ind+1).zfill(2) + ': ' + u'其它商品'
    return show_str

  def rest_bg_color(self, cvmat, ind):
    cvmat.flags.writeable = True
    cvmat[30*(ind+1):30*(ind+2), 20:] = self.font_bg_color
    return cvmat

  #@time_cost
  def crete_infos_image(self, det_frame, out_infos_list):
    # import pudb; pu.db
    # det_frame = self.draw_detect_rect(det_frame, out_infos_list)
    # det_frame = det_frame[:, 300:]
    # det_frame = det_frame[self.ymin:, :]

    h, w, c = det_frame.shape
  
    self.font_space = self.gen_font_space_PIL(w, h)
    # self.font_space = self.gen_font_space(self.width, self.height)

    for ind, out_info in enumerate(out_infos_list):
      take_state = out_info['take_tag']
      if take_state == 1:
        self.color = self.color_space[1] 
      elif take_state == 2:
        self.color = (0, 255, 255)
      else:
        self.color = self.color_space[0]

      show_str = self.get_show_text(out_info, ind, show_conf=False)
      # if show_str is None:
      #   continue

      draw = ImageDraw.Draw(self.font_space)
      draw.text((20, self.show_font_height*(ind+1)), unicode(show_str), font=self.font, fill=self.color)

      # show_img = self.font_type.putText(img=self.font_space,
      #                                   text=show_str,
      #                                   org=(25, 25*(ind+1)),
      #                                   fontHeight=self.show_font_height,
      #                                   color=self.color,
      #                                   thickness=-1,
      #                                   line_type=cv2.LINE_AA,
      #                                   bottomLeftOrigin=True)
      # self.color = self.color_space[0]
    # merger det frame and show text cvmat
    # show_img = np.concatenate((det_frame, show_img), axis=1)
    return self.font_space

  def draw_detect_rect(self, frame, out_list):
    if not out_list:
       return frame
    for ind, out_infos in enumerate(out_list):
      take_state = out_infos['take_tag']
      # skiping shade sku 
      if take_state == 2:
        continue

      cc = color[2] if take_state == 1 else color[1]
      bbox = out_infos['bbox']
      [x1, y1, x2, y2] = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
      # cv2.rectangle(frame, (x1, y1), (x2, y2), cc, 2)

      box_h = (y2-y1)/10
      box_w = (x2-x1)/5
      cv2.line(frame, (x1, y1), (x1+box_w, y1), cc, thickness=2, lineType=8, shift=0)
      cv2.line(frame, (x1, y1), (x1, y1+box_h), cc, thickness=2, lineType=8, shift=0)
      cv2.line(frame, (x2, y1), (x2-box_w, y1), cc, thickness=2, lineType=8, shift=0)
      cv2.line(frame, (x2, y1), (x2, y1+box_h), cc, thickness=2, lineType=8, shift=0)
      cv2.line(frame, (x1, y2), (x1+box_w, y2), cc, thickness=2, lineType=8, shift=0)
      cv2.line(frame, (x1, y2), (x1, y2-box_h), cc, thickness=2, lineType=8, shift=0)
      cv2.line(frame, (x2, y2), (x2-box_w, y2), cc, thickness=2, lineType=8, shift=0)
      cv2.line(frame, (x2, y2), (x2, y2-box_h), cc, thickness=2, lineType=8, shift=0)

      cv2.putText(frame, str(ind+1).zfill(2), (x2-20, y2-self.show_font_height/4), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                  (9,247,247), 2)
    return frame

  def debug_detect_window(self, frame, xmin, ymin, xmax, ymax):
    frame = self.add_logo(frame)
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
    cv2.imshow("debug_windows", frame)
    print("xmin: {0}, ymin: {1}, xmax: {2}, ymax: {3}".format(xmin, ymin, xmax, ymax))

  def merge_show_image(self, box_image, info_image):
    box_image = self.add_logo(box_image)
    show_img = np.concatenate((box_image, np.asarray(info_image)), axis=1)
    return show_img

  def add_logo(self, frame):
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    frame[-LOGO_H:, -LOGO_W:][LOG_ALP_IND] = LOG_IMG[LOG_ALP_IND]
    return frame

