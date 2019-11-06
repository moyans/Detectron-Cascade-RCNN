# coding: utf-8
import os
import cv2
import time
import socket
import requests
import base64
import __init__
from BASE_TOOLS import walkDir2RealPathList

import sys
reload(sys)
sys.setdefaultencoding('utf8')

global_port = '7000'
hostname = socket.gethostname()
# hostname = 'm'
if hostname.startswith('m'):
    # 线上环境
    # global_host = '121.201.72.140:'
    global_host = '121.201.72.137:'
else:
    # 本地环境s
    global_port = '5436'
    global_host = '10.196.51.24:'    # Moyan

def api_get(path, data):
    url = ('%s%s%s%s') % ('http://', global_host, global_port, path)
    res = requests.post(url, data=data)
    return res.json()

def api_price_tag_recognize(image_str):
    """
    :param imgs:  list[image_str1, image_str2, image_str3, ...]
    :return: labels: list[label1, label2, ...]
    """
    data_dict = {
        'image': image_str,
    }
    return api_get('/price_tag_recognize/', data_dict)


def getPriceApi(im):
    image_str = cv2.imencode('.png', im)[1]
    image_str = base64.b64encode(image_str)
    st_time = time.time()
    result = api_price_tag_recognize([image_str])
    print("API price tag recognize image time {0}".format(time.time() - st_time), result['prices'][0])
    if result['message'] == '图象识别成功':
        return result['prices'][0]
    else:
        print(result['message'])
        raise Exception('图象识别失败')


if __name__ == "__main__":

    pass

