# coding: utf-8
import os
import cv2
import time
import socket
import requests
import base64

# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

global_port = '7000'
hostname = socket.gethostname()
# hostname = 'm'
if hostname.startswith('m'):
    # 线上环境
    # global_host = '121.201.72.140:'
    global_host = '121.201.72.137:'
else:
    # 本地环境s
    global_port = '5437'
    global_host = '192.168.72.39:'    # Moyan
    # global_host = '192.168.72.39:'

def api_get(path, data):
    url = ('%s%s%s%s') % ('http://', global_host, global_port, path)
    return requests.post(url, data=data).json()

def api_ocr_recognize(image_str):
    """
    :param imgs:  list[image_str1, image_str2, image_str3, ...]
    :return: labels: list[label1, label2, ...]
    """
    data_dict = {
        'image': image_str,
    }
    return api_get('/ocr_recognize/', data_dict)


def getRecApi(im):
    image_str = cv2.imencode('.png', im)[1]
    image_str = base64.b64encode(image_str)
    st_time = time.time()
    result = api_ocr_recognize([image_str])
    print("result: ", result)
    print("API ocr recognize image time {0}".format(time.time() - st_time), result['predict'][0])
    if result['message'] == '图象识别成功':
        return result['predict'][0]
    else:
        print(result['message'])
        raise Exception('图象识别失败')

def main():
    testDir = '/data/code/zhangshu/OCR_API/text_recognition/demo_image'
    testLists = os.listdir(testDir)
    for nname in testLists:
        print(nname)
        imgPath = os.path.join(testDir,nname)
        assert os.path.exists(imgPath)
        getRecApi(cv2.imread(imgPath))


if __name__ == "__main__":
    
    print("hello world")
    # main()
