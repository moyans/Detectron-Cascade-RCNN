# coding=utf-8
import logging
import os
import cv2
from gevent.pool import Pool

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s %(levelname)s] %(message)s')


class LabelGenerator(object):
    """
    通过获取大图矩阵以及bndbox列表
    将小图输出到指定目录(`output_path`)
    """
    def __init__(self, output_path):
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self._pool = Pool(size=5)

    def splitImg(self, img, xmin, xmax, ymin, ymax):
        #   img = cv2.imread(file)
        #   if img is None:
        #       return None

        xmin, ymin, xmax, ymax = self.expan_sku_area(img,
                                                     [xmin, ymin, xmax, ymax])
        return img[int(float(ymin)):int(float(ymax)),
                   int(float(xmin)):int(float(xmax))]

    def expan_sku_area(self, img, crop_area, crop_ratio=0.075):
        h, w, c = img.shape
        xmin, ymin, xmax, ymax = map(float, crop_area)

        crop_x = (xmax - xmin) * crop_ratio
        crop_y = (ymax - ymin) * crop_ratio
        exp_xmin = max(xmin - crop_x, 0)
        exp_ymin = max(ymin - crop_y, 0)
        exp_xmax = min(xmax + crop_x, w)
        exp_ymax = min(ymax + crop_y, h)
        return exp_xmin, exp_ymin, exp_xmax, exp_ymax

    def _gen_labels(self, img_name, img, bndbox_list):
        """
        img_name: 图片名字
        img: 图片矩阵
        bndbox_list: bndbox 列表
        bndbox 格式：
        {
          xmin:
          ymin:
          xmax:
          ymax:
        }
        """

        # log down image info
        logging.info("handling image {}".format(img_name))
        for idx, bndbox in enumerate(bndbox_list):
            xmin = bndbox.get("xmin")
            xmax = bndbox.get("xmax")
            ymin = bndbox.get("ymin")
            ymax = bndbox.get("ymax")

            subImg = self.splitImg(img, xmin, xmax, ymin, ymax)
            if subImg is None:
                logging.warn(
                    "failed to split image in bndbox {}".format(bndbox))
                continue

            pos = str(int(float(xmin))) + '-' + str(int(
                float(ymin))) + '-' + str(int(float(xmax))) + '-' + str(
                    int(float(ymax)))
            imgname = "{}_{}.jpg".format(img_name, pos)
            #           im = imgname.split('_')
            output = os.path.join(self.output_path, imgname)

            cv2.imwrite(output, subImg)

        logging.info("image {} handled".format(img_name))

    def gen_labels(self, img_name, img, bndbox_list):
        self._pool.spawn(self._gen_labels, img_name, img, bndbox_list)

    def join_all(self):
        """
        等待所有异步任务完成
        """
        self._pool.join()


if __name__ == '__main__':
    # 只需要初始化一次
    generator = LabelGenerator(output_path="/tmp/test_label_generator")
    fn = "/home/abc/test.jpg"  # 文件名
    img = cv2.imread(fn)
    bndbox_list = [
        {
            "xmin": 100,
            "ymin": 100,
            "xmax": 200,
            "ymax": 200
        },
        {
            "xmin": 300,
            "ymin": 300,
            "xmax": 500,
            "ymax": 500
        },
    ]
    # 异步调用
    generator._gen_labels(fn, img, bndbox_list)

    # 等待所有线程执行完毕
    generator.join_all()
