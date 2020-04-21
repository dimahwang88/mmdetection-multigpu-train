from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import cv2
import numpy as np
import time
import sys
import glob
import os

rec_id = sys.argv[1]

IMAGE_DIR = '/home/dmitriy.khvan/data/%s/' % (rec_id)

config_file = '../configs/cascade_rcnn_r101_fpn_1x_test.py'
checkpoint_file = '../checkpoints/cascade_rcnn_r101_fpn_1x_20181129-d64ebac7.pth'
#checkpoint_file = '../checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'
#config_file = '../configs/faster_rcnn_r50_fpn_1x.py'

model = init_detector(config_file, checkpoint_file, device='cuda:0')

for num, filename in enumerate(sorted(glob.glob(os.path.join(IMAGE_DIR,'*.jpg')))):
    print(filename)
    image = cv2.imread(filename)

    start_time = time.time()
    result = inference_detector(model, filename)
    end_time = time.time()
    print('[DBG] inference time: ' + str(end_time-start_time) + ' s.')
    
    bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)

    labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
    labels = np.concatenate(labels)

    for i in range(len(bboxes)):
        bb = bboxes[i]
        if labels[i] != 0:  continue
        d = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
        cv2.rectangle(image, (d[0], d[1]), (d[2], d[3]), (255,0,0), 2)

    dump_path = "%sdump/dump-%06d.jpg" % (IMAGE_DIR, num)
    print(dump_path)
    cv2.imwrite(dump_path, image)

print('[DBG] detection complete!')


