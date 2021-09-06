from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
import numpy as np

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)


def plot_bboxes(image, bboxes, frame_mask, line_thickness=None):
    alert_obj = 0
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        # if cls_id in ['smoke', 'phone', 'eat']:
        #     color = (0, 0, 255)
        # else:
        #     color = (0, 255, 0)
        # if cls_id == 'eat':
        #     cls_id = 'eat-drink'
        c1, c2 = (x1, y1), (x2, y2)
        # cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        # t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
        #             [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        cv2.putText(image, '{}'.format(cls_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        center_xy = (int(np.average([x1, x2])), int(np.average([y1, y2])))  # 计算中心点
        if (frame_mask[(center_xy[1], center_xy[0])] == [255, 0, 0]).all():  # 中心点在警戒区
            obj_color = (255, 0, 0)  # 改变中心点颜色
            alert_obj += 1
        else:
            obj_color = (0, 255, 0)  # 改变中心点颜色
        cv2.circle(image, center_xy, 5, obj_color, 6)  # 开始画点
        cv2.rectangle(image, c1, c2, obj_color, thickness=tl, lineType=cv2.LINE_AA)

    return image, alert_obj


def update_tracker(target_detector, image, frame_mask):
    new_faces = []
    _, bboxes = target_detector.detect(image)

    bbox_xywh = []
    confs = []
    bboxes2draw = []
    face_bboxes = []
    classifications = []
    if len(bboxes):

        # Adapt detections to deep sort input format
        for x1, y1, x2, y2, lbl, conf in bboxes:
            obj = [
                int((x1 + x2) / 2), int((y1 + y2) / 2),
                x2 - x1, y2 - y1
            ]
            bbox_xywh.append(obj)
            confs.append(conf)
            classifications.append(lbl)

        xywhs = torch.Tensor(bbox_xywh)
        confss = torch.Tensor(confs)

        # Pass detections to deepsort
        outputs = deepsort.update(xywhs, confss, image)

        for value, classification in zip(list(outputs), classifications):
            x1, y1, x2, y2, track_id = value
            bboxes2draw.append(
                (x1, y1, x2, y2, classification, track_id)
            )

    image, alert_obj = plot_bboxes(image, bboxes2draw, frame_mask)

    return image, new_faces, face_bboxes, alert_obj