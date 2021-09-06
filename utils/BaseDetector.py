from tracker import update_tracker
import cv2
import time


class baseDet(object):

    def __init__(self):

        self.img_size = 640
        self.threshold = 0.3
        self.stride = 1
        self.alert_obj=0

    def build_config(self):

        self.faceTracker = {}
        self.faceClasses = {}
        self.faceLocation1 = {}
        self.faceLocation2 = {}
        self.frameCounter = 0
        self.currentCarID = 0
        self.recorded = []

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def feedCap(self, im, func_status, frame_mask):

        retDict = {
            'frame': None,
            'faces': None,
            'list_of_ids': None,
            'face_bboxes': []
        }
        self.frameCounter += 1

        im, faces, face_bboxes, alert_obj = update_tracker(self, im, frame_mask)
        if self.alert_obj != alert_obj:
            result = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            time_to_str = time.strftime('%Y%m%d%H%M%S')
            cv2.imwrite("./runs/photos/%s.jpg" % time_to_str, result, [int(cv2.IMWRITE_JPEG_QUALITY), 50])  # 保存图片
            cv2.putText(im, 'warning!', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)  # 文字信息显示
            self.alert_obj=alert_obj

        retDict['frame'] = im
        retDict['faces'] = faces
        retDict['face_bboxes'] = face_bboxes

        return retDict

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")
