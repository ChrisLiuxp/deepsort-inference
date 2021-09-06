from AIDetector_pytorch import Detector
import imutils
import cv2
import numpy as np

name = 'video surveillance'
frame_h = 480
frame_w = 800
frame_mask = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)  # 做一个相同尺寸格式的图片mask
postion = []
add_flag = False  # 是否可以添加区域点

def process_img(original_image):  # 原图处理函数
    processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # BGR格式转换RGB
    processed_img = cv2.resize(processed_img, (frame_w, frame_h))  # 改变输入尺寸
    return processed_img

def MouseEvent1(object):  # 鼠标处理事件响应函数
    pass

def MouseEvent2(object):  # 鼠标处理事件响应函数
    global add_flag
    switch = cv2.getTrackbarPos('add points', name)
    if switch == 0:
        add_flag = False
    else:
        add_flag = True
    # if len(postion) != 0:
    #     print('draw')  # 是否画mask
    #     cv2.fillPoly(frame_mask, [np.array(postion)], (255, 0, 0))  # 警戒区内数字填充255，0，0成为mask

def MouseEvent3(object):  # 鼠标处理事件响应函数
    global postion
    global frame_h
    global frame_w
    global frame_mask
    switch = cv2.getTrackbarPos('del_points', name)
    if switch == 1:
        frame_mask = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)  # 做一个相同尺寸格式的图片mask
        postion = []

def draw_area(event, x, y, flags, param):
    global postion
    if add_flag == True & event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        postion.append(point)
        print('check point number :%s' % len(postion))  # 监控区域点的个数

        if len(postion) != 0:
            cv2.fillPoly(frame_mask, [np.array(postion)], (255, 0, 0))  # 警戒区内数字填充255，0，0成为mask

def main():
    global frame_h
    global frame_w
    global frame_mask

    func_status = {}
    func_status['headpose'] = None

    cv2.namedWindow(name)
    cv2.setMouseCallback(name, draw_area)  # 窗口与回调函数绑定
    cv2.createTrackbar('add points', name, 0, 1, MouseEvent2)
    cv2.createTrackbar('transparency', name, 0, 100, MouseEvent1)
    cv2.createTrackbar('del_points', name, 0, 1, MouseEvent3)

    det = Detector()
    cap = cv2.VideoCapture('D:/Software/DevelopTool/PyCharm/Yolov5/yolov5-master/data/videos/1.mp4')
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)

    # max = a if a>b else b
    frame_h = 648 if int(cap.get(4)) > 648 else int(cap.get(4))
    frame_w = 1152 if int(cap.get(3)) > 1152 else int(cap.get(3))
    frame_mask = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)  # 做一个相同尺寸格式的图片mask

    size = None
    videoWriter = None

    while cap.isOpened():

        # try:
        _, im = cap.read()
        if im is None:
            break

        im = process_img(im)

        result = det.feedCap(im, func_status, frame_mask)
        result = result['frame']
        # result = imutils.resize(result, height=500)

        transparency = cv2.getTrackbarPos('transparency', name)
        result = cv2.addWeighted(result, 1.0, frame_mask, transparency / 100.0, 0.0)  # 叠加掩码图片进实时图

        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        videoWriter.write(result)
        cv2.imshow(name, result)
        cv2.waitKey(t)

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break
        # except Exception as e:
        #     print(e)
        #     break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main()