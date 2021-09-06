import cv2
import numpy as np
import time
from AIDetector_pytorch import Detector
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QLineEdit, QFileDialog, QMessageBox, QSlider
from monitor_system import Ui_MainWindow

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.timer_video = QtCore.QTimer()
        self.setupUi(self)
        self.init_logo()
        self.cap = cv2.VideoCapture()
        self.det = Detector()

        self.func_status = {}
        self.func_status['headpose'] = None
        self.frame_mask = None  # 做一个相同尺寸格式的图片mask
        self.editAlertAreaFlag = False
        self.postion = []

        # max = a if a>b else b
        self.frame_h = 720
        self.frame_w = 960
        # 适当调节
        self.fps = 30
        self.timerStride = 30
        self.filePath = "D:/"
        self.transparency = 0.5

        self.init_slots()


    def init_slots(self):
        self.timer_video.timeout.connect(self.show_video_frame)

        # 摄像头IP
        self.lineEdit1.setAlignment(Qt.AlignLeft)
        self.lineEdit1.setEchoMode(QLineEdit.Normal)
        self.lineEdit1.setText("0")
        # 测试IP
        self.pushButton1.clicked.connect(self.testIP)

        # 文件保存地址
        self.lineEdit2.setAlignment(Qt.AlignLeft)
        self.lineEdit2.setEchoMode(QLineEdit.Normal)
        self.lineEdit2.setText(self.filePath)
        # 文件夹选择
        self.pushButton2.clicked.connect(self.showFilePath)

        # 智能监控
        self.pushButton3.clicked.connect(self.monitorControl)

        # 警戒区域编辑
        self.pushButton4.clicked.connect(self.editAlertArea)
        self.pushButton4.setDisabled(True)

        # 警戒区域清空
        self.pushButton5.clicked.connect(self.clearAlertArea)
        self.pushButton5.setDisabled(True)

        # 警戒区域透明度
        self.horizontalSlider1.valueChanged.connect(self.changeTransparency)


    def init_logo(self):
        pix = QtGui.QPixmap('images/logo.jpg')
        self.label1.setScaledContents(True)
        self.label1.setPixmap(pix)


    def testIP(self):
        # 验证其他情况还没写........
        if self.lineEdit1.text() == "0":
            flag = self.cap.open(0)
        else:
            flag = self.cap.open(self.lineEdit1.text())

        if flag == False:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            reply = QMessageBox.information(self, "IP有效", "测试通过，可以使用", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            self.cap.release()


    def showFilePath(self):
        dir_choose = QFileDialog.getExistingDirectory(self,"选取文件夹","D:/")
        if dir_choose == "":
            return
        self.lineEdit2.setText(dir_choose)
        self.filePath = dir_choose


    def monitorControl(self):
        if self.lineEdit1.text() == "0":
            self.localCameraControl(0)
        else:
            self.localCameraControl(self.lineEdit1.text())


    def localCameraControl(self, IP):
        if not self.timer_video.isActive():
            # 默认使用第一个本地camera
            flag = self.cap.open(IP)
            if flag == False:
                QtWidgets.QMessageBox.warning(self, u"Warning", u"打开摄像头失败", buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.frame_mask = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)  # 做一个相同尺寸格式的图片mask

                time_to_str = time.strftime('%Y%m%d%H%M%S')
                self.out = cv2.VideoWriter(self.filePath + '/review%s.mp4'% time_to_str, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), self.fps,
                                           (int(self.frame_w), int(self.frame_h)))

                self.timer_video.start(self.timerStride)
                self.pushButton1.setDisabled(True)
                self.pushButton2.setDisabled(True)
                self.pushButton3.setText(u"停止")
                self.pushButton4.setDisabled(False)
                if len(self.postion) != 0:
                    self.pushButton5.setDisabled(False)

        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label1.clear()
            self.init_logo()
            self.pushButton1.setDisabled(False)
            self.pushButton2.setDisabled(False)
            self.pushButton3.setDisabled(False)
            self.pushButton4.setDisabled(True)
            self.pushButton5.setDisabled(True)
            self.pushButton3.setText(u"开始")


    def editAlertArea(self):
        if self.pushButton4.text() == u"开":
            self.pushButton4.setText(u"关")
            self.editAlertAreaFlag = True
        else:
            self.pushButton4.setText(u"开")
            self.editAlertAreaFlag = False

    def clearAlertArea(self):
        self.frame_mask = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)  # 做一个相同尺寸格式的图片mask
        self.postion = []
        self.pushButton5.setDisabled(True)

    def changeTransparency(self):
        self.transparency = int(self.horizontalSlider1.value())/100.0

    def mousePressEvent(self, event):
        if self.editAlertAreaFlag != True:
            return
        s = event.windowPos()
        self.setMouseTracking(True)
        point = (int(s.x())-10, int(s.y())-10)
        self.postion.append(point)
        self.pushButton5.setDisabled(False)


    def show_video_frame(self):
        flag, img = self.cap.read()
        if img is not None:

            im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR格式转换RGB
            im = cv2.resize(im, (self.frame_w, self.frame_h))  # 改变输入尺寸

            result = self.det.feedCap(im, self.func_status, self.frame_mask)
            result = result['frame']
            if len(self.postion) != 0:
                cv2.fillPoly(self.frame_mask, [np.array(self.postion)], (255, 0, 0))  # 警戒区内数字填充255，0，0成为mask
            result = cv2.addWeighted(result, 1.0, self.frame_mask, self.transparency, 0.0)  # 叠加掩码图片进实时图，第三个参数为透明度

            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            self.out.write(result)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(result.data, result.shape[1], result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.label1.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.label1.clear()
            self.pushButton1.setDisabled(False)
            self.pushButton2.setDisabled(False)
            self.pushButton3.setDisabled(False)
            self.pushButton4.setDisabled(False)
            self.pushButton5.setDisabled(False)
            self.init_logo()


