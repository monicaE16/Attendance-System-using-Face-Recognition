
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot, QTimer, QDate
from PyQt5.QtWidgets import QDialog,QMessageBox
import cv2
import numpy as np
import datetime
import os
from main import *


class Ui_OutputDialog(QDialog):
    def __init__(self):
        super(Ui_OutputDialog, self).__init__()
        loadUi("./outputwindow.ui", self)
        now=QDate.currentDate()
        current_date=now.toString('ddd dd MMMM yyyy')
        current_time=datetime.datetime.now().strftime('%I:%M:%p')   
        self.Date_Label.setText(current_date)
        self.Time_Label.setText(current_time)
        self.image = None

    @pyqtSlot()
    def startVideo(self, camera_name) :
        if len(camera_name) == 1:
            self.capture = cv2.VideoCapture(int(camera_name))
        else:
            self.capture = cv2.VideoCapture(camera_name)
        
        self.timer = QTimer(self) 
        #will return prjoctions of pca data 
        self.projections,self.y,self.eigenvectors,self.mu=prepare_Data()
        
        self.timer.timeout.connect(self.update_frame)  # Connect timeout to the output function
        self.timer.start(40)  # emit the timeout() signal at x=40ms
        
        
        
        
        
        

    def face_rec_(self, frame, projections,y,eigenvectors,mu):
       
        
        name,points,rescaling_factor=start_Reco(frame,projections,y,eigenvectors,mu)
        if(len(points)):
             x1=round(int(points[1])*rescaling_factor[1])
             y1=round(int(points[0])*rescaling_factor[0])
             x2=round(int(points[3])*rescaling_factor[1])
             y2=round(int(points[2])*rescaling_factor[0])
             cv2.rectangle(frame,(x1,y1),(x2,y2),color=(0,255,0),thickness=2)
             cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
             self.name_label.setText(name)
             self.date_label.setText(datetime.datetime.now().strftime('%I:%M:%p'))
        else:
            self.name_label.setText('Unknown face')
        

             #mark_attendance(name)

        
      
        return frame

    def update_frame(self):
        current_time=datetime.datetime.now().strftime('%I:%M:%p')  
        self.Time_Label.setText(current_time)
        ret, self.image = self.capture.read()
        self.displayImage(self.image, self.projections,self.y,self.eigenvectors,self.mu, 1)

    def displayImage(self, image, projections,y,eigenvectors,mu, window=1):
        """
        :param image: frame from camera
        :param encode_list: known face encoding list
        :param class_names: known face names
        :param window: number of window
        :return:
        """
        try:
            image = self.face_rec_(image, projections,y,eigenvectors,mu)
        except Exception as e:
            print(e)
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        outImage = outImage.rgbSwapped()

        if window == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(outImage))
            self.imgLabel.setScaledContents(True)
