import random 
import sys  
import time
from QtFusion.config import QF_Config
import cv2  
from QtFusion.widgets import QMainWindow 
from QtFusion.utils import cv_imread, drawRectBox  
from PySide6 import QtWidgets, QtCore
from QtFusion.path import abs_path
from YOLOv8v5Model import YOLOv8v5Detector  
from datasets.Tumor.label_name import Label_list

QF_Config.set_verbose(False)

cls_name = Label_list  
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(cls_name))]

model = YOLOv8v5Detector() 
model.load_model(abs_path("weights/best-yolov8n.pt", path_type="current"))  

class MainWindow(QMainWindow):  
    def __init__(self): 
        super().__init__() 
        self.resize(850, 500)  
        self.label = QtWidgets.QLabel(self)  
        self.label.setGeometry(0, 0, 850, 500)  

    def keyPressEvent(self, event): 
        if event.key() == QtCore.Qt.Key.Key_Q: 
            self.close()  


if __name__ == '__main__': 
    app = QtWidgets.QApplication(sys.argv)  
    window = MainWindow() 

    img_path = abs_path("test_media/m1.jpg") 
    image = cv_imread(img_path)  

    image = cv2.resize(image, (850, 500))  
    pre_img = model.preprocess(image) 

    t1 = time.time() 
    pred = model.predict(pre_img)  
    t2 = time.time()  
    use_time = t2 - t1  

    det = pred[0]

    if det is not None and len(det):
        det_info = model.postprocess(pred)  
        for info in det_info: 

            name, bbox, conf, cls_id = info['class_name'], info['bbox'], info['score'], info['class_id']
            label = '%s %.0f%%' % (name, conf * 100)  
 
            image = drawRectBox(image, bbox, alpha=0.2, addText=label, color=colors[cls_id])  

    print("预测时间: %.2f" % use_time) 
    window.dispImage(window.label, image) 

    window.show()

    sys.exit(app.exec())
