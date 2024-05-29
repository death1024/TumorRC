import sys  
import time  
import cv2 
from QtFusion.path import abs_path
from QtFusion.config import QF_Config
from QtFusion.widgets import QMainWindow 
from QtFusion.handlers import MediaHandler  
from QtFusion.utils import drawRectBox 
from QtFusion.utils import get_cls_color 
from PySide6 import QtWidgets, QtCore  
from YOLOv8v5Model import YOLOv8v5Detector 
from datasets.Tumor.label_name import Label_list

QF_Config.set_verbose(False)


class MainWindow(QMainWindow):  
    def __init__(self):  
        super().__init__()  
        self.resize(850, 500)  
        self.label = QtWidgets.QLabel(self)  
        self.label.setGeometry(0, 0, 850, 500)  

    def keyPressEvent(self, event): 
        if event.key() == QtCore.Qt.Key.Key_Q: 
            self.close() 


def frame_process(image): 
    image = cv2.resize(image, (850, 500))  
    pre_img = model.preprocess(image)  

    t1 = time.time() 
    pred = model.predict(pre_img) 
    t2 = time.time() 
    use_time = t2 - t1  

    print("推理时间: %.2f" % use_time) 
    det = pred[0] 

    if det is not None and len(det):
        det_info = model.postprocess(pred) 
        for info in det_info:  
            name, bbox, conf, cls_id = info['class_name'], info['bbox'], info['score'], info[
                'class_id'] 
            label = '%s %.0f%%' % (name, conf * 100)  
            image = drawRectBox(image, bbox, alpha=0.2, addText=label, color=colors[cls_id]) 

    window.dispImage(window.label, image)


cls_name = Label_list  

model = YOLOv8v5Detector() 
model.load_model(abs_path("weights/best-yolov8n.pt", path_type="current")) 
colors = get_cls_color(model.names) 

app = QtWidgets.QApplication(sys.argv)  
window = MainWindow() 

filename = abs_path("test_media/肿瘤识别.mp4", path_type="current") 
videoHandler = MediaHandler(fps=30)  
videoHandler.frameReady.connect(frame_process)  
videoHandler.setDevice(filename)  
videoHandler.startMedia()  

window.show()
sys.exit(app.exec())
