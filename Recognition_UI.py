import random
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from QtFusion.path import abs_path
from QtFusion.utils import drawRectBox

from LoggerRes import ResultLogger, LogTable
from YOLOv8v5Model import YOLOv8v5Detector
from datasets.Tumor.label_name import Label_list
from style_css import def_css_hitml
from utils_web import save_uploaded_file, concat_results, load_default_image, get_camera_names
import tempfile
import os
from Classification import Classification_CNN
from Classification_Resnet import Classification_Resnet





class Detection_UI:

    def __init__(self):
        self.cls_name = Label_list
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                       range(len(self.cls_name))]

        self.title = "肿瘤图像检测系统"
        self.setup_page()
        def_css_hitml()

        self.model_type = None
        self.model_type_c = None
        self.conf_threshold = 0.25  
        self.iou_threshold = 0.5 

        self.selected_camera = None
        self.file_type = None
        self.uploaded_file = None
        self.uploaded_video = None
        self.custom_model_file = None  
        self.page = None

        self.detection_result = None
        self.detection_location = None
        self.detection_confidence = None
        self.detection_time = None
        self.classification_result = None

        self.display_mode = None 
        self.close_flag = None  
        self.close_placeholder = None 
        self.image_placeholder = None 
        self.image_placeholder_res = None  
        self.table_placeholder = None 
        self.log_table_placeholder = None 
        self.selectbox_placeholder = None 
        self.selectbox_target = None 
        self.progress_bar = None 

        self.saved_log_data = abs_path("tempDir/log_table_data.csv", path_type="current")

        if 'logTable' not in st.session_state:
            st.session_state['logTable'] = LogTable(self.saved_log_data)

        if 'available_cameras' not in st.session_state:
            st.session_state['available_cameras'] = get_camera_names()
        self.available_cameras = st.session_state['available_cameras']

        self.logTable = st.session_state['logTable']

        if 'model' not in st.session_state:
            st.session_state['model'] = YOLOv8v5Detector()

        self.model = st.session_state['model']
        self.model.load_model(model_path=abs_path("weights/best-yolov8n.pt", path_type="current"))

        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                       range(len(self.model.names))]
        self.setup_sidebar()  

    def setup_page(self):
        st.set_page_config(
            page_title=self.title,
            page_icon="👁‍🗨",
            initial_sidebar_state="expanded"
        )

    def setup_sidebar(self):

        st.sidebar.header("识别目标配置")
        page_names = ["实时摄像头识别","上传文件识别"]
        self.page = st.sidebar.radio('选择识别配置',page_names)
        st.sidebar.write("目前选择的是:",self.page)
        
        if self.page == "实时摄像头识别":
            self.selected_camera = st.sidebar.selectbox("选择摄像头", self.available_cameras)
            st.toast("点击'开始运行'，启动摄像头")
        else:
            self.file_type = st.sidebar.selectbox("选择文件类型", ["图片文件", "视频文件"])
            if self.file_type == "图片文件":
                self.uploaded_file = st.sidebar.file_uploader("上传图片", type=["jpg", "png", "jpeg"])
                st.toast("上传图片后点击'开始运行'，检测目标图片")
            elif self.file_type == "视频文件":
                self.uploaded_video = st.sidebar.file_uploader("上传视频文件", type=["mp4"])
                st.toast("上传视频后点击'开始运行'，检测目标视频")


    def load_model_file(self):
        if self.custom_model_file:
            self.model.load_model(self.custom_model_file)
        else:
            pass  # 载入

    

    def process_camera_or_file(self):

        if self.page == "实时摄像头识别" :
            self.logTable.clear_frames()  
            
            self.close_flag = self.close_placeholder.button(label="停止")

            cap = cv2.VideoCapture(int(self.selected_camera))

            
            total_frames = 1000
            current_frame = 0
            self.progress_bar.progress(0) 
            while cap.isOpened() and not self.close_flag:
                ret, frame = cap.read()
                if ret:
                    image, detInfo, _ = self.frame_process(frame, "Camera: " + self.selected_camera)

                    new_width = 1080
                    new_height = int(new_width * (9 / 16))
                    resized_image = cv2.resize(image, (new_width, new_height)) 
                    resized_frame = cv2.resize(frame, (new_width, new_height))

                    if self.display_mode == "单画面显示":
                        self.image_placeholder.image(resized_image, channels="BGR", caption="摄像头画面")
                    else:
                        self.image_placeholder.image(resized_frame, channels="BGR", caption="原始画面")
                        self.image_placeholder_res.image(resized_image, channels="BGR", caption="识别画面")
                    self.logTable.add_frames(image, detInfo, cv2.resize(frame, (640, 640)))

                    progress_percentage = int((current_frame / total_frames) * 100)
                    self.progress_bar.progress(progress_percentage)
                    current_frame = (current_frame + 1) % total_frames
                else:
                    st.error("无法获取图像。")
                    break

            if self.close_flag:
                self.logTable.save_to_csv()
                self.logTable.update_table(self.log_table_placeholder)
                cap.release()

            self.logTable.save_to_csv()
            self.logTable.update_table(self.log_table_placeholder)
            cap.release()
        else:
            if self.uploaded_file is not None:
                self.logTable.clear_frames()
                self.progress_bar.progress(0)

                source_img = self.uploaded_file.read()
                file_bytes = np.asarray(bytearray(source_img), dtype=np.uint8)
                image_ini = cv2.imdecode(file_bytes, 1)
                with tempfile.NamedTemporaryFile(delete=False, suffix='-' + self.uploaded_file.name) as tmp_file:
                    tmp_file.write(source_img)
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())
                    # 获取路径
                    tmp_path = tmp_file.name
                st.write("文件已保存到:", tmp_path)

                image, detInfo, select_info = self.frame_process(image_ini, self.uploaded_file.name,tmp_path)

                self.selectbox_target = self.selectbox_placeholder.selectbox("目标过滤", select_info, key="22113")

                self.logTable.save_to_csv()
                self.logTable.update_table(self.log_table_placeholder)
                new_width = 1920
                new_height = int(new_width * (9 / 16))
                resized_image = cv2.resize(image, (new_width, new_height))
                resized_frame = cv2.resize(image_ini, (new_width, new_height))
                if self.display_mode == "单画面显示":
                    self.image_placeholder.image(resized_image, channels="BGR", caption="图片显示")
                else:
                    self.image_placeholder.image(resized_frame, channels="BGR", caption="原始画面")
                    self.image_placeholder_res.image(resized_image, channels="BGR", caption="识别画面")

                self.logTable.add_frames(image, detInfo, cv2.resize(image_ini, (640, 640)))
                self.progress_bar.progress(100)

            elif self.uploaded_video is not None:
                self.logTable.clear_frames()
                self.close_flag = self.close_placeholder.button(label="停止")

                video_file = self.uploaded_video
                tfile = tempfile.NamedTemporaryFile()
                tfile.write(video_file.read())
                cap = cv2.VideoCapture(tfile.name)

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                total_length = total_frames / fps if fps > 0 else 0

                self.progress_bar.progress(0)

                current_frame = 0
                while cap.isOpened() and not self.close_flag:
                    ret, frame = cap.read()
                    if ret:
                        image, detInfo, _ = self.frame_process(frame, self.uploaded_video.name)

                        new_width = 1080
                        new_height = int(new_width * (9 / 16))
                        resized_image = cv2.resize(image, (new_width, new_height))
                        resized_frame = cv2.resize(frame, (new_width, new_height))
                        if self.display_mode == "单画面显示":
                            self.image_placeholder.image(resized_image, channels="BGR", caption="视频画面")
                        else:
                            self.image_placeholder.image(resized_frame, channels="BGR", caption="原始画面")
                            self.image_placeholder_res.image(resized_image, channels="BGR", caption="识别画面")

                        self.logTable.add_frames(image, detInfo, cv2.resize(frame, (640, 640)))

                        if total_length > 0:
                            progress_percentage = int(((current_frame + 1) / total_frames) * 100)
                            self.progress_bar.progress(progress_percentage)

                        current_frame += 1
                    else:
                        break
                if self.close_flag:
                    self.logTable.save_to_csv()
                    self.logTable.update_table(self.log_table_placeholder)
                    cap.release()

                self.logTable.save_to_csv()
                self.logTable.update_table(self.log_table_placeholder)
                cap.release()

            else:
                st.warning("请选择摄像头或上传文件。")

    def toggle_comboBox(self, frame_id):


        if len(self.logTable.saved_results) > 0:
            frame = self.logTable.saved_images_ini[-1] 
            image = frame  

            for i, detInfo in enumerate(self.logTable.saved_results):
                if frame_id != -1:

                    if frame_id != i:
                        continue

                if len(detInfo) > 0:
                    name, bbox, conf, use_time, p_label, cls_id = detInfo
                    label = '%s %.0f%%' % (p_label, conf * 100) 

                    disp_res = ResultLogger()  
                    res = disp_res.concat_results(name, bbox, str(round(conf, 2)), str(round(use_time, 2)),p_label) 
                    self.table_placeholder.table(res) 

                    if len(self.logTable.saved_images_ini) > 0:
                        if len(self.colors) < cls_id:
                            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                                           range(cls_id+1)]
                        image = drawRectBox(image, bbox, alpha=0.2, addText=label,
                                            color=self.colors[cls_id]) 

            new_width = 1080
            new_height = int(new_width * (9 / 16))
            resized_image = cv2.resize(image, (new_width, new_height))
            resized_frame = cv2.resize(frame, (new_width, new_height))

            if self.display_mode == "单画面显示":
                self.image_placeholder.image(resized_image, channels="BGR", caption="识别画面")
            else:
                self.image_placeholder.image(resized_frame, channels="BGR", caption="原始画面")
                self.image_placeholder_res.image(resized_image, channels="BGR", caption="识别画面")
                

    def frame_process(self, image, file_name,img_path):

        image = cv2.resize(image, (640, 640))
        pre_img = self.model.preprocess(image)  

        params = {'conf': self.conf_threshold, 'iou': self.iou_threshold}
        self.model.set_param(params)
        
        t1 = time.time()
        pred = self.model.predict(pre_img)  
        t2 = time.time()
        use_time = t2 - t1  

        det = pred[0] 

        detInfo = []
        select_info = ["全部目标"]
        if(self.model_type_c == "CNN"):
            pred_label = Classification_CNN(img_path)
            if pred_label == "Glioma":
                p_label = "神经胶质瘤"
            elif pred_label == "Meninigioma":
                p_label = "脑（脊）膜瘤"
            elif pred_label == "Notumor":
                p_label = "正常"
            elif pred_label == "Pituitary":
                p_label = "垂体肿瘤"
            else:
                p_label = pred_label
        elif(self.model_type_c == "ResNet-SNGP"):
            pred_label = Classification_Resnet(img_path)
            if pred_label == "glioma":
                p_label = "神经胶质瘤"
            elif pred_label == "meningioma":
                p_label = "脑（脊）膜瘤"
            elif pred_label == "notumor":
                p_label = "正常"
            elif pred_label == "pituitary":
                p_label = "垂体肿瘤"
            else:
                p_label = pred_label
 
        if det is not None and len(det):
            det_info = self.model.postprocess(pred)  
            if len(det_info):
                disp_res = ResultLogger()
                res = None
                cnt = 0

                for info in det_info:
                    name, bbox, conf, cls_id = info['class_name'], info['bbox'], info['score'], info['class_id']
                    label = '%s %.0f%%' % (p_label, conf * 100)

                    res = disp_res.concat_results(name, bbox, str(round(conf, 2)), str(round(use_time, 2)), p_label)

                    image = drawRectBox(image, bbox, alpha=0.2, addText=label, color=self.colors[cls_id])
                    self.logTable.add_log_entry(file_name, name, bbox, conf, use_time, p_label)
                    detInfo.append([name, bbox, conf, use_time,p_label, cls_id])
                    select_info.append(name + "-" + str(cnt))
                    cnt += 1

                self.table_placeholder.table(res)

        return image, detInfo, select_info


    
    def frame_table_process(self, frame, caption):
        self.image_placeholder.image(frame, channels="BGR", caption=caption)

        detection_result = "None"
        detection_location = "[0, 0, 0, 0]"
        detection_confidence = str(random.random())
        detection_time = "0.00s"

        res = concat_results(detection_result, detection_location, detection_confidence, detection_time)
        self.table_placeholder.table(res)
        cv2.waitKey(1)

    def setupMainWindow(self):

        st.title(self.title)  
        st.divider()
        st.write("ZJU生物医学工程-------------https://github.com/death1024")
        st.divider()
        
        tab1,tab2 = st.tabs(["模型设置","运行页"])
        
        with tab1:
            column1, column2 = st.columns([3, 3])
            with column1:
                with st.popover(":rainbow[肿瘤识别模型设置]"):
                    self.model_type = st.selectbox("选择模型类型", ["YOLOv8", "其他模型"])

                    model_file_option = st.radio("模型文件", ["默认", "自定义"])
                    if model_file_option == "自定义":
                        model_file = st.file_uploader("选择.pt文件", type="pt")

                        if model_file is not None:
                            self.custom_model_file = save_uploaded_file(model_file)
                            self.model.load_model(model_path=self.custom_model_file)
                            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                                        range(len(self.model.names))]
                    elif model_file_option == "默认":
                        self.model.load_model(model_path=abs_path("weights/best-yolov8n.pt", path_type="current"))
                        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                                    range(len(self.model.names))]    
                    
                    self.conf_threshold = float(st.slider("置信度阈值", min_value=0.0, max_value=1.0, value=0.25))
                    self.iou_threshold = float(st.slider("IOU阈值", min_value=0.0, max_value=1.0, value=0.5))
            with column2:
                with st.popover(":rainbow[肿瘤分类模型设置]"):
                    self.model_type_c = st.selectbox("选择模型类型", ["CNN", "ResNet-SNGP"])

        with tab2:
            col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 2, 1])

            with col1:
                self.display_mode = st.radio("显示模式", ["画面对比显示"])

            if self.display_mode == "画面对比显示":
                self.image_placeholder = st.empty()
                self.image_placeholder_res = st.empty()
                if not self.logTable.saved_images_ini:
                    self.image_placeholder.image(load_default_image(), caption="原始画面")
                    self.image_placeholder_res.image(load_default_image(), caption="识别画面")

            self.progress_bar = st.progress(0)

            res = concat_results("None", "[0, 0, 0, 0]", "0.00", "0.00s", "None")
            self.table_placeholder = st.empty()
            self.table_placeholder.table(res)

            st.divider()
            if st.button("导出结果"):
                self.logTable.save_to_csv()
                res = self.logTable.save_frames_file()
                st.write("😎识别结果文件已经保存：" + self.saved_log_data)
                if res:
                    st.write(f"😎结果的视频/图片文件已经保存：{res}")
                self.logTable.clear_data()

            self.log_table_placeholder = st.empty()
            self.logTable.update_table(self.log_table_placeholder)

            with col5:
                st.write("")
                self.close_placeholder = st.empty()

            with col2:
                self.selectbox_placeholder = st.empty()
                detected_targets = ["全部目标"]

                for i, info in enumerate(self.logTable.saved_results):
                    name, bbox, conf, use_time, p_label ,cls_id = info
                    detected_targets.append(name + "-" + str(i))
                self.selectbox_target = self.selectbox_placeholder.selectbox("目标过滤", detected_targets)

                for i, info in enumerate(self.logTable.saved_results):
                    name, bbox, conf, use_time, p_label,cls_id = info
                    if self.selectbox_target == name + "-" + str(i):
                        self.toggle_comboBox(i)
                    elif self.selectbox_target == "全部目标":
                        self.toggle_comboBox(-1)

            with col4:
                st.write("")
                run_button = st.button("开始运行")

                if run_button:
                    self.process_camera_or_file()  
                else:
                    if not self.logTable.saved_images_ini:
                        self.image_placeholder.image(load_default_image(), caption="原始画面")
                        if self.display_mode == "画面对比显示":
                            self.image_placeholder_res.image(load_default_image(), caption="识别画面")
                              

if __name__ == "__main__":
    app = Detection_UI()
    app.setupMainWindow()
