import os

import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from QtFusion.path import abs_path


def save_uploaded_file(uploaded_file):

    if uploaded_file is not None:
        base_path = "tempDir" 

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        file_path = os.path.join(base_path, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())  # 写入

        return file_path 

    return None  


def concat_results(result, location, confidence, time, types):

    result_data = {
        "识别结果": [result],
        "位置": [location],
        "置信度": [confidence],
        "用时": [time],
        "类型":[types]
    }

    results_df = pd.DataFrame(result_data)
    return results_df


def load_default_image():

    ini_image = abs_path("icon/ini-image.png")
    return Image.open(ini_image)


def get_camera_names():

    camera_names = ["未启用摄像头", "0"]
    max_test_cameras = 3 

    for i in range(max_test_cameras):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened() and str(i) not in camera_names:
            camera_names.append(str(i))
            cap.release()
    if len(camera_names) == 1:
        st.write("未找到可用的摄像头")
    return camera_names
