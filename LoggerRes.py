import os
import time
import cv2
import pandas as pd
from QtFusion.path import abs_path


class ResultLogger:
    def __init__(self):

        self.results_df = pd.DataFrame(columns=["识别结果", "位置", "置信度", "用时", "类型"])

    def concat_results(self, result, location, confidence, time, types):
        result_data = {
            "识别结果": [result],
            "位置": [location],
            "置信度": [confidence],
            "用时": [time],
            "类型": [types]
        }

        new_row = pd.DataFrame(result_data)
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)

        return self.results_df


class LogTable:
    def __init__(self, csv_file_path=None):

        self.csv_file_path = csv_file_path
        self.saved_images = []
        self.saved_images_ini = []
        self.saved_results = []

        columns = ['文件路径', '识别结果', '位置', '置信度', '用时', '类型']


        try:

            if not os.path.exists(csv_file_path):
                empty_df = pd.DataFrame(columns=columns)
                empty_df.to_csv(csv_file_path, index=False, header=True)

            self.data = pd.DataFrame(columns=columns)

        except (FileNotFoundError, pd.errors.EmptyDataError):
            columns = ['文件路径', '识别结果', '位置', '置信度', '用时', '类型']
            self.data = pd.DataFrame(columns=columns)

    def add_frames(self, image, detInfo, img_ini):
        self.saved_images.append(image)
        self.saved_images_ini.append(img_ini)
        self.saved_results = detInfo

    def clear_frames(self):
        self.saved_images = []
        self.saved_images_ini = []
        self.saved_results = []

    def save_frames_file(self):
        if self.saved_images: 
            # 保存
            now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            if len(self.saved_images) == 1:
                # 保存图片
                file_name = abs_path('tempDir/pic_' + str(now_time) + '.png', path_type="current")
                cv2.imwrite(file_name, self.saved_images[0])
                return file_name
            else:
                # 保存视频
                height, width, layers = self.saved_images[0].shape
                size = (width, height)
                file_name = abs_path('tempDir/video_' + str(now_time) + '.avi', path_type="current")
                out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
                for img in self.saved_images:
                    out.write(img)
                out.release()
                return file_name
        return False

    def add_log_entry(self, file_path, recognition_result, position, confidence, time_spent, types):

        position_str = str(position)
        file_path = str(file_path)
        new_entry = pd.DataFrame([[file_path, recognition_result, position_str, confidence, time_spent, types]],
                                 columns=['文件路径', '识别结果', '位置', '置信度', '用时', '类型'])

        self.data = pd.concat([new_entry, self.data]).reset_index(drop=True)

        return self.data

    def clear_data(self):
        columns = ['文件路径', '识别结果', '位置', '置信度', '用时', '类型']
        self.data = pd.DataFrame(columns=columns)

    def save_to_csv(self):
        self.data.to_csv(self.csv_file_path, index=False, encoding='utf-8', mode='a', header=False)

    def update_table(self, log_table_placeholder):
        if len(self.data) > 500:
            display_data = self.data.head(500)
        else:
            display_data = self.data

        log_table_placeholder.table(display_data)