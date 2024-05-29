# -*- coding: utf-8 -*-

__version__ = '2.0'
__author__ = 'LCY'

AUTHOR_INFO = ("基于YOLO的肿瘤图像检测系统v1.0\n"
               "作者：LCY\n"
               "github: https://github.com/death1024\n"
               "")

ENV_CONFIG = ("[配置环境]\n"
              "请按照给定的python版本配置环境，否则可能会因依赖不兼容而出错\n"
              "(1)使用anaconda新建python3.10环境:\n"
              "conda create -n env_rec python=3.10\n"
              "(2)激活创建的环境:\n"
              "conda activate env_rec\n"
              "(3)使用pip安装所需的依赖，可通过requirements.txt:\n"
              "pip install -r requirements.txt\n")

with open('./环境配置.txt', 'w', encoding='utf-8') as f:
    f.writelines(ENV_CONFIG + "\n\n" + AUTHOR_INFO)
