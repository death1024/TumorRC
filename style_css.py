import base64
import streamlit as st

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as file:
        data = file.read()
    return base64.b64encode(data).decode()

def def_css_hitml():
    st.markdown("""
        <style>
        /* 全局样式 */
        body {
            font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif;
            background-color: #f0f2f6; /* 更浅的背景色 */
            color: #333; /* 更深的文本颜色 */
        }

        /* 按钮样式 */
        .stButton > button {
            border: none;
            color: #fff; /* 文本颜色 */
            padding: 10px 20px; /* 内边距 */
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 2px 1px;
            cursor: pointer;
            border-radius: 8px;
            background-color: #4a90e2; /* 主题色 */
            box-shadow: 0 2px 4px 0 rgba(0,0,0,0.2);
            transition-duration: 0.4s;
        }
        .stButton > button:hover {
            background-color: #357ae8; /* 鼠标悬停时的颜色 */
            box-shadow: 0 8px 12px 0 rgba(0,0,0,0.24);
        }

        /* 侧边栏样式 */
        .css-1lcbmhc.e1fqkh3o0 {
            background-color: #3f51b5; /* 侧边栏背景色 */
            color: #fff; /* 侧边栏文本颜色 */
            border-right: 2px solid #DDD;
        }

        /* Radio 按钮样式 */
        .stRadio > label {
            display: inline-flex;
            align-items: center;
            cursor: pointer;
            color: #555; /* 单选按钮文本颜色 */
            margin-right: 10px;
        }
        .stRadio > label > span:first-child {
            background-color: #fff; /* 单选按钮未选中时的背景色 */
            border: 1px solid #ccc;
            width: 1em;
            height: 1em;
            border-radius: 50%;
            margin-right: 5px;
            display: inline-block;
        }
        .stRadio > label > input[type="radio"]:checked + span:first-child {
            background-color: #4a90e2; /* 单选按钮选中时的背景色 */
        }

        /* 滑块样式 */
        .stSlider .thumb {
            background-color: #4a90e2; /* 滑块的颜色 */
            box-shadow: none;
        }
        .stSlider .track {
            background-color: #ccc; /* 滑轨的颜色 */
        }

        /* 表格样式 */
        table {
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 16px; /* 调整字体大小 */
            font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif;
            min-width: 400px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        thead tr {
            background-color: #4a90e2; /* 表头背景色 */
            color: #fff; /* 表头文本颜色 */
            text-align: left;
        }
        th, td {
            padding: 12px 18px; /* 调整单元格内边距 */
        }
        tbody tr {
            border-bottom: 2px solid #ddd;
        }
        tbody tr:nth-of-type(even) {
            background-color: #f5f5f5; /* 偶数行背景色 */
        }
        tbody tr:last-of-type {
            border-bottom: 3px solid #4a90e2; /* 最后一行的底部边框颜色 */
        }
        tbody tr:hover {
            background-color: #e0e0e0; /* 鼠标悬停时的背景色 */
        }
        </style>
        """, unsafe_allow_html=True)


