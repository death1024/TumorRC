import sys
import os
import subprocess

from QtFusion.path import abs_path


def run_script(script_path):

    python_path = sys.executable

    command = f'"{python_path}" -m streamlit run "{script_path}"'

    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print("脚本运行出错。")


if __name__ == "__main__":
    script_path = abs_path("Recognition_UI.py")

    run_script(script_path)
