import os
import subprocess

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.webdriver import WebDriver

chrome_options = Options()
chrome_options.binary_location = "/usr/bin/chromium-browser"
chrome_options.add_experimental_option("detach", True)

PORT = 8585


def open_log_web() -> WebDriver:
    subprocess.Popen(
        f"tensorboard --logdir=logs --host=0.0.0.0 --port={PORT}", shell=True
    )

    driver: WebDriver = webdriver.Chrome(options=chrome_options)

    driver.get(f"http://localhost:{PORT}/")

    return driver


def close_tensorboard():
    if os.name == "nt":
        subprocess.call("TASKKILL /F /IM tensorboard.exe", shell=True)
    else:
        subprocess.call(f"kill -9 `lsof -t -i:{PORT}`", shell=True)
