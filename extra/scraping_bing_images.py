"""
This script searches for images on www.bing.com and saves the results to a folder
"""

import time
from bs4 import BeautifulSoup
import urllib.request
import shutil
import os

# ____selenium____
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException

"""
Parameters description:

SCROLLS: number of times the url page get scrolled down to load more images
SCROLL_PAUSE_TIME: number of seconds to pause the script after a scroll to give it time to load the new images
SEARCH: the input to the search box
FOLDER_NAME: name of the folder for the result images ( will be created if not exist)
"""

SCROLLS = 5
SCROLL_PAUSE_TIME = 5
SEARCH = 'renovation trash'
FOLDER_NAME = 'renovation_trash'

# selenium activation
def act_selenium():
    category_url = f"https://www.bing.com/images/search?q={'+'.join(SEARCH.split())}&form=RESTAB&first=1"
    # category_url = 'https://www.google.com/search?q=landscape+waste&tbm=isch&hl=en&chips=q:landscape+waste,online_chips:yard+trimmings:X7_U7Gj3MhE%3D&bih=764&biw=1425&rlz=1C5CHFA_enIL948IL949&sa=X&ved=2ahUKEwjKu62amob9AhXOhCcCHbodBWMQ4lYoB3oECAEQMQ'
    options = Options()
    options.headless = False
    options.add_argument('--window-size=1920,1080')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(category_url)
    return driver

# create folder (if not exist)
def mk_folder():
    if not os.path.exists(FOLDER_NAME):
        os.mkdir(FOLDER_NAME)

# scrolling the search page to load more images
def scroll_page(scrolls, driver):
    for scroll in range(scrolls):
        try:
            driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
            time.sleep(SCROLL_PAUSE_TIME)
        except TimeoutException as err:
            print(f"Selenium driver failed to execute script: {err} ")
            continue

def main():
    count = 1
    sel_driver = act_selenium()
    mk_folder()
    scroll_page(SCROLLS, sel_driver)


    # getting content
    html = sel_driver.page_source
    full_content = BeautifulSoup(html, "lxml")
    class_type = ['mimg', 'mimg rms_ing', 'mimg vimgld']
    products = full_content.find_all('img', class_=class_type)

    # Download the file from `url` and save it locally at `FOLDER_NAME`, under 'FOLDER_NAME'+'count'  :
    for product in products:
        try:
            url = product["src"]
            with urllib.request.urlopen(url) as response, open(f'{FOLDER_NAME}/{FOLDER_NAME}{count}.jpeg', 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            count += 1

        except AttributeError:
            continue

    sel_driver.close()


if __name__ == '__main__':
    main()
