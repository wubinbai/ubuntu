#! usr/bin/python
import os 
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
 
## ID pssd
stuID = '2018401160'
pw = '075015'
 
# urls
url = 'https://ids.xmu.edu.cn/authserver/login?service=https%3A%2F%2Fi.xmu.edu.cn%2FEIP%2Fcaslogin.jsp'
SLEEP_TIME = 10


def autoSignIn():
    # change this debug
    driver = webdriver.Chrome(executable_path='/home/wb/Downloads/chrome//chromedriver')
    driver.get(url)
    driver.maximize_window()
    time.sleep(SLEEP_TIME)
    
    ### for debug only:
    driver.find_element_by_xpath('//input[@id="username"]').send_keys(stuID)
    driver.find_element_by_xpath('//input[@id="password"]').send_keys(pw)
    driver.find_element_by_xpath('//input[@id="password"]').send_keys(Keys.ENTER)
    time.sleep(SLEEP_TIME)
    os.system('./mouse_1024_768_feb.sh')


if __name__ == '__main__':
        autoSignIn()
