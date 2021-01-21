#! usr/bin/python
# -*- coding: utf-8 -*-
import os 
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
 
# 学号密码
stuID = '2018401160'
pw = '075015'
 
# 打卡网址
url = 'https://ids.xmu.edu.cn/authserver/login?service=https%3A%2F%2Fi.xmu.edu.cn%2FEIP%2Fcaslogin.jsp'
# login_pate = 'https://ids.xmu.edu.cn/authserver/login?service=https%3A%2F%2Fi.xmu.edu.cn%2FEIP%2Fcaslogin.jsp'
# temp_url = 'https://ids.xmu.edu.cn/authserver/login?service=https://xmuxg.xmu.edu.cn/login/cas/xmu'

start = time.perf_counter()
 
# 日志记录
def log():
    logfile = open("./打卡记录.txt", 'a+', encoding='utf-8')
 
    timeNow = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    duration = time.perf_counter() - start
 
    info = "{0}\n今日健康打卡完成！共耗时：{1}\n\n".format(timeNow, duration)
    logfile.write(info)
    logfile.close()
 
    print(info)
 
# 判断是否在打卡时间内
def timeFlag():
    startTime = time.strftime("%Y-%m-%d 07:00:00", time.localtime())
    endTime = time.strftime("%Y-%m-%d 19:30:00", time.localtime())
 
    startTimeStamp = time.mktime(time.strptime(startTime, "%Y-%m-%d %H:%M:%S"))
    endTimeStamp = time.mktime(time.strptime(endTime, "%Y-%m-%d %H:%M:%S"))
 
    if startTimeStamp < time.time() < endTimeStamp:
        return True
    else:
        print("不在打卡时间!")
        return False
 
# 自动打卡
def autoSignIn():
    #driver = webdriver.Chrome(executable_path='/home/wb/Downloads/chrome/chromedriver')
    driver = webdriver.Chrome(executable_path='/home/b/Downloads/chromedriver')
    driver.get(url)
    driver.maximize_window()
    time.sleep(1)
 
    #driver.find_element_by_xpath('//*[@id="loginLayout"]/div[3]/div[2]/div/button[2]').click()
    time.sleep(1)
 
    driver.find_element_by_xpath('//input[@id="username"]').send_keys(stuID)
    driver.find_element_by_xpath('//input[@id="password"]').send_keys(pw)
    driver.find_element_by_xpath('//input[@id="password"]').send_keys(Keys.ENTER)
    time.sleep(2)
    os.system('./mouse_action.sh')
    pause


if __name__ == '__main__':
    if timeFlag():
        autoSignIn()
        #log()
