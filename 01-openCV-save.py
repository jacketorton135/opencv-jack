#!/usr/bin/env python
# author: Powen Ko  柯博文老師  www.powenko.com
# -*- coding: utf-8 -*-

import cv2  # 匯入 OpenCV 庫，用於影像處理
import numpy as np  # 匯入 NumPy 庫，用於數據操作
from time import gmtime, strftime  # 從 time 模組中匯入 gmtime 和 strftime 用於時間處理

# 開啟電腦攝影機（0 代表默認攝影機）
cap = cv2.VideoCapture(1)

# 進入無窮迴圈，用於不斷讀取影像
count=0
while(True):
    ret, img = cap.read()  # 從攝影機捕捉一張影像，ret 為捕捉狀態，img 為影像資料
    if ret == True:  # 確認成功讀取影像
        cv2.imshow('frame', img)  # 顯示影像在一個名為 'frame' 的視窗中

    key = cv2.waitKey(1)  # 等待按鍵事件，延遲 1 毫秒
    if key & 0xFF == ord('q') or key==27:  # 如果按下 'q' 鍵，退出迴圈
        break
    elif key & 0xFF == ord('s'):  # 如果按下 's' 鍵，儲存當前影像
        if ret == True:  # 確認成功讀取影像
            # 生成當前時間作為檔名（格式為 YYYYMMDDHHMMSS.jpg）
            filename1 = strftime("%Y%m%d%H%M%S_", gmtime()) +str(count) +'.jpg'
            count=count+1
            print(filename1)  # 輸出檔名
            # 將影像大小調整為 224x224
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            # 儲存影像到檔案
            cv2.imwrite(filename=filename1, img=img)

# 釋放攝影機資源
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
