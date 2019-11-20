import cv2
import os
import numpy as np
test1 = []
test2 = []
test3 = []
train1 = []
responses1 = []

#讀圖
for i in range(71,101):
    image = cv2.imread(".\\arrage\\1\\1 (%d).jpg"%(i))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    test1.append(image)
    image = cv2.imread(".\\arrage\\2\\2 (%d).jpg"%(i))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    test2.append(image)
    image = cv2.imread(".\\arrage\\3\\3 (%d).jpg"%(i))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    test3.append(image)

for i in range(1,71):
    image = cv2.imread(".\\arrage\\1\\1 (%d).jpg"%(i))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    train1.append(image)
    responses1.append([1])
    image = cv2.imread(".\\arrage\\2\\2 (%d).jpg"%(i))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    train1.append(image)
    responses1.append([2])
    image = cv2.imread(".\\arrage\\3\\3 (%d).jpg"%(i))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    train1.append(image)
    responses1.append([3])


#轉成np.array
train1 = np.array(train1).astype(np.float32)
responses1 = np.array(responses1)
test1 = np.array(test1).astype(np.float32)
test2 = np.array(test2).astype(np.float32)
test3 = np.array(test3).astype(np.float32)



#圖片攤成一維陣列,讓他能塞進svm.train
train1 = train1.reshape(-1,150*150)
test1 = test1.reshape(-1,150*150)
test2 = test2.reshape(-1,150*150)
test3 = test3.reshape(-1,150*150)


#建立svm及參數
SVM = cv2.ml.SVM_create()
SVM.setType(cv2.ml.SVM_C_SVC)
SVM.setKernel(cv2.ml.SVM_LINEAR)


#訓練(210筆)
SVM.train(train1, cv2.ml.ROW_SAMPLE, responses1)


#測試(90筆)
count = 0
result = SVM.predict(test1)
for i in range(30):
    if result[1][i] == [1.]:
        count+=1
result = SVM.predict(test2)
for i in range(30):
    if result[1][i] == [2.]:
        count+=1
result = SVM.predict(test3)
for i in range(30):
    if result[1][i] == [3.]:
        count+=1

print('分類成功率:%.2f'%((count / 90) * 100))