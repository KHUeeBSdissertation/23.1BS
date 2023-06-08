import cv2, sys
from matplotlib import pyplot as plt
import numpy as np
car_image = cv2.imread('../datasets/licence_dataset/images/test/Cars19.png')
license = cv2.imread('../Korean-license-plate-Generator/AUG/A27tj4323.jpg')
license_gray = cv2.imread('../Korean-license-plate-Generator/AUG/A27tj4323.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('mask',license) #배경 흰색, 로고 검정
# cv2.imshow('gray',license_gray ) #배경 흰색, 로고 검정

# finding teduri with adaptive threshold
thresh1 = cv2.adaptiveThreshold(license_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
cv2.imshow('teduri',thresh1 ) #배경 흰색, 로고 검정
# cany edge detection doenst work properly

# finding closed edge 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
closed = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
total = 0
contours_xy = np.array(contours)
contours_xy.shape

print(len(contours_xy))
cv2.imshow('closed', closed)

# x의 min과 max 찾기
x_min, x_max = 0,0
value = list()
for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][0]) #네번째 괄호가 0일때 x의 값
        value = sorted(value)

print(value)
x_min = value[1]
x_max = value[len(value)-2]
print(x_min)
print(x_max)
 
# y의 min과 max 찾기
y_min, y_max = 0,0
value = list()
for i in range(len(contours_xy)):
    for j in range(len(contours_xy[i])):
        value.append(contours_xy[i][j][0][1]) #네번째 괄호가 0일때 x의 값
        value = sorted(value)

print(value)
y_min = value[1]
y_max = value[len(value)-2]
print(y_min)
print(y_max)

# image trim 하기
x = x_min
y = y_min
w = x_max-x_min
h = y_max-y_min

img_trim = license[y:y+h, x:x+w]
cv2.imshow('org_trim.jpg', img_trim)

cv2.waitKeyEx()
cv2.destroyAllWindows()
