import os
# from bs4 import BeautifulSoup as bsoup # to read xml files
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import random
# from PIL import Image, ImageEnhance


import argparse

parser = argparse.ArgumentParser(description='이 프로그램의 설명(그 외 기타등등 아무거나)')    # 2. parser를 만든다.

# 3. parser.add_argument로 받아들일 인수를 추가해나간다.
parser.add_argument('-c', '--carimg', help='car img number', type=int)    # 필요한 인수를 추가
parser.add_argument('-l', '--lpimg', help='license plate img number', type=int)    # 필요한 인수를 추가
args = parser.parse_args()

path_dir = '/home/percv-d0/dyn/2023.1_BSpaper/Korean-license-plate-Generator'
# change the directorys below to change the directories for saving 
kset_dir = '/home/percv-d0/dyn/2023.1_BSpaper/datasets/klicense_dataset/images/train'
anno_dir = '/home/percv-d0/dyn/2023.1_BSpaper/datasets/klicense_dataset/labels/train'

carim_num = args.carimg
lpimg_num = args.carimg

 # all directories of car images in list
# car_img_dir = os.path.join(path_dir, "dataset/images")
car_img_dir = '/home/percv-d0/dyn/2023.1_BSpaper/datasets/kaggle_dataset/images/'
# pick random 20 car images from all car

car_list = random.choices(os.listdir(car_img_dir), k = carim_num)
# print(car_list)
# print(car_list[0].split('.')[0])
car_list_name= [0 for i in range(carim_num)]
for i in range(carim_num):
    car_list_name[i] = car_list[i].split('.')[0]

# all directories of bounding box xml files in list
# org_lp_bbox = os.path.join(path_dir, "dataset/annotations")
org_lp_bbox = '/home/percv-d0/dyn/2023.1_BSpaper/datasets/kaggle_dataset/annotations'
# bbox_list = sorted(os.listdir(org_lp_bbox))
# print(bbox_list[0:9])
bbox_list = [(car_list_name[i] + ".xml") for i in range(carim_num)]
# print(bbox_list)

# # Passing the path of the xml document to enable the parsing process
# tree = ET.parse(os.path.join(org_lp_bbox,bbox_list[0]))
# # getting the parent tag of the xml document
# root = tree.getroot()
# # printing the root (parent) tag of the xml document, along with its memory location
# print(root)
# for child in root:
#     print(child.tag, child.attrib)

# # print(int(root[2][0].text)) # width
# # print(int(root[2][1].text))  #height

# # print(int(root[4][5][0].text)) # xmin
# # print(int(root[4][5][1].text)) # ymin
# # print(int(root[4][5][2].text)) # xmax
# # print(int(root[4][5][3].text)) # ymax
# xmin = int(root[4][5][0].text)
# xmax = int(root[4][5][2].text)
# ymin = int(root[4][5][1].text)
# ymax = int(root[4][5][3].text)
# print(xmin, xmax, ymin, ymax)
# xcenter = (xmin + xmax)//2
# ycenter = (ymin + ymax)//2
# print(xcenter, ycenter)

# license plate image path dir
lp_img_dir = os.path.join(path_dir,"DB")

lp_list_name= [0 for i in range(lpimg_num)]
lp_list = random.choices(os.listdir(lp_img_dir), k = lpimg_num)
for i in range(lpimg_num):
    lp_list_name[i] = lp_list[i].split('.')[0]
# src1 = cv2.imread(os.path.join(car_img_dir, car_list[0]))

for i in range(carim_num):
    src1 = cv2.imread(os.path.join(car_img_dir, car_list[i])) # car image 파일 읽기
    
    # Passing the path of the xml document to enable the parsing process
    tree = ET.parse(os.path.join(org_lp_bbox,bbox_list[i]))
    # getting the parent tag of the xml document
    root = tree.getroot()
    # printing the root (parent) tag of the xml document, along with its memory location
    # print(root)
    # for child in root: print(child.tag, child.attrib)

    # number of licenses in image
    lp_num = len(tree.findall("object"))
    # print("# lp = ", lp_num)
    
        
    for k in range(lpimg_num):
        src1_1 = src1.copy()
        src2 = cv2.imread(os.path.join(lp_img_dir, lp_list[k])) # korean lp 읽기
        # src2_copy = src2.copy()
        
        for j in range(lp_num):
            lp_h, lp_w, lp_c = src2.shape    

            xmin = int(root[4+j][5][0].text)
            xmax = int(root[4+j][5][2].text)
            ymin = int(root[4+j][5][1].text)
            ymax = int(root[4+j][5][3].text)
            # print(xmin, xmax, ymin, ymax)
            xcenter = (xmin + xmax)//2
            ycenter = (ymin + ymax)//2
            # print(xcenter, ycenter)
            ch, cw, cc = src1.shape 
            
            # cv2.imshow('dst2',src2)
            # print("lp", lp_h, lp_w,"car", ch, cw, "\n")
            if (lp_w/lp_h > (xmax-xmin)/(ymax-ymin)): # if lp width is larger than original box
                dst1 = cv2.resize(src2, (0,0), fx = (ymax-ymin)/lp_h, fy = (ymax-ymin)/lp_h)
            else:  # the other way around
                dst1 = cv2.resize(src2, (0,0), fx = (xmax-xmin)/lp_w, fy = (xmax-xmin)/lp_w)
            
            lp_h, lp_w, lp_c = dst1.shape
            # print(j, lp_h, lp_w)
            src1_1[round(ycenter-lp_h/2):round(ycenter-lp_h/2)+lp_h,round(xcenter-lp_w/2):round(xcenter-lp_w/2)+lp_w] = dst1

            if (j != 0):
                with open(anno_dir+"/"+car_list_name[i]+"_"+lp_list_name[k]+".txt", 'a') as f:
                    f.write("0 %.3f %.3f %.3f %.3f" % (xcenter/cw, ycenter/ch, lp_w/cw, lp_h/ch) + "\n")
                    f.close()
            else:
                with open(anno_dir+"/"+car_list_name[i]+"_"+lp_list_name[k]+".txt", 'w') as f:
                    f.write("0 %.3f %.3f %.3f %.3f" % (xcenter/cw, ycenter/ch, lp_w/cw, lp_h/ch) + "\n")
                    f.close()

        cv2.imwrite(kset_dir+"/"+car_list_name[i]+"_"+lp_list_name[k]+".png", src1_1)
        

# src1[xmin:xmax, ymin:ymax] = src2
# cv2.imshow('dst',src1)

# cv2.waitKeyEx()
# cv2.destroyAllWindows()