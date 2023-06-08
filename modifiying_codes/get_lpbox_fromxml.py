import os
# from bs4 import BeautifulSoup as bsoup # to read xml files
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString
import xml.dom
import random

path_dir = '/home/percv-d0/dyn/2023.1_BSpaper/Korean-license-plate-Generator'
kset_dir = '/home/percv-d0/dyn/2023.1_BSpaper/datasets/klicense_dataset/images'
anno_dir = '/home/percv-d0/dyn/2023.1_BSpaper/datasets/klicense_dataset/labels'

 # all directories of car images in list
# car_img_dir = os.path.join(path_dir, "dataset/images")
car_img_dir = '/home/percv-d0/dyn/2023.1_BSpaper/datasets/kaggle_dataset/images/'
# pick random 20 car images from all car
carim_num = 1
# car_list = random.choices(os.listdir(car_img_dir), k = carim_num)
car_list = ['Cars87.png']
# print(car_list)
# print(car_list[0].split('.')[0])
car_list_name= [0 for i in range(carim_num)]
for i in range(carim_num):
    car_list_name[i] = car_list[i].split('.')[0]

# all directories of bounding box xml files in list
# org_lp_bbox = os.path.join(path_dir, "dataset/annotations")
org_lp_bbox = '/home/percv-d0/dyn/2023.1_BSpaper/datasets/kaggle_dataset/annotations/Cars89.xml'
# bbox_list = sorted(os.listdir(org_lp_bbox))
# print(bbox_list[0:9])
bbox_list = [(car_list_name[i] + ".xml") for i in range(carim_num)]
# print(bbox_list)

# Passing the path of the xml document to enable the parsing process
tree = ET.parse(org_lp_bbox)
# tree = ET.parse(os.path.join(org_lp_bbox,bbox_list[0]))

# xmlfile = open(os.path.join(org_lp_bbox,bbox_list[0]), 'r')
# xmlfile = xmlfile.read()
# objects = parseString(xmlfile)

# getting the parent tag of the xml document
root = tree.getroot()
# printing the root (parent) tag of the xml document, along with its memory location
print(root)
for child in root:
    print(child.tag, child.attrib)

hi = tree.findall('object')
print(hi)


print(int(root[2][0].text)) # width
print(int(root[2][1].text))  #height

print(int(root[4][5][0].text)) # xmin
print(int(root[4][5][1].text)) # ymin
print(int(root[4][5][2].text)) # xmax
print(int(root[4][5][3].text)) # ymax
print("\n", int(root[5][5][0].text)) # xmin
print(int(root[5][5][1].text)) # ymin
print(int(root[5][5][2].text)) # xmax
print(int(root[5][5][3].text)) # ymax
xmin = int(root[4][5][0].text)
xmax = int(root[4][5][2].text)
ymin = int(root[4][5][1].text)
ymax = int(root[4][5][3].text)
print(xmin, xmax, ymin, ymax)
xcenter = (xmin + xmax)//2
ycenter = (ymin + ymax)//2
print(xcenter, ycenter)
