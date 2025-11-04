#本程式碼將label從xml格式轉成txt格式
import copy
from lxml.etree import Element,SubElement,tostring,ElementTree
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir,getcwd
from os.path import join
#辨識的類別
classes=['with_helmet']
def convert(size,box):
    dw=1./size[0]
    dh=1./size[1]
    x=(box[0]+box[1])/2.0
    y=(box[2]+box[3])/2.0
    w=box[1]-box[0]
    h=box[3]-box[2]
    x=x*dw
    w=w*dw
    y=y*dh
    h=h*dh
    return (x,y,w,h)

def convert_annotation(image_id):
    #轉換前的標註檔案位置
    in_file=open('/mnt/c/Users/joshu/文件/career/資工所/交大/課程/嵌入式系統2025Fall/Lab3/3-2/helmet_dataset/labels/%s.xml'%(image_id),encoding='UTF-8')
    #轉換後的標註檔案位置
    out_file=open('/mnt/c/Users/joshu/文件/career/資工所/交大/課程/嵌入式系統2025Fall/Lab3/3-2/helmet_dataset/txt/%s.txt'%(image_id),'w')
    tree=ET.parse(in_file)
    root=tree.getroot()
    size=root.find('size')
    w=int(size.find('width').text)
    h=int(size.find('height').text)
    for obj in root.iter('object'):
        cls=obj.find('name').text
        if cls not in classes:
            continue
        cls_id=classes.index(cls)
        xmlbox=obj.find('bndbox')
        b=(float(xmlbox.find('xmin').text),float(xmlbox.find('xmax').text),float(xmlbox.find('ymin').text),float(xmlbox.find('ymax').text))
        bb=convert((w,h),b)
        out_file.write(str(cls_id)+" "+" ".join([str(a) for a in bb])+'\n')
#轉換前的標註檔案位置，這邊是要將要轉換的檔案名稱列出來
xml_path=os.path.join('/mnt/c/Users/joshu/文件/career/資工所/交大/課程/嵌入式系統2025Fall/Lab3/3-2/helmet_dataset/txt/')
img_xmls=os.listdir(xml_path)
for img_xml in img_xmls:
    label_name=img_xml.split('.')[0]
    #print(label_name)
    convert_annotation(label_name)