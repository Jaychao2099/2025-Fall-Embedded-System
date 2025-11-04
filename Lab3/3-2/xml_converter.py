#本程式碼將label從xml格式轉成txt格式
import copy
from lxml.etree import Element,SubElement,tostring,ElementTree
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir
from os.path import join
#辨識的類別
classes=['helmet']

# 改為接受 (xmin, ymin, xmax, ymax)
def convert(size,box):
	# size: (width, height)
	dw=1.0/size[0]
	dh=1.0/size[1]
	x=(box[0]+box[2])/2.0
	y=(box[1]+box[3])/2.0
	w=box[2]-box[0]
	h=box[3]-box[1]
	x=x*dw
	w=w*dw
	y=y*dh
	h=h*dh
	return (x,y,w,h)

def convert_annotation(image_id):
	# 以 labels 資料夾為來源
	in_path = '/mnt/c/Users/joshu/文件/career/資工所/交大/課程/嵌入式系統2025Fall/Lab3/3-2/safety_helmet_dataset/labels'
	out_path = '/mnt/c/Users/joshu/文件/career/資工所/交大/課程/嵌入式系統2025Fall/Lab3/3-2/safety_helmet_dataset/txt'
	os.makedirs(out_path, exist_ok=True)

	in_file_path = os.path.join(in_path, f'{image_id}.xml')
	out_file_path = os.path.join(out_path, f'{image_id}.txt')

	try:
		tree = ET.parse(in_file_path)
	except Exception as e:
		print(f"Cannot open {in_file_path}: {e}")
		return

	root = tree.getroot()
	size = root.find('size')
	if size is None:
		print(f"No size info in {in_file_path}")
		return
	w = int(size.find('width').text)
	h = int(size.find('height').text)

	# 使用 with 確保關閉檔案，並格式化為 6 位小數
	with open(out_file_path, 'w', encoding='utf-8') as out_file:
		for obj in root.iter('object'):
			cls = obj.find('name').text.strip()
			if cls not in classes:
				continue
			cls_id = classes.index(cls)
			xmlbox = obj.find('bndbox')
			# 取得標準順序 xmin, ymin, xmax, ymax
			xmin = float(xmlbox.find('xmin').text)
			ymin = float(xmlbox.find('ymin').text)
			xmax = float(xmlbox.find('xmax').text)
			ymax = float(xmlbox.find('ymax').text)
			b = (xmin, ymin, xmax, ymax)
			bb = convert((w,h), b)
			out_file.write(str(cls_id) + " " + " ".join([f"{a:.6f}" for a in bb]) + '\n')

# 轉換前的標註檔案位置（修正為 labels 目錄）
xml_path = '/mnt/c/Users/joshu/文件/career/資工所/交大/課程/嵌入式系統2025Fall/Lab3/3-2/safety_helmet_dataset/labels'
if not os.path.exists(xml_path):
	print("Cannot find the xml path")
	exit(0)

img_xmls = os.listdir(xml_path)
for img_xml in img_xmls:
	# 只處理 .xml 檔
	if not img_xml.lower().endswith('.xml'):
		continue
	label_name = img_xml.rsplit('.',1)[0]
	convert_annotation(label_name)
