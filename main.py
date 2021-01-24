import cv2
import sys
from PIL import Image, ImageFilter, ImageDraw
import pytesseract
from tesserocr import PyTessBaseAPI, PSM, RIL

import pyocr
import pyocr.builders

import numpy as np

filename = './Result/test1.jpeg'

# 画像ファイルパスから読み込み
im = Image.open(filename)

#順位付きクロップ
#im2 = im.crop((50, 130, 600, 633))

#名前からクロップ(順位ごとにクロップする)
#リザルト画像の順位が出るところが固定なので定数で問題ない
#値は経験的に決めた
crop = []
#im = im.crop((150, 130, 600, 633))

temp=42
crop_name = [(1,4,350,37)]
crop_item = [(350,4,440,37)]
for i in range(12):
 crop.append((150, 130+temp*i, 600, 130+temp*(i+1)))

tools = pyocr.get_available_tools()
if len(tools) == 0:
 print("No OCR tool found")
 sys.exit(1)
# The tools are returned in the recommended order of usage
tool = tools[0]
print("Will use tool '%s'" % (tool.get_name()))
# Ex: Will use tool 'libtesseract'

langs = tool.get_available_languages()
print("Available languages: %s" % ", ".join(langs))
#lang = langs[0]
#print("Will use lang '%s'" % (lang))
# Ex: Will use lang 'fra'
# Note that languages are NOT sorted in any way. Please refer
# to the system locale settings for the default language
# to use.

#timeは入力画像のサイズを何倍にするかの引数
time = 3
b = 8
for i in range(12):
 save_name = './Result/test1_rank' + str(i+1) + '_name.jpeg'
 save_item = './Result/test1_rank' + str(i+1) + '_item.jpeg'


 im2 = im.crop(crop[i])

 #im2 = im2.filter(ImageFilter.FIND_EDGES)
 im2 = im2.convert('L')
 im2 = im2.filter(ImageFilter.MedianFilter())

 im2 = im2.filter(ImageFilter.FIND_EDGES)

 #im2 = im2.filter(ImageFilter.MedianFilter())

 border = 0

 for x in range(im2.size[0]):
  for y in range(im2.size[1]):
   border += im2.getpixel((x, y))

 border = border / (im2.size[0]*im2.size[1])
 #print(border)
 border = 5

 for x in range(im2.size[0]):
  for y in range(im2.size[1]):
   item = im2.getpixel((x, y))
   if item > border:
    item = 255
   else:
    item = 0
   im2.putpixel((x, y), item)
 im2 = im2.filter(ImageFilter.MedianFilter())
 im2 = im2.filter(ImageFilter.MedianFilter())
 im2 = im2.filter(ImageFilter.MedianFilter())


 im_name = im2.crop(crop_name[0])
 im_item = im2.crop(crop_item[0])
 #if i == 6:
  #im_name.show()
  #im_item.show()
 """
 im_name = im_name.resize((im_name.width*time, im_name.height*time))
 im_item = im_item.resize((im_item.width*time, im_item.height*time))

 api_name = PyTessBaseAPI(psm=PSM.AUTO, lang='jpn')
 api_name.SetImage(im_name)

 api_item = PyTessBaseAPI(psm=PSM.AUTO, lang='jpn')
 api_item.SetImage(im_item)
 a = ':'
 print((i+1), api_name.GetUTF8Text(), a, api_item.GetUTF8Text())
 """

 ######################################################

 txt_name = tool.image_to_string(
  #Image.open(filename),
  im_name,
  lang="jpn",
  builder=pyocr.builders.TextBuilder(tesseract_layout=b)
 )

 txt_item = tool.image_to_string(
  #Image.open(filename),
  im_item,
  lang="eng",
  builder=pyocr.builders.TextBuilder(tesseract_layout=b)
 )

 #txt_item = pytesseract.image_to_string(im_item)

 # txt is a Python string
 a = ':'
 print((i+1), txt_name, a, txt_item)

 #文字だと認識した範囲を出力する
 res = tool.image_to_string(
  im_name,
  lang="jpn",
  builder=pyocr.builders.WordBoxBuilder(tesseract_layout=b)
 )

 for d in res:
     #print(d.content)
     #print(d.position)
     out = ImageDraw.Draw(im_name)
     out.rectangle([d.position[0], d.position[1]], outline='lime', width=2)

 #文字だと認識した範囲を出力する
 res = tool.image_to_string(
  im_item,
  lang="eng",
  builder=pyocr.builders.WordBoxBuilder(tesseract_layout=b)
 )

 for d in res:
     #print(d.content)
     #print(d.position)
     out = ImageDraw.Draw(im_item)
     out.rectangle([d.position[0], d.position[1]], outline='lime', width=2)

 #im2.show()
 #print("----------------")
 im_name.save(save_name)
 im_item.save(save_item)


