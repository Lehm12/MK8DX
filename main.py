import cv2
import sys
from PIL import Image, ImageFilter, ImageDraw

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
im2 = im.crop((150, 130, 600, 633))
temp=42
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
time = 1
for i in range(12):
 im2 = im.crop(crop[i])
 #im2 = im2.convert("L")
 #im2.resize((im2.weight*time, im2.height*time), Image.LANCZOS)
 im2 = im2.resize((im2.width*time, im2.height*time))
 save_name = './Result/test1_rank' + str(i+1) + '.jpeg'
 im2.save(save_name)

 #im2.show()
 txt = tool.image_to_string(
  #Image.open(filename),
  im2,
  lang="jpn",
  builder=pyocr.builders.TextBuilder(tesseract_layout=6)
 )

# txt is a Python string
 print((i+1), txt)

 #文字だと認識した範囲を出力する
 res = tool.image_to_string(
  im2,
  lang="jpn",
  builder=pyocr.builders.WordBoxBuilder(tesseract_layout=6)
 )

 for d in res:
     #print(d.content)
     #print(d.position)
     out = ImageDraw.Draw(im2)
     out.rectangle([d.position[0], d.position[1]], outline='lime', width=2)

 #im2.show()
 im2.save(save_name)
