import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
from urllib.request import urlretrieve ,urlopen
import os
import torch
from distutils.dir_util import copy_tree
import tempfile
import zipfile
import subprocess
import mmcv
from mmcv import Config
import sys
from shapely.geometry import Polygon 


import shutil
import ssl

import urllib.request
from pathlib import Path

import cv2
from PIL import Image
import utilss
if 'model_inference' not in sys.modules:
  from mmocr.apis import init_detector ,model_inference
else:
  model_inference = sys.modules['model_inference']
  init_detector = sys.modules['init_detector']
# if 'init_detector' not in sys.modules:
# import mmocr
# from mmocr.apis import init_detector ,model_inference
  

########## utils

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized
def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return local_filename
def load_and_preprocess_img(img_path, bbox=None):
  img = Image.open(img_path).convert('RGB')
  img = np.array(img).astype(np.uint8)
  return img


# @st.cache()
def predictim(im , model):
  if 'model_inference' not in sys.modules:
    # if not model_inference:
    from mmocr.apis import model_inference
  result = model_inference(model,im )
  # result[]
  return result
  # return result
# minfer=None
# minit=None
@st.cache(allow_output_mutation=True)
def loading_model():
  # global minfer
  # global minit
  try:
    os.system('rm epoch_200.pth')
  except:
    pass

  # if 'sn1.mar' not in os.listdir('./'):
  #   url = 'https://www.dropbox.com/s/raw/hhv2o15qnyzohx8/sn1.mar'
  #   urlretrieve(url, 'sn1.mar' , )
  
  # if 'epoch_100.pth' not in os.listdir('./'):
  #   url = 'https://www.dropbox.com/s/raw/vbxve7cn0q9ir78/epoch_100.pth'
  #   urlretrieve(url, 'epoch_100.pth' , )
  
  # if 'epoch_200.pth' not in os.listdir('./'):
  #   url = 'https://www.dropbox.com/s/raw/lkk1avkvlk8ykzw/epoch_200.pth'
  #   urlretrieve(url, 'epoch_200.pth' , )
  if '200s.pth' not in os.listdir('./'):
    url = 'https://www.dropbox.com/s/raw/5qjrtvqsfmbzgfk/200s.pth'
    urlretrieve(url, '200s.pth' , )

  # cfg = Config.fromfile('./cfgsn.py')
  # checkpoint = "./epoch_100.pth"
  # if 'init_detector' not in sys.modules:
  #   # if not init_detector:
  #   from mmocr.apis import init_detector ,model_inference
  # model1 = init_detector(cfg, checkpoint, device="cpu")
  # if model1.cfg.data.test['type'] == 'ConcatDataset':
  #   model1.cfg.data.test.pipeline = model1.cfg.data.test['datasets'][0].pipeline
  # return model1
  from mmdet.apis import set_random_seed
  set_random_seed(0, deterministic=False)
  cfg = Config.fromfile('./cfgsn.py')
  checkpoint = "./200s.pth"
  # if 'init_detector' not in sys.modules:
  #   from mmocr.apis import init_detector ,model_inference
    # minfer = model_inference
    # minit = init_detector
  model1 = init_detector(cfg, checkpoint, device="cpu")
  if model1.cfg.data.test['type'] == 'ConcatDataset':
    model1.cfg.data.test.pipeline = model1.cfg.data.test['datasets'][0].pipeline
  return model1
model = loading_model()



# st.write('testcv5')
# if st.button('tt'):
#   xx = predictim('./s1.jpg',model)
#   st.write(xx)
def main():
  # st.sidebar.info('PRIVACY POLICY: uploaded images are never saved or stored. and discarded from RAM instantly.')
  st.sidebar.info('Click predict after selection of an image')
  form0 =st.sidebar.form("my_form0")

  f = form0.file_uploader("Please Select to Upload an Image", type=['png', 'jpg', 'jpeg', 'tiff', 'gif'])
  btnpredicrupload = form0.form_submit_button("predict",)


  st.sidebar.write(" ------ ")
  form =st.sidebar.form("my_form")
  photos = ['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg',]
  option = form.selectbox('or select a sample image', photos)
  submitted = form.form_submit_button("predict",)

  st.sidebar.write(" ------ ")
  fcolor = st.sidebar.selectbox('color', ['red','green','blue'])
  do_thresh_box = st.sidebar.checkbox('adaptive threshold')

  if btnpredicrupload and f is not None:
    tfile = tempfile.NamedTemporaryFile(delete=True)
    tfile.write(f.read())
    run_app(tfile.name ,do_thresh_box,fcolor)
  if submitted:
    st.empty()
    directory ='./imgs/'
    pic = os.path.join(directory, option)
    run_app(directory+option , do_thresh_box,fcolor)
  # inp = st.text_input('t2','')
  # if st.button('run'):
  #   subprocess.run(inp.split(' '))


def run_app(imgpath,do_thresh,fcolor):
  d0 =dict(red=[255,0,0] ,green=[0,255,0] ,blue=[0,0,255]  )
  fcolor = d0[fcolor]
  # st.sidebar.write(imgpath)
  # r = np.random.randint(1e3,1e7)
  tfile = tempfile.NamedTemporaryFile(delete=True,suffix='.jpg')
  dst = tfile.name
  # tfile = open()
  img = load_and_preprocess_img(imgpath)
  # img = utilss.resize_with_pad(img,800,800)
  if do_thresh:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshim = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 51, 20)
    img= cv2.cvtColor(threshim, cv2.COLOR_GRAY2RGB)
  
  cv2.imwrite(dst,cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  # imgo = image_resize(img , width = 640)
  st.write('input:')
  st.image(img, caption = "Selected Input" , width =640)

  # if st.button('predict2'):
  
  # try:
  # except: print('errr--------------')
  ''
  ##################
  preds,img_metas,downsample_ratio =  model_inference(model,dst )
  # with open('./0.json','r') as f:
  #   res = json.load(f)
  ################
  # conts = [ np.int0(c[:-1]).reshape(-1,1,2) for c in  res['boundary_result']]
  # polys = [utilss.contour_to_polygon(c).simplify(2) for c in conts]
  # conts =[utilss.poly2cont(p) for p in polys]
  # mm = utilss.draw_conts(img,conts,fcolor)
  ## mm=image_resize(mm, width = 640)
  mm=preds[1]
  mm=cv2.cvtColor(mm, cv2.COLOR_GRAY2RGB)
  st.image(mm,caption='output image' , width = 640)
  del polys
  del res
  del mm 
  del conts
  del img
  del img_metas
  # os.remove(dst)
  # x = predictim(img,model)
  # st.write(x)

main()










  