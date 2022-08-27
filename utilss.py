
import os
import shutil
import ssl

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from shapely.geometry import Polygon 

from PIL import Image

def draw_conts(im,conts,fcolor=None):
  opacity = .4
  im =im.copy()
  fcolor =fcolor if fcolor else get_random_color()
  frontmm = np.full_like(im,fcolor,np.uint8)
  mask = np.zeros_like(im)
  for cont in conts:
    cv2.drawContours(mask, [cont], -1, color=[int(opacity*255)]*3, thickness=cv2.FILLED)
  mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
  rgba = np.dstack((frontmm, mask))
  im = overlay_image_alpha(im,rgba)
  # drawline
  cv2.drawContours(im, conts, -1, color=fcolor, thickness=2)
  return im
def get_random_color():
  return [255,0,0]
  # return [0,0,255]

def resize_with_pad(im, target_width, target_height, val = 255):
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    im = im.copy()
    tonp = False
    shp ='rgb'
    if isinstance(im,np.ndarray):
      shp = 'gray' if len(im.shape) ==2 else 'rgb'
      im = Image.fromarray(im)
      tonp = True
    target_ratio = target_height / target_width
    im_ratio = im.height / im.width
    if target_ratio > im_ratio:
        # It must be fixed by width
        resize_width = target_width
        resize_height = round(resize_width * im_ratio)
    else:
        # Fixed by height
        resize_height = target_height
        resize_width = round(resize_height / im_ratio)

    image_resize = im.resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new('RGBA', (target_width, target_height), (val,val,val,255))
    offset = (round((target_width - resize_width) / 2), round((target_height - resize_height) / 2))
    background.paste(image_resize, offset)
    if tonp:
      res=np.array(background,dtype=np.uint8)
      if shp =='gray':
        res = cv2.cvtColor(res,cv2.COLOR_BGRA2GRAY)
      else:
        res = cv2.cvtColor(res,cv2.COLOR_BGRA2BGR)
      return res 
    return background  #.convert('L')

def poly2cont(poly):
  return np.array(list(poly.exterior.coords)).astype(np.int32).reshape(-1,1,2)

def contour_to_polygon(cont):
  return Polygon(cont.reshape(-1,2).tolist())

# def draw_conts(im,conts):
#   opacity = .3
#   im =im.copy()
#   for cont in conts:
#     #select random dark color
#     fcolor = get_random_color()
#     #create trans colored rgba
#     mask = np.zeros_like(im)
#     cv2.drawContours(mask, [cont], -1, color=[int(opacity*255)]*3, thickness=cv2.FILLED)
#     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#     frontmm = np.full_like(im,fcolor,np.uint8)
#     rgba = np.dstack((frontmm, mask))
#     im = overlay_image_alpha(im,rgba)
#     # drawline
#     cv2.drawContours(im, [cont], -1, color=fcolor, thickness=2)
#   return im

# def get_random_color():
#   return [255,0,0]

def overlay_image_alpha(img, img_overlay_rgba, x=0, y=0):
  """Overlay `img_overlay` onto `img` at (x, y) and blend using `alpha_mask`.

  `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
  """
  
  alpha_mask = img_overlay_rgba[:, :, 3] / 255.0
  img_overlay = img_overlay_rgba[:, :, :3]
  # Image ranges
  img=img.copy()
  y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
  x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

  # Overlay ranges
  y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
  x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

  # Exit if nothing to do
  if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
      return None

  # Blend overlay within the determined ranges
  img_crop = img[y1:y2, x1:x2]
  img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
  alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
  alpha_inv = 1.0 - alpha

  img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop
  return img