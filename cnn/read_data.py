import cv2
import numpy as np
import pandas as pd
import glob
import random
import sys

def read_imgs(img_size):

  path = '/home/kb/tf/cnn/data/train/'

  nimgs = 25000

  x_imgs = np.zeros((nimgs,img_size,img_size,3),dtype=np.float)
  y_imgs = np.zeros((nimgs,2),dtype=np.float)

  k = 0
  for img_path in glob.glob(path + "*.jpg"):
    #print(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size,img_size), interpolation = cv2.INTER_CUBIC)
    img = img/255.0 
    x_imgs[k,:,:,:] = img

    if ('cat' in img_path):    
      y_imgs[k,:] = np.array([1.0, 0.0])
    elif ('dog' in img_path):
      y_imgs[k,:] = np.array([0.0, 1.0])
    else:
      print("error ")
      sys.exit()
    k += 1

  print("k=",k)

  
  for i in range(int(nimgs/2)):
   j = np.random.randint(nimgs)
   k = np.random.randint(nimgs)
   # swap j and k
   temp = x_imgs[j,:,:,:]
   x_imgs[j,:,:,:] = x_imgs[k,:,:,:]
   x_imgs[k,:,:,:] = temp
   temp = y_imgs[j,:]
   y_imgs[j,:] = y_imgs[k,:]
   y_imgs[k,:] = temp


  ntrain = int(0.9*nimgs)
  nval = nimgs - ntrain

  x_train = x_imgs[0:ntrain,:,:,:]
  y_train = y_imgs[0:ntrain,:]
  x_val = x_imgs[ntrain:,:,:,:]
  y_val = y_imgs[ntrain:,:]

  return ntrain, nval, x_train, y_train, x_val, y_val
