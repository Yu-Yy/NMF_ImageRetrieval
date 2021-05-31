import csv
import random
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os,sys
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import cv2
from img_gist_feature.utils_gist import *
import skimage
from skimage import feature as ft
import scipy

TEST = 'datasets/test_divide_mini50_eq.pkl' 
TRAIN = 'datasets/train_divide_mini50_eq.pkl'


class mydataset_NMF:
    def __init__(self, image_dir,resize_height=256, resize_width=256,is_train = True, is_augmentation = False):
        '''
        Input:  image_dir -图片路径(image_dir+imge_name.jpg构成图片的完整路径)
                text_path - 文本数据的路径
                resize_height -图像高，
                resize_width  -图像宽    
        '''

        self.new_size = (resize_height,resize_width)
        self.image_dir = image_dir

        if is_train:
            self.dataset = TRAIN
        else:
            self.dataset = TEST
        with open(self.dataset,'rb') as dfile:
            raw_data = pickle.load(dfile)
        self.imgs = list()
        self.texts = list()

        for l,v in raw_data.items():
            self.imgs.extend(v)
            self.texts.extend([l for _ in range(len(v))])
       
 
    def __getitem__(self, i):
        #获取图像
        img_path = os.path.join(self.image_dir,self.imgs[i][1])
        pil_img = Image.open(img_path)
        # pil_img = pil_img.convert('L') 
        pil_img = pil_img.resize(self.new_size)

        pil_img = pil_img.convert('L') 
        pil_img = np.asarray(pil_img,dtype=np.float32)/255
        # HOG, GIST, LBP
        LBP_feature = ft.local_binary_pattern(pil_img, 510, 3.0, method='uniform') #method='var'
        LBP_feature_num = 512
        comparision_list = list(range(LBP_feature_num))
        # 0 in the non_value
        # histogram=scipy.stats.itemfreq(LBP_feature)
        histogram = np.asarray([np.sum(LBP_feature==x) for x in comparision_list], dtype=np.float32)
        LBP_feature = histogram.reshape(1,-1)
        # LBP_feature = LBP_feature/np.max(LBP_feature)
        # LBP_feature = np.nan_to_num(LBP_feature)
        # LBP_feature = LBP_feature.reshape(1,-1)

        HOG_feature = ft.hog(pil_img, pixels_per_cell=(42,42),cells_per_block=(4,4),orientations=8,feature_vector=True,block_norm='L2')
        HOG_feature = HOG_feature.reshape(1,-1)

        gist_helper = GistUtils()
        gist_feature = gist_helper.get_gist_vec(pil_img, mode="gray")
        gist_feature = gist_feature.reshape(1,-1)

        text_data = self.texts[i]
        #返回数据
        return LBP_feature, HOG_feature, gist_feature, text_data
 
    def __len__(self):
        return len(self.imgs)   