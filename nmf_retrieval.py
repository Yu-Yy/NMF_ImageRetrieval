import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from dataset import mydataset_NMF
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import sklearn.decomposition as dc

def get_distance(features):
    samples_num = features.shape[0]
    process_feat = features[:,None,:].repeat(samples_num,axis=1)
    Distance = process_feat - features
    Norm_dist = np.linalg.norm(Distance,axis=-1)
    F_square = Norm_dist ** 2.0
    return F_square

def get_distance_single(test_image, features):
    '''
    test_image is in shape (1,F)
    features is in shape (N,F)
    '''
    distance = test_image - features
    norm_dist = np.linalg.norm(distance,axis=-1,keepdims=True)
    norm_dist = norm_dist.reshape(1,-1)
    s = norm_dist ** 2.0
    return s


def main():
    # initialed the dataset 
    image_folder = 'F:\\课件\\2020-2021-2\\模式识别\\project\\shopee-product-matching\\train_images\\'
    image_shape = (256,256)
    train_dataset = mydataset_NMF(image_dir=image_folder,is_train=True)
    test_dataset = mydataset_NMF(image_dir=image_folder, is_train=False)
    # process the train dataset
    train_length = train_dataset.__len__()
    test_length = test_dataset.__len__()
    # dataset index and process 
    LBP_collect = []
    HOG_collect = []
    gist_collect = []
    train_label = []
    for idx in tqdm(range(train_length)):
        LBP_feature, HOG_feature, gist_feature, label= train_dataset.__getitem__(idx)
        LBP_collect.append(LBP_feature)
        HOG_collect.append(HOG_feature)
        gist_collect.append(gist_feature)
        train_label.append(label)
    LBP_collect = np.concatenate(LBP_collect,axis=0)
    HOG_collect = np.concatenate(HOG_collect,axis=0)
    gist_collect = np.concatenate(gist_collect,axis=0)
    train_label = np.asarray(train_label)
    # create the F norm square distance matrix
    LBP_matrix = get_distance(LBP_collect)
    HOG_matrix = get_distance(HOG_collect)
    gist_matrix = get_distance(gist_collect)
    lbp_median = np.median(LBP_matrix)
    hog_median = np.median(HOG_matrix)
    gist_median = np.median(gist_matrix)
    s_lbp = np.exp(-LBP_matrix/(2*lbp_median))
    s_hog = np.exp(-HOG_matrix/(2*hog_median))
    s_gist = np.exp(-gist_matrix/(2*gist_median))
    S = (s_lbp + s_hog + s_gist)/3
    # S is the pre decomposition matrix
    nmf_dc = dc.NMF(n_components=100,init='nndsvda', tol=5e-3, max_iter=1000)
    trainned_base = nmf_dc.fit_transform(S)
    acc5 = 0
    acc1 = 0
    for idx in tqdm(range(test_length)):
        LBP_feature, HOG_feature, gist_feature, label= test_dataset.__getitem__(idx)
        s_lbp = np.exp(-get_distance_single(LBP_feature,LBP_collect)/(2*lbp_median))
        s_hog = np.exp(-get_distance_single(HOG_feature,HOG_collect)/(2*hog_median))
        s_gist = np.exp(-get_distance_single(gist_feature,gist_collect)/(2*gist_median))
        s_test = (s_lbp+s_hog+s_gist)/3
        s_low_dim = nmf_dc.transform(s_test)
        distance_s = np.sum((s_low_dim - trainned_base) ** 2, axis=-1)
        idx_sort = np.argsort(distance_s)
        idx_top5 = idx_sort[:5]
        pred_label = train_label[idx_top5]
        if label in pred_label: #TODO: text need to be further index
            acc5 = acc5 + 1
        if label == pred_label[0]:
            acc1 = acc1 + 1

    acc_rate5 = acc5 / test_length
    acc_rate1 = acc1 / test_length
    print('----------------------------')
    # print(f"err = {err_rate:.4f}")
    print(f"acc1 = {acc_rate1:.4f}")
    print(f"acc5 = {acc_rate5:.4f}")


if __name__ == '__main__':
    main()