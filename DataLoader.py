import csv
import os
import numpy as np
import math
import torch
import torch.utils.data as data
from PIL import Image
import sys
from PIL import ImageChops
import random
import scipy.io
import torchvision.transforms as transforms
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def getData_DR(fn, srcDir, train):
    lblarr = []
    imgFnarr = []
    file = open(fn, 'r')
    for line in file:
        line = line.split(',')
        if line[0] == 'image': continue
        lbl = int(line[1])
        if len(line) > 2:
            tmp = line[2].strip()
        if (train == 'TRAIN') or ((train == 'VAL') and (tmp == 'Public')) or ((train == 'TEST') and (tmp == 'Private')):
            lblarr.append(lbl)
            imgFnarr.append(os.path.join(srcDir, line[0] + '.png'))
    file.close()
    return imgFnarr, lblarr

def getData_LIMUC(srcDir):
    lblarr = []
    imgFnarr = []
    classes = ['Mayo 0', 'Mayo 1', 'Mayo 2', 'Mayo 3']

    for i, class_name in enumerate(classes):
        subdir = os.path.join(srcDir, class_name)
        fnArr = os.listdir(subdir)
        for fn in fnArr:
            imgFnarr.append(os.path.join(subdir, fn))
            lblarr.append(i)
    imgFnarr = np.array(imgFnarr)
    lblarr = np.array(lblarr)
    return imgFnarr, lblarr

def splitTrainValidation(fnArr, lblArr, val_ratio=0.1, seed=0):
    train_fn, val_fn, train_lbl, val_lbl = train_test_split(
        fnArr,
        lblArr,
        test_size=val_ratio,
        random_state=seed,
        stratify=lblArr
    )
    return (np.array(train_fn),
            np.array(train_lbl),
            np.array(val_fn),
            np.array(val_lbl))

def splitData_Pecentage(lblArr, pL, seed):
    uniqueLbls = np.unique(lblArr)
    trainL_idx = []
    trainUL_idx = []
    for lbl in uniqueLbls:
        idx = [i for i, x in enumerate(lblArr) if x==lbl]
        idx = shuffle(idx, random_state=seed)

        # labeled train
        nl = int(np.ceil(len(idx) * pL))
        tmpL = idx[0:nl]
        trainL_idx.extend(tmpL)

        # unlabeled train
        if pL != 1:
            tmpUL = idx[nl:]
            trainUL_idx.extend(tmpUL)
    return trainL_idx, trainUL_idx


# To count the number of images from each classes
def getIdx(lbl_arr, lbl_search):
    return [i for i, x in enumerate(lbl_arr) if x == lbl_search]

def calWeights(lbl_arr):
    if torch.is_tensor(lbl_arr):
        lbl_arr = lbl_arr.detach().cpu().numpy()

    unique_lbls = np.unique(lbl_arr)

    weights = np.zeros(int(unique_lbls.max()) + 1)

    for lbl in unique_lbls:
        idx = np.where(lbl_arr == lbl)[0]
        weights[int(lbl)] = 1 / len(idx)

    weights = weights / weights.sum()
    weights = torch.tensor(weights, dtype=torch.float32)

    return weights


def getTransform(dataset):
    if dataset == 'DR':
        imsize =  512
    else:
        imsize = 256
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.08, 0.08, 0.08])
    transform_train_W = transforms.Compose([
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize])

    transform_train_S = transforms.Compose([
        transforms.RandomAffine([0, 360], translate=[0.1, 0.1], scale=[0.8, 1.2], shear=20), #translate=[0.02, 0.02],
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    return transform_train_W, transform_train_S, transform_test


def printStatDatasets(dataset_L, dataset_UL, dataset_Te, dataset_Val):
    y_L, y_te, y_val = dataset_L.lblArr, dataset_Te.lblArr, dataset_Val.lblArr
    y_UL = None
    if dataset_UL is not None:
        y_UL = dataset_UL.lblArr
    unique_lbls = np.unique(y_L)

    def countLbls(lblArr, lbl):
        idx = [i for i, x in enumerate(lblArr) if x == lbl]
        return len(idx)

    print('lbl \t Lbl \t UNLbl \t Test \t Validation')
    for lbl in unique_lbls:
        tmp_L = countLbls(y_L, lbl)
        tmp_un = 0
        if y_UL is not None:
            tmp_un = countLbls(y_UL, lbl)
        tmp_te = countLbls(y_te, lbl)
        tmp_val = countLbls(y_val, lbl)
        print('%1d\t%5d\t%5d\t%5d\t%5d'%(lbl, tmp_L, tmp_un, tmp_te, tmp_val))
    if y_UL is not None:
        print('\t%5d\t%5d\t%5d\t%5d' % (len(y_L), len(y_UL), len(y_te), len(y_val)))
    else:
        print('\t%5d\t%5d\t%5d\t%5d' % (len(y_L), 0, len(y_te),len(y_val)))


class DataSetDR(data.Dataset):
    def __init__(self, dataset, fnArr, lblArr, train='TRAIN'):
        self.transformTrain_W, self.transformTrain_S, self.transformTest = getTransform(dataset)
        self.train = train
        self.fnArr = fnArr
        self.lblArr = lblArr

    def __getitem__(self, index):
        img_fn = self.fnArr[index]
        target = self.lblArr[index]
        I = pil_loader(img_fn)

        if self.train in ['TEST', 'VAL']:
            return self.transformTest(I), target
        else:
            return index, self.transformTrain_W(I), self.transformTrain_S(I), target

    def __len__(self):
        return len(self.lblArr)

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def get_datasetsDR(pL, seed):
    src_dir = '/home/suthakaran/Data/DR/'
    # TRAIN
    img_dir = os.path.join(src_dir, 'Preprocessed_512/Train/')
    annot_fn = os.path.join(src_dir, 'trainLabels.csv')
    fnArr_tr, lbl_tr = getData_DR(annot_fn, img_dir, 'TRAIN')

    fnArr_tr = np.array(fnArr_tr)
    lbl_tr = np.array(lbl_tr)

    trainL_idx, trainUL_idx = splitData_Pecentage(lbl_tr, pL, seed)

    xtr = fnArr_tr[trainL_idx].tolist()
    ytr = lbl_tr[trainL_idx].tolist()
    xul = fnArr_tr[trainUL_idx].tolist()
    yul = lbl_tr[trainUL_idx].tolist()

    # TEST
    img_dir = os.path.join(src_dir, 'Preprocessed_512/Test/')
    annot_fn = os.path.join(src_dir, 'retinopathy_solution.csv')
    xte, yte = getData_DR(annot_fn, img_dir, 'TEST')

    # VALIDATION
    xv, yv = getData_DR(annot_fn, img_dir, 'VAL')

    return xtr, ytr, xul, yul, xv, yv, xte, yte


def get_datasetsLIMUC(pL, seed, val_ratio=0.2):
    srcDir = "/home/suthakaran/Data/LIMUC"
    fnArr, lblArr = getData_LIMUC(os.path.join(srcDir, "train_and_validation_sets"))
    fnTrain, lblTrain, fnVal, lblVal = splitTrainValidation(fnArr, lblArr, val_ratio, seed)
    trainL_idx, trainUL_idx = splitData_Pecentage(lblTrain, pL, seed)
    xtr = fnTrain[trainL_idx].tolist()
    ytr = lblTrain[trainL_idx].tolist()

    xul = fnTrain[trainUL_idx].tolist()
    yul = lblTrain[trainUL_idx].tolist()

    xv = fnVal.tolist()
    yv = lblVal.tolist()
    xte, yte = getData_LIMUC( os.path.join(srcDir, "test_set") )
    return xtr, ytr, xul, yul, xv, yv, xte, yte

def getDataLoaders(dataset, pL, seed, bs_L, bs_U, bs_Te, bs_Val):
    if dataset == 'DR':
        xtr, ytr, xul, yul, xv, yv, xte, yte = get_datasetsDR(pL, seed)
    else:
        xtr, ytr, xul, yul, xv, yv, xte, yte = get_datasetsLIMUC(pL, seed)

    uniqueLbls = np.unique(ytr)
    cw = calWeights(np.asarray(ytr))

    trainset_L = DataSetDR(dataset, xtr, ytr, 'TRAIN')
    trainset_UL = None

    if len(xul) > 0:
        trainset_UL = DataSetDR(dataset, xul, yul, 'TRAIN')

    valset = DataSetDR(dataset, xv, yv, 'VAL')
    testset = DataSetDR(dataset, xte, yte, 'TEST')

    trainloader_L = torch.utils.data.DataLoader(trainset_L, batch_size=bs_L, shuffle=True, num_workers=12)

    trainloader_UL = None
    if trainset_UL is not None:
        trainloader_UL = torch.utils.data.DataLoader(trainset_UL, batch_size=bs_U, shuffle=True, num_workers=12)

    valloader = torch.utils.data.DataLoader(valset, batch_size=bs_Val, shuffle=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=bs_Te, shuffle=False, num_workers=4)

    printStatDatasets(trainset_L, trainset_UL, testset, valset)

    return trainloader_L, trainloader_UL, testloader, valloader, cw, uniqueLbls

