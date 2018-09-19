
import os
import pdb
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw
import skimage.io
import skimage.transform
import skimage.color
import skimage
import cv2
from utils_aug.augmentations import SSDAugmentation
import copy
from encoder import DataEncoder
import torchvision.transforms.functional as F2

kitti_label = ['car', 'pedestrian', 'cyclist']

voc_label = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

coco_label = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
#
class LoadDataset(Dataset):
    def __init__(self, root, scale=None, shuffle=True, transform=None, train=False, \
            batch_size=16, num_workers=2):
        with open(root, 'r') as file:
            self.lines = file.readlines()

        if shuffle:
            random.shuffle(self.lines)

        self.nSamples = len(self.lines)
        self.transform = transform
        self.train = train
        self.min_scale = scale[0]
        self.max_scale = scale[1]
        self.batch_size = batch_size

        self.encoder = DataEncoder()
        self.mean = (0.485,0.456,0.406)
        self.std = (0.229,0.224,0.225)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        ori_img_size, img, label, scale = load_data_detection(imgpath, self.min_scale, self.max_scale, self.train, self.transform)
        #pdb.set_trace()
        img = torch.from_numpy(img)
        img = img.permute(2,0,1)

        norm_img = F2.normalize(img/255., self.mean, self.std)

        label = torch.from_numpy(label)  

        boxes = label[:,1:]    # split the bbx label and cls label
        labels = label[:,0]
        #pdb.set_trace()
        return (ori_img_size, norm_img, boxes, labels, scale)

    def collate_fn(self, batch):
        ori_shape = [x[0] for x in batch]
        imgs = [x[1] for x in batch]
        boxes = [x[2] for x in batch]
        labels = [x[3] for x in batch]
        scales = [x[4] for x in batch]
        
        inputs = []
        ori_img_shape = []
        for img in imgs:
            ori_img_shape.append(img.shape)
        max_shape = np.array(ori_img_shape).max(axis=0)
        max_width = max_shape[2]
        max_height = max_shape[1]

        if max_width%32 == 0:
            pad_w = 0
        else:
            pad_w = 32 - max_width%32
        if max_height%32 == 0:
            pad_h = 0
        else:
            pad_h = 32 - max_height%32

        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, max_height + pad_h, max_width + pad_w)
        #pdb.set_trace()
        inputs += torch.tensor(-2.1179)
        loc_targets = []
        cls_targets = []
        # pdb.set_trace()
        for i in range(num_imgs):
            inputs[i, :, 0:imgs[i].shape[1], 0:imgs[i].shape[2]] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i],\
                    input_size=(inputs.shape[3], inputs.shape[2]))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        if self.train:
            return inputs, torch.stack(loc_targets), torch.stack(cls_targets), ori_img_shape, scales, (max_width, max_height), ori_shape
        else:
            return inputs, torch.stack(boxes), torch.stack(labels), scales, (max_width, max_height), ori_shape


def load_data_detection(imgpath, min_scale, max_scale, train, transform):
    labelpath = imgpath.replace('rgb','labels').replace('images','labels').replace('JPEGImages','labels').replace('.jpg','.txt').replace('.png','.txt')
    img = skimage.io.imread(imgpath)
    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)

    ori_img_size, resized_img, scale = data_augmentation(img, min_scale, max_scale, train)  # augment the img 
    label = load_label_minmax(labelpath, img, resized_img) # load the label
    if train:
        if transform is not None:
            img_aug, boxes, labels = transform(resized_img, label[:,1:5], label[:,0])
            img_aug = cv2.resize(img_aug, (resized_img.shape[1], resized_img.shape[0]))
    else:
        img_aug = resized_img
        boxes = label[:,1:5]
        labels = label[:,0]
        
    height, width, channels = img_aug.shape

    boxes[:, 0] *= width
    boxes[:, 2] *= width
    boxes[:, 1] *= height
    boxes[:, 3] *= height

    boxes_wh = minmax2wh(boxes)

    label_aug = np.concatenate([np.reshape(labels,[len(labels),1]),boxes_wh],axis=1)

    return ori_img_size, img_aug, label_aug, scale

def data_augmentation(img, min_scale, max_scale, train):
    rows, cols, channels = img.shape
    smallest_side = min(rows, cols)
    scale = min_scale / smallest_side 
    largest_side = max(rows, cols)

    if largest_side * scale > max_scale:
        scale = max_scale / largest_side

    resized_img = skimage.transform.resize(img, (int(round(rows*scale)), int(round((cols*scale)))), order=0, preserve_range=True)
    return (rows, cols), resized_img, scale

def minmax2wh(bbx_ori):
    bbx = np.zeros_like(bbx_ori)

    xmin = bbx_ori[:,0]
    xmax = bbx_ori[:,2]
    ymin = bbx_ori[:,1]
    ymax = bbx_ori[:,3]

    bbx[:,0] = ((xmax + xmin)/2)   # center_x
    bbx[:,1] = ((ymin + ymax)/2)   # center_y
    bbx[:,2] = ((xmax - xmin))     # width
    bbx[:,3] = ((ymax - ymin))     # height
    return bbx 

def load_label(labelpath, img, resized_img):
    # pdb.set_trace()
    if os.path.isfile(labelpath) == False:
        bbx = np.zeros([1,5]) - 10000
    else:
        bbx = np.loadtxt(labelpath) # load the label
        img_height, img_width, img_channel = img.shape
        resized_img_height, resized_img_width, resized_img_channel = resized_img.shape
        if len(bbx.shape) == 1:
            bbx = np.reshape(bbx,[1,5])  # if the label is only one, we have to resize the shape of the bbx
        x1 = (bbx[:,1] - bbx[:,3]/2)*img_width  # calculate the original label x1_min
        x2 = (bbx[:,1] + bbx[:,3]/2)*img_width  # calculate the original label x2_max
        y1 = (bbx[:,2] - bbx[:,4]/2)*img_height # calculate the original label y1_min
        y2 = (bbx[:,2] + bbx[:,4]/2)*img_height # calculate the original label y2_max
        r_x1 = x1 * resized_img_width / img_width     # calculate the resized label x1_min
        r_x2 = x2 * resized_img_width / img_width     # calculate the resized label x2_max
        r_y1 = y1 * resized_img_height / img_height   # calculate the resized label y1_min
        r_y2 = y2 * resized_img_height / img_height   # calculate the resized label y2_max
        bbx[:,1] = ((r_x1 + r_x2)/2)   # center_x
        bbx[:,2] = ((r_y1 + r_y2)/2)   # center_y
        bbx[:,3] = ((r_x2 - r_x1))     # width
        bbx[:,4] = ((r_y2 - r_y1))     # height
    return bbx 
def load_label_minmax(labelpath, img, resized_img):
    # pdb.set_trace()
    if os.path.isfile(labelpath) == False:
        bbx = np.zeros([1,5]) - 10000
    else:
        bbx = np.loadtxt(labelpath) # load the label
        img_height, img_width, img_channel = img.shape
        resized_img_height, resized_img_width, resized_img_channel = resized_img.shape
        if len(bbx.shape) == 1:
            bbx = np.reshape(bbx,[1,5])  # if the label is only one, we have to resize the shape of the bbx
        x1 = (bbx[:,1] - bbx[:,3]/2)*img_width  # calculate the original label x1_min
        x2 = (bbx[:,1] + bbx[:,3]/2)*img_width  # calculate the original label x2_max
        y1 = (bbx[:,2] - bbx[:,4]/2)*img_height # calculate the original label y1_min
        y2 = (bbx[:,2] + bbx[:,4]/2)*img_height # calculate the original label y2_max
        r_x1 = x1 * resized_img_width / img_width     # calculate the resized label x1_min
        r_x2 = x2 * resized_img_width / img_width     # calculate the resized label x2_max
        r_y1 = y1 * resized_img_height / img_height   # calculate the resized label y1_min
        r_y2 = y2 * resized_img_height / img_height   # calculate the resized label y2_max
        bbx[:,1] = r_x1   # x1_min
        bbx[:,2] = r_y1   # y1_min
        bbx[:,3] = r_x2     # x2_max
        bbx[:,4] = r_y2     # y2_max
    return bbx 


def debug_img(img, labels):
    import cv2
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # img = img * std + mean
    # img = (img*255.0).astype(np.uint8)
    img = img.astype(np.uint8)
    COLOR = (255,0,0)
    for label in labels:
        xywh = [int(label[1]), int(label[2]), int(label[3]), int(label[4])]
        xyxy = [int(xywh[0]-xywh[2]/2), int(xywh[1]-xywh[3]/2), int(xywh[0]+xywh[2]/2), int(xywh[1]+xywh[3]/2)]
        coor_min = (xyxy[0], xyxy[1])
        coor_max = (xyxy[2], xyxy[3])
        cv2.rectangle(img, coor_min, coor_max, COLOR, 2)
        cv2.putText(img, coco_label[int(label[0])] + ' | ' + 'GT', (coor_min[0]+5, coor_min[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite('a.png', img)
    pdb.set_trace()

def test():
    import torchvision
# 
    transform = transforms.Compose([transforms.ToTensor(), \
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    transform = SSDAugmentation()
    trainlist = '/media/NAS/dataset/PASCALVOC/train.txt'
    dataset = VOCDataset(trainlist, shape=(600,600), shuffle=True, transform=transform, \
            train=True,batch_size=16,num_workers=0)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, \
            num_workers=0, collate_fn=dataset.collate_fn)

    for images, loc_targets, cls_targets in trainloader:
        print(images.size())
        print(loc_targets.size())
        print(cls_targets.size())
        grid = torchvision.utils.make_grid(images, 1)
        torchvision.utils.save_image(grid, 'a.jpg')

#test()

