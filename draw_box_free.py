import pdb
import time
import argparse
import os
import datasets_skimage as datasets
from PIL import Image
import numpy as np
import json
import cv2

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from model import *
from loss import FocalLoss
from utils import freeze_bn
from logger import Logger
from encoder import DataEncoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-data', type=str, default='VOC')
    parser.add_argument('--loss_fn', '-loss', type=str, default='sigmoid')
    parser.add_argument('--epoch', '-e', type=str, default='None')
    parser.add_argument('--debug', '-d', type=str, default='False')
    parser.add_argument('--weight_path', '-w', type=str, default='None')
    args = parser.parse_args()

    # KITTI = [384, 1248], COCO,VOC = [480,800]
    if args.data == 'COCO' or 'VOC':
        min_scale = 480
        max_scale = 800
    elif args.data == 'KITTI':
        min_scale = 384
        max_scale = 1248
    nms_thres = 0.5
    conf_thres = 0.5
    use_cuda = torch.cuda.is_available() 
    num_workers = 0
    batch_size = 1
    gpus = [0,1]
    save_path = args.weight_path

    if args.debug == 'True':
        num_workers = 0
    
    transform = transforms.Compose([transforms.ToTensor(), \
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])

    if args.data == "VOC":
        test_root = '/media/NAS/dataset/PASCALVOC/VOCdevkit/07+12/test.txt'
        testset = datasets.LoadDataset(test_root, scale=(min_scale,max_scale), shuffle=False, \
                transform=transform, train=False, batch_size=batch_size, num_workers=num_workers)
        if args.loss_fn == 'sigmoid':
            label = ['aeroplane','bicycle','bird','boat','bottle','bus','car',
                 'cat','chair','cow','diningtable','dog','horse','motorbike'
                 ,'person','pottedplant','sheep','sofa','train','tvmonitor',]
            num_classes = 20
        elif args.loss_fn == 'softmax':
            label = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car',
                 'cat','chair','cow','diningtable','dog','horse','motorbike'
                 ,'person','pottedplant','sheep','sofa','train','tvmonitor',]
            num_classes = 21
        if not len(label) == num_classes:
            print("The label number is wrong")

    elif args.data == "COCO":
        test_root = '/media/NAS/dataset/COCO/minival2014/minival2014.txt'
        testset = datasets.LoadDataset(test_root, scale=(min_scale,max_scale), shuffle=False, \
                transform=transform, train=False, batch_size=batch_size, num_workers=num_workers)
        label_prototxt = '/media/NAS/dataset/COCO/evaluation/labelmap_coco.txt'
        labels = {}
        if args.loss_fn == 'sigmoid':
            num_classes = 80
            label = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        elif args.loss_fn == 'softmax':
            num_classes = 81
            label = ['background', 'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        with open(label_prototxt) as file:
            while True:
                label = file.readline()
                if not label: break
                label = label.rstrip()
                label = label.split(",")
                labels[int(label[1])] = [int(label[0]), label[2]]

    elif args.data == "KITTI":
        test_root = '/media/NAS/dataset/KITTI/object_detection/training/train_val_split/retinanet/val.txt'
        testset = datasets.LoadDataset(test_root, scale=(min_scale,max_scale), shuffle=False, \
                transform=transform, train=False, batch_size=batch_size, num_workers=num_workers)
        if args.loss_fn == 'sigmoid':
            label = ['Car','Pedestrian','Cyclist']
            num_classes = 3
        elif args.loss_fn == 'softmax':
            label = ['Background','Car','Pedestrian','Cyclist']
            num_classes = 4
        if not len(label) == num_classes:
            print("The label number is wrong")

    global device
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, \
            num_workers=num_workers, collate_fn=testset.collate_fn)
    device = torch.device("cuda" if use_cuda else "cpu")

    print('Loading model..')
    if args.data == 'VOC':
       weights = './{}/retina_{}.pth'.format(args.weight_path,args.epoch)
    elif args.data == 'COCO':
       weights = './{}/retina_{}.pth'.format(args.weight_path,args.epoch)
    elif args.data == 'KITTI':
       weights = './{}/retina_{}.pth'.format(args.weight_path,args.epoch)

    model = ResNet(num_classes)
    checkpoint = torch.load(weights)
    if use_cuda:
        if len(gpus) >= 1:
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)
        model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    print('\nTest')

    with open(test_root, 'r') as file:
        lines = file.readlines()

    encoder = DataEncoder(args.loss_fn, conf_thres, nms_thres)
    
    model.eval()
    if args.data == 'VOC':
        result = ''
    elif args.data == 'COCO':
        result = []


    for batch_idx, (data, gt_boxes, gt_labels, scales, max_shape, ori_shape) in enumerate(testloader):
        img_path = lines[batch_idx].rstrip()
        inputs = data.to(device).float()
        rows = ori_shape[0][1]
        cols = ori_shape[0][0]
        max_width = max_shape[0]
        max_height = max_shape[1]
        loc_preds_split, cls_preds_split = model(inputs)
        loc_preds_nms, cls_preds_nms, score = encoder.decode(loc_preds_split,
                                                             cls_preds_split,
                                                             data.shape,
                                                             (1,3,max_width,max_height),
                                                             (3,cols,rows),
                                                             0)

        new_img = cv2.imread(img_path)
        if args.data == 'VOC':
            image_id = img_path[-10:-4]
        elif args.data == 'KITTI':
            image_id = img_path[-10:-4]
        elif args.data == 'COCO':
            image_id = img_path[-16:-4]

        if not os.path.exists(save_path+'/val_epoch_{}/test_img'.format(args.epoch)):
            os.mkdir(save_path+'/val_epoch_{}/test_img'.format(args.epoch))

        gt_x_center = gt_boxes[0,:,0]
        gt_y_center = gt_boxes[0,:,1]
        gt_width = gt_boxes[0,:,2]
        gt_height = gt_boxes[0,:,3]

        gt_xmin = np.array(gt_x_center - gt_width/2)
        gt_ymin = np.array(gt_y_center - gt_height/2)
        gt_xmax = np.array(gt_x_center + gt_width/2)
        gt_ymax = np.array(gt_y_center + gt_height/2)

        gt_xmin_ori = gt_xmin * rows / max_width
        gt_xmax_ori = gt_xmax * rows / max_width
        gt_ymin_ori = gt_ymin * cols / max_height
        gt_ymax_ori = gt_ymax * cols / max_height

        for idx, gt_label in enumerate(gt_labels[0]):
            box_pred_min = (int(gt_xmin_ori[idx]), int(gt_ymin_ori[idx]))
            box_pred_max = (int(gt_xmax_ori[idx]), int(gt_ymax_ori[idx]))
            cls_name = label[int(gt_label)]
            cv2.rectangle(new_img, box_pred_min, box_pred_max, (0,0,255), 2)
            cv2.putText(new_img, cls_name, (box_pred_min[0]+5, box_pred_min[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        if score.shape[0] != 0:
            box_preds = loc_preds_nms.cpu().detach().numpy()
            xmin = box_preds[:,0]
            ymin = box_preds[:,1]
            xmax = box_preds[:,2]
            ymax = box_preds[:,3]
            xmin[xmin < 0] = 0
            ymin[ymin < 0] = 0
            xmax[xmax > rows] = rows
            ymax[ymax > cols] = cols
            box_preds[:,0] = xmin
            box_preds[:,1] = ymin
            box_preds[:,2] = xmax
            box_preds[:,3] = ymax
            box_preds = box_preds.astype(str)
            box_preds = np.ndarray.tolist(box_preds)
            category_preds = cls_preds_nms.cpu().detach().numpy().astype(str)
            category_preds = np.ndarray.tolist(category_preds)
            score_preds = score.cpu().detach().numpy().astype(str)
            score_preds = np.ndarray.tolist(score_preds)
            for i in range(len(score_preds)):
                ## [Image ID, Class, Box()]
                box_pred_min = (int(float(box_preds[i][0])), int(float(box_preds[i][1])))
                box_pred_max = (int(float(box_preds[i][2])), int(float(box_preds[i][3])))
                conf = str(float('{:03.2f}'.format(float(score_preds[i]))))
                cls_name = label[int(category_preds[i])]
                cv2.rectangle(new_img, box_pred_min, box_pred_max, (0,250,0), 2)
                cv2.putText(new_img, cls_name + ' | ' + conf, (box_pred_min[0]+5, box_pred_min[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        print(img_path)
        cv2.imwrite(save_path+'/val_epoch_{}/test_img/{}.jpg'.format(args.epoch, image_id), new_img)

        

if __name__ == '__main__':
    main()
