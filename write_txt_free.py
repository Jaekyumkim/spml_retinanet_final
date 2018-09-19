import pdb
import time
import argparse
import os
import datasets_skimage as datasets
from PIL import Image
import numpy as np
import json

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
    parser.add_argument('--network', '-net', type=str, default='FPN50')
    args = parser.parse_args()

    # KITTI = [384, 1248], COCO,VOC = [480,800]
    if args.data == 'COCO' or 'VOC':
        min_scale = 480
        max_scale = 800
    elif args.data == 'KITTI':
        min_scale = 384
        max_scale = 1248
    nms_thres = 0.5
    conf_thres = 0.05
    use_cuda = torch.cuda.is_available() 
    if args.debug == 'True':
        num_workers = 0
    num_workers = os.cpu_count()
    batch_size = 1
    gpus = [0]
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
        testset = dataset.LoadDataset(test_root, scale=(min_scale,max_scale), shuffle=False, \
                transform=transform, train=False, batch_size=batch_size, num_workers=num_workers)
        label_prototxt = '/media/NAS/dataset/COCO/evaluation/labelmap_coco.txt'
        labels = {}
        if args.loss_fn == 'sigmoid':
            num_classes = 80
        elif args.loss_fn == 'softmax':
            num_classes = 81
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
    weights = './{}/retina_{}.pth'.format(args.weight_path,args.epoch)

    model = ResNet(num_classes, args.network)
    checkpoint = torch.load(weights)
    if use_cuda:
        if len(gpus) >= 1:
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)
        model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    print('\nTest')
    
    print('\nTest')

    with open(test_root, 'r') as file:
        lines = file.readlines()

    encoder = DataEncoder(args.loss_fn, conf_thres, nms_thres)
    
    model.eval()
    if args.data == 'VOC':
        result = ''
    elif args.data == 'COCO':
        result = []
    elif args.data == 'KITTI':
        result = ''


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

        if args.data == 'VOC':
            image_id = img_path[-10:-4]

            if not os.path.exists(save_path+'/val_epoch_{}'.format(args.epoch)):
                os.mkdir(save_path+'/val_epoch_{}'.format(args.epoch))

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
                    #pdb.set_trace()
                    #if category_preds[i] == 0:
                    result = ' '.join([image_id,score_preds[i],
                                   box_preds[i][0], box_preds[i][1],
                                   box_preds[i][2],box_preds[i][3]]) + '\n'
                    f = open(save_path+'/val_epoch_{}/comp3_det_test_{}.txt'\
                        .format(args.epoch,label[int(category_preds[i])]), 'a')
                    f.write(result)

            print(img_path)

        if args.data == 'KITTI':
            image_id = img_path[-10:-4]

            if not os.path.exists(save_path+'/val_epoch_{}'.format(args.epoch)):
                os.mkdir(save_path+'/val_epoch_{}'.format(args.epoch))

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
                box_preds[:,0] = xmin.astype(int).astype(float)
                box_preds[:,1] = ymin.astype(int).astype(float)
                box_preds[:,2] = xmax.astype(int).astype(float)
                box_preds[:,3] = ymax.astype(int).astype(float)
                box_preds = box_preds.astype(str)
                box_preds = np.ndarray.tolist(box_preds)
                category_preds = cls_preds_nms.cpu().detach().numpy().astype(str)
                category_preds = np.ndarray.tolist(category_preds)
                score_preds = score.cpu().detach().numpy().astype(str)
                score_preds = np.ndarray.tolist(score_preds)
                for i in range(len(score_preds)):
                    ## [Image ID, Class, Box()]
                    class_name = label[int(category_preds[i])]
                    result = ' '.join([class_name, '-1', '-1', '-10',
                                       box_preds[i][0], box_preds[i][1],
                                       box_preds[i][2],box_preds[i][3], '-1', '-1', '-1', '-1000', '-1000', '-1000', '-10', score_preds[i]]) + '\n'
                    f = open(save_path+'/val_epoch_{}/{}.txt'.format(args.epoch,image_id), 'a')
                    f.write(result)

            else:
                f = open(save_path+'/val_epoch_{}/{}.txt'.format(args.epoch,image_id), 'a')
                f.write(result)

            print(img_path)

        elif args.data == 'COCO':
            image_id = int(img_path[-16:-4])
            if not os.path.exists(save_path+'/val_epoch_{}'.format(args.epoch)):
                os.mkdir(save_path+'/val_epoch_{}'.format(args.epoch))

            if score.shape[0] != 0:
                box_preds = loc_preds_nms.cpu().detach().numpy()
                xmin   = box_preds[:,0]
                ymin   = box_preds[:,1]
                xmax   = box_preds[:,2]
                ymax   = box_preds[:,3]
                xmin[xmin < 0] = 0
                ymin[ymin < 0] = 0
                xmax[xmax > rows] = rows
                ymax[ymax > cols] = cols
                width  = xmax-xmin
                height = ymax-ymin
                box_preds[:,0] = xmin
                box_preds[:,1] = ymin
                box_preds[:,2] = width
                box_preds[:,3] = height
                box_preds = np.ndarray.tolist(box_preds)
                category_preds = cls_preds_nms.cpu().detach().numpy()
                category_preds = np.ndarray.tolist(category_preds)
                score_preds = score.cpu().detach().numpy()
                score_preds = np.ndarray.tolist(score_preds)

                for idx in range(len(box_preds)):
                    j = {"image_id": image_id, 
                             "category_id": labels[category_preds[idx]+1][0],
                             "bbox": box_preds[idx],
                             "score": score_preds[idx]}
                    result.append(j)
            else: continue
            print(img_path)
    if args.data == 'COCO':
        json.dump(result, open(save_path+'/val_epoch_{}/coco_results.json'.format(args.epoch),'w'))

        

if __name__ == '__main__':
    main()
