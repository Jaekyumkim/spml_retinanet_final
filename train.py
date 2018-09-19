
import pdb
import time
import argparse
import os
import datasets_skimage as datasets

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from utils_aug.augmentations import SSDAugmentation

from model import *
from loss import FocalLoss
from utils import freeze_bn
from logger import Logger
from encoder import DataEncoder

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, trainloader, optimizer, device, epoch, criterion, step, logger, batch_size, loss_fn, loss_accum, conf_thres, nms_thres):
    encoder = DataEncoder(loss_fn, conf_thres, nms_thres)
    model.train()
    freeze_bn(model) # http://qr.ae/TUIS14
    start_time = time.time()
    optimizer.zero_grad()

    batch_loc_loss, batch_cls_loss, batch_num_pos, batch_loss_sum = 0, 0, 0, 0
    batch_loss = 0

    for batch_idx, (data,loc_targets,cls_targets,ori_img_shape, scales, max_shape, ori_shape) in enumerate(trainloader):

        inputs, loc_targets, cls_targets = data.to(device).float(),loc_targets.to(device), cls_targets.to(device)
        optimizer.zero_grad()
        loc_preds_split, cls_preds_split = model(inputs)
        loc_preds = torch.cat(loc_preds_split, 1)
        cls_preds = torch.cat(cls_preds_split, 1)
        # pdb.set_trace()
        loss, loc_loss, cls_loss=criterion(loc_preds.float(), loc_targets.cuda(), \
                cls_preds.float(), cls_targets.cuda())
        batch_cls_loss += cls_loss.detach().cpu().numpy() / loss_accum
        batch_loc_loss += loc_loss.detach().cpu().numpy() / loss_accum
        batch_loss += loss.detach().cpu().numpy() / loss_accum
        loss = torch.div(loss, loss_accum)
        loss.backward()
        if loss_accum and (batch_idx % loss_accum == loss_accum - 1):
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % 10 == 0:
                end_time = time.time()
                print('[%d,%5d] cls_loss: %.5f loc_loss: %.5f train_loss: %.5f time: %.3f lr: %.6f' % \
                        (epoch, step, cls_loss, loc_loss, loss,  \
                        end_time-start_time, optimizer.param_groups[0]['lr']))
                start_time = time.time()
                info = {'training loss': loss.item(), 'loc_loss': loc_loss.item(), \
                        'cls_loss': cls_loss.item()}
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step)

        if batch_idx % 200 == 0:
            pred_boxes, pred_labels, score_all = [], [], []
            if batch_size > 10:
                show_num_img = 10
            else:
                show_num_img = batch_size
            for img_idx in range(show_num_img):
                max_width, max_height = max_shape
                rows, cols = ori_shape[img_idx]
                pred_box, pred_label, score = encoder.decode(loc_preds_split, cls_preds_split, data.shape, (1,3,max_height,max_width), (3,cols,rows),0)
                pred_boxes.append(pred_box)
                pred_labels.append(pred_label)
                score_all.append(score)
            info = {'images': inputs[:show_num_img].cpu().numpy()}
            for tag, images in info.items():
                images = logger.image_drawbox(images, pred_boxes, pred_labels, score_all)
                logger.image_summary(tag, images, step)
    return step

def test(model, testloader, device, criterion, logger, step):
    model.eval()
    losses = AverageMeter()
    losses_loc_loss = AverageMeter()
    losses_cls_loss = AverageMeter()
    for data, loc_targets, cls_targets, _ in testloader:
        inputs = data.to(device)
        loc_preds, cls_preds = model(inputs)
        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)
        loss, loc_loss, cls_loss  = criterion(loc_preds.float(), loc_targets.cuda(), \
                cls_preds.float(), cls_targets.cuda())
        losses.update(loss.item())
        losses_loc_loss.update(loc_loss.item())
        losses_cls_loss.update(cls_loss.item())
        break
    info = {'test loss': losses.avg, 'test_cls_loss': losses_cls_loss.avg, \
            'test_loc_loss': losses_loc_loss.avg}
    for tag, value in info.items():
        logger.scalar_summary(tag, value, step)
    print('\nTest set: Test_loss: %.5f Test_cls_loss: %.5f Test_loc_loss: %.5f\n'%(losses.avg, \
            losses_cls_loss.avg, losses_loc_loss.avg))

def save_checkpoint(state, epoch, save_path):
    path = save_path+"/retina_{}.pth".format(epoch)
    torch.save(state,path)
    print("Checkpoint saved to {}".format(path))

def adjust_learning_rate(optimizer, lr, decay_idx, decay_param):
    for lr_idx in range(decay_idx+1):
        lr /= decay_param
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data','-data', type=str, default='COCO')
    parser.add_argument('--weights','-w', type=str, default='False')
    parser.add_argument('--lr_decay_method','-lrm', type=str, default='retina')
    parser.add_argument('--opt','-opt', type=str, default='SGD')  
    parser.add_argument('--debug','-d', type=str, default='False')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num-workers', '-n', type=int, default=os.cpu_count())
    parser.add_argument('--loss_fn', '-loss', type=str, default='sigmoid')
    parser.add_argument('--batch', '-b', type=int, default=1)
    parser.add_argument('--loss_accum', '-la', type=int, default=1)
    parser.add_argument('--network', '-net', type=str, default='FPN50')
    args = parser.parse_args()

    transform_use = True
    conf_thres = 0.05
    nms_thres = 0.5
    num_workers = args.num_workers
    batch_size = args.batch
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    gpus = [0,1]
    use_cuda = torch.cuda.is_available() 
    opt = args.opt
    step = 0
    if args.data == 'COCO' or 'VOC':
        min_scale = 600
        max_scale = 1000
    elif args.data == 'KITTI':
        min_scale = 384
        max_scale = 1248
    lr_decay_param = 0
    save_path = './weights_'+args.data+'_'+args.loss_fn+'_%.5f_%d/'%(lr,batch_size)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if args.debug == 'True':
        num_workers = 0 

    ##### Data Loading #####
    # transform = transforms.Compose([transforms.ToTensor(), \
            # transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    if transform_use:
        transform = SSDAugmentation()
    else:
        transform = None

    # dataset
    if args.data == "VOC":
        trainlist = '/media/NAS/dataset/PASCALVOC/VOCdevkit/07+12/train.txt'
        testlist = '/media/NAS/dataset/PASCALVOC/VOCdevkit/07+12/test.txt'
        print("==>>  Loading the data.....", args.data)
        trainset = datasets.LoadDataset(trainlist, scale=(min_scale,max_scale), shuffle=True, \
                transform=transform, train=True, batch_size=batch_size, num_workers=num_workers)
        testset = datasets.LoadDataset(testlist, scale=(min_scale,max_scale), shuffle=False, \
                transform=transform, train=False, batch_size=batch_size, num_workers=num_workers)
        if args.loss_fn == "sigmoid":
            num_classes = 20
        elif args.loss_fn == "softmax":
            num_classes = 21
        total_iter = 180000.*4
    elif args.data == "COCO":
        trainlist = '/media/NAS/dataset/COCO/minival2014/train.txt'
        testlist = '/media/NAS/dataset/COCO/minival2014/test.txt'
        print("==>>  Loading the data.....", args.data)
        trainset = datasets.LoadDataset(trainlist, scale=(min_scale,max_scale), shuffle=True, \
                transform=transform, train=True, batch_size=batch_size, num_workers=num_workers)
        testset = datasets.LoadDataset(testlist, scale=(min_scale,max_scale), shuffle=False, \
                transform=transform, train=False, batch_size=batch_size, num_workers=num_workers)
        if args.loss_fn == "sigmoid":
            num_classes = 80
        elif args.loss_fn == "softmax":
            num_classes = 81
        total_iter = 90000.*8
    if args.data == "KITTI":
        trainlist = '/media/NAS/dataset/KITTI/object_detection/training/train_val_split/retinanet/train.txt'
        testlist = '/media/NAS/dataset/KITTI/object_detection/training/train_val_split/retinanet/val.txt'
        print("==>>  Loading the data.....", args.data)
        trainset = datasets.LoadDataset(trainlist, scale=(min_scale,max_scale), shuffle=True, \
                transform=transform, train=True, batch_size=batch_size, num_workers=num_workers)
        testset = datasets.LoadDataset(testlist, scale=(min_scale,max_scale), shuffle=False, \
                transform=transform, train=False, batch_size=batch_size, num_workers=num_workers)
        if args.loss_fn == "sigmoid":
            num_classes = 3
        elif args.loss_fn == "softmax":
            num_classes = 4
        total_iter = 90000*2
    # trainloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, \
            num_workers=num_workers, collate_fn=trainset.collate_fn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, \
            num_workers=num_workers, collate_fn=testset.collate_fn)

    ##### Training Setup #####

    total_epoch = int(total_iter/len(trainloader.dataset)*batch_size)
    print('==>>  Total_epoch size is %d'%(total_epoch))

    lr_decay_method = args.lr_decay_method
    if lr_decay_method == 'luong5':
        start_decay_epoch = int(total_epoch/2)
        decay_times = 5
        remain_epoch = total_epoch - start_decay_epoch
        decay_epochs = [start_decay_epoch]
        for decay_idx in range(decay_times):
            decay_epochs += [int(start_decay_epoch+remain_epoch/decay_times*(decay_idx+1))]
        decay_param = 2
    elif lr_decay_method == 'luong10':
        start_decay_epoch = int(total_epoch/2)
        decay_times = 10
        remain_epoch = total_epoch - start_decay_epoch
        decay_epochs = [start_decay_epoch]
        for decay_idx in range(decay_times):
            decay_epochs += [int(start_decay_epoch+remain_epoch/decay_times*(decay_idx+1))]
        decay_param = 2
    elif lr_decay_method == 'luong234':
        start_decay_epoch = int(total_epoch*2/3)
        decay_times = 4
        remain_epoch = total_epoch - start_decay_epoch
        decay_epochs = [start_decay_epoch]
        for decay_idx in range(decay_times):
            decay_epochs += [int(start_decay_epoch+remain_epoch/decay_times*(decay_idx+1))]
        decay_param = 2
    elif lr_decay_method == 'retina':
        decay_epochs = [int(total_epoch*2/3),int(total_epoch*8/9)]
        decay_param = 10
    decay_idx = 0

    if args.debug == 'True':
        logger = Logger('./logs_debug')
    else:
        logger = Logger('./logs_'+args.data+'_'+args.loss_fn+'_%.5f_%d'%(lr,batch_size))
    
    seed = int(time.time())
    torch.manual_seed(seed)
    global device
    device = torch.device("cuda" if use_cuda else "cpu")

    # setting network
    model = ResNet(num_classes, args.network)

    # setting optimizer
    if opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        print('==>>wrong opt name')

    # setting loss
    criterion = FocalLoss(num_classes,args.loss_fn) # nn.CrossEntropyLoss()

    checkpoint = torch.load('./init_weight/net_{}_{}_{}.pt'.format(args.data,args.loss_fn,args.network))
    model.load_state_dict(checkpoint)

    if use_cuda:
        if len(gpus) > 1:
            model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        else:
            model = model.to(device)
        model.cuda()

    if args.weights:
        if os.path.isfile(args.weights):
            print("==>>  Loading checkpoint '{}'".format(args.weights))
            checkpoint = torch.load(args.weights)
            args.start_epoch = checkpoint['epoch']+1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("==>>  Loaded checkpoint '{}' (epoch {}))".format(args.weights,checkpoint['epoch']))
        else:
            print("==>>  No checkpoint found")

    # train
    for epoch in range(args.start_epoch, total_epoch):
        step = train(model, trainloader, optimizer, device, epoch, criterion, \
               step, logger, batch_size, args.loss_fn, args.loss_accum, conf_thres, nms_thres)
    #    test(model, testloader, device, criterion, logger, step)
        save_checkpoint({'epoch':epoch, 'state_dict':model.state_dict(), \
                'optimizer':optimizer.state_dict()},epoch,save_path)
        if lr_decay_param == 0:
            if epoch == decay_epochs[decay_idx]:
                adjust_learning_rate(optimizer, lr, decay_idx, decay_param)
                decay_idx += 1
                if len(decay_epochs) == decay_idx:
                    lr_decay_param = 1

if __name__ == '__main__':
    main()
