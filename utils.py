import pdb
import math

import torch
import torch.nn as nn

def meshgrid(x, y, row_major=True):
    '''Return meshgrid in  range x& y

    Args:
        x: (int) first dim range
        y: (int) second dim range
        row_major: (bool) row major or column major

    Returns:
        meshgrid: (tensor) size[x*y,2]

    Example:
    >>meshgrid(3,2)     >>meshgrid(3,2,row_major=False)
    0 0                 0 0
    1 0                 0 1
    2 0                 0 2
    0 1                 1 0
    1 1                 1 1
    2 1                 1 2
    '''

    w = torch.arange(0,x)
    h = torch.arange(0,y)
    xx = w.repeat(y).view(-1,1)
    yy = h.view(-1,1).repeat(1,x).view(-1,1)
    if row_major:
        xy=torch.cat([xx,yy],1)
    else:
        xy=torch.cat([yy,xx],1)
    return xy


def change_box_order(boxes, order):
    '''Change the order xywh -> xyxy

    Args:
        boxes: (tensor) bounding bounding box, [N,4]
        order: (str) either 'xyxy2xywh' or 'xywh2xyxy'

    Returns:
        boxes: (tensor) converted bounding box, [N,4]
    '''
    assert order in ['xyxy2xywh', 'xywh2xyxy']
    a123 = boxes[:,:2]
    b123 = boxes[:,2:]
    #pdb.set_trace()
    if order == 'xyxy2xywh':
        return torch.cat([(a123+b123)/2, b123-a123+1], 1)
    return torch.cat([a123-b123/2,a123+b123/2], 1)

def box_ious(box1, box2, order='xywh'):
    '''Compute the intersection over union of two set of boxes

    Args:
        box1: (tensor) bounding boxes, size [N,4]
        box2: (tensor) bounding boxes, size [M,4]
        order: (str) box order, either 'xyxy' or 'xywh'

    Return:
        iou: (tensor) sized [N,M]
    '''
    if order == 'xywh':
        box1_coor = change_box_order(box1, 'xywh2xyxy')
        box2_coor = change_box_order(box2, 'xywh2xyxy')

    left_top = torch.max(box1_coor[:,None, :2], box2_coor[:,:2])        # [N,M,2]
    right_bottom = torch.min(box1_coor[:,None, 2:], box2_coor[:,2:])    # [N,M,2]

    wh = (right_bottom - left_top+1).clamp(min=0)                       # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]                                       # [N,M]

    area1 = (box1[:,2] + 1) * (box1[:,3] + 1)                           # [N, ]
    area2 = (box2[:,2] + 1) * (box2[:,3] + 1)                           # [M, ]
    iou = inter / (area1[:,None] + area2 - inter)                       # [N,M]
    return iou

def box_nms(boxes, scores, threshold=0.5):
    '''Non maximum suppresion

    Args:
        bboxes: (tensor) bounding boxes, sized[N,4]
        scores: (tensor) bbox scores, sized[N, ]
        threhold: (float) overlap threshold

    Returns:
        keep: (tensor) selected indices

    Reference:
        https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py  
    '''
    try:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
    except:
        pdb.set_trace()

    areas = (x2 - x1 + 1) * (y2-y1+1)
    _, order = scores.sort(0, descending = True)

    keep_order = []
    while order.numel() > 0:
        i = order[0]
        keep_order.append(i)

        if order.numel() == 1:
            break

        inter_x1 = x1[order[1:]].clamp(min=float(x1[i]))
        inter_y1 = y1[order[1:]].clamp(min=float(y1[i]))
        inter_x2 = x2[order[1:]].clamp(max=float(x2[i]))
        inter_y2 = y2[order[1:]].clamp(max=float(y2[i]))

        inter_w = (inter_x2 - inter_x1 + 1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1 + 1).clamp(min=0)
        inter = inter_w * inter_h

        over = inter / (areas[i] + areas[order[1:]] - inter)

        ids = (over <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]

    return torch.LongTensor(keep_order)
        
def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form

    Args:
        labels: (LongTensor) class labels, sized [#labels,]
        num_classes: (int) number of classes

    Return:
        one_hot_label: (tensor) encoded labels, size [#labels, #classes]
    '''
    one_hot = torch.eye(num_classes)            # [#classes, #classes]
    return one_hot[labels.long()]               # [#labels,  #classes]

def freeze_bn(model):
    '''Freeze models BN parameter
    Args:
        model: (nn.Module) 

    Return:
        model: (nn.Module) freezed BN model
    '''
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()
