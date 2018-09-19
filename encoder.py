import pdb
import math

import torch

from utils import meshgrid, box_ious, box_nms, one_hot_embedding

class DataEncoder:
    def __init__(self, loss_fn='sigmoid', conf_thres=0.05, nms_thres=0.5):
        self.anchor_areas = [32*32., 64*64., 128*128., 256*256., 512*512.]
        self.aspect_ratios = [1/2., 1/1., 2/1.]
        self.scale_ratios = [1., pow(2,1/3.), pow(2,2/3.)]
        self.num_anchors = len(self.aspect_ratios)*len(self.scale_ratios)
        self.num_fms = len(self.anchor_areas)
        self.anchor_wh = self._get_anchor_wh()
        self.loss_fn = loss_fn
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

    def _get_anchor_wh(self):
        '''Compute anchor width and height for each feature map.

        Returns:
          anchor_wh: (tensor) anchor wh, sized [#self.num_fms, #anchors_per_cell, 2].
        '''
        anchor_wh = []
        for anchor_size in self.anchor_areas:
            for ar in self.aspect_ratios:
                width = math.sqrt(anchor_size/ar)
                height = ar * width
                for sr in self.scale_ratios:
                    anchor_h = height * sr
                    anchor_w = width * sr
                    anchor_wh.append([anchor_w,anchor_h])
        return torch.Tensor(anchor_wh).view(self.num_fms,-1,2)

    def _get_anchor_boxes(self, input_size):
        '''Compute anchor boxes for each feature map.

        Args:
          input_size: (tensor) model input size of (w,h).

        Returns:
          boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                        where #anchors = fmw * fmh * #anchors_per_cell
        '''
        fm_sizes = [(input_size/pow(2.,i+3)).ceil() for i in range(self.num_fms)]
        boxes = []
        grid_size = [8., 16., 32., 64., 128.]
        for i in range(self.num_fms):
            fm_size = fm_sizes[i]
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_w, fm_h) + 0.5
            xy = (xy*grid_size[i]).view(fm_w,fm_h,1,2).expand(fm_w,fm_h,9,2)
            xy = xy.to(torch.float32)
            wh = self.anchor_wh[i].view(1,1,self.num_anchors,2).expand(fm_w,fm_h,self.num_anchors,2)
            wh = wh.to(torch.float32)
            box = torch.cat([xy,wh],3)
            boxes.append(box.view(-1,4))
        return boxes

    def encode(self, boxes, labels, input_size):
        '''Encode target bounding boxes and class labels.

        We obey the Faster RCNN box coder:
          tx = (x - anchor_x) / anchor_w
          ty = (y - anchor_y) / anchor_h
          tw = log(w / anchor_w)
          th = log(h / anchor_h)

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        '''
        input_size = torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)
        anchor_boxes = torch.cat(anchor_boxes,0)
        anchor_boxes = anchor_boxes.to(torch.float64)

        ious = box_ious(anchor_boxes,boxes)
        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]

        loc_xy = (boxes[:,:2]-anchor_boxes[:,:2])/anchor_boxes[:,2:]
        loc_wh = torch.log(boxes[:,2:]/anchor_boxes[:,2:])
        loc_targets = torch.cat([loc_xy,loc_wh],1)
        cls_targets = 1 + labels[max_ids]

        cls_targets[max_ious<0.5] = 0
        ignore = (max_ious>0.4) & (max_ious<0.5)
        cls_targets[ignore] = -1
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, pad_data, input_size, ori_img_shape, img_idx):
        '''Decode outpus back to bounding box locations and class labels

        Args:
            loc_preds: (tesnor) predicted locations, sized [#anchors, 4].
            cls_preds: (tensor) predicted class labels, sized [#anchors, #classes].
            input_size : (int/tuple) model input size of (w,h)

        Returns:
            boxes (tensor) decode box locations, size [#obj, 4]
            lbaels: (tensor) class labels for each box, sized [#obj,]
        '''
        CONF_THRES = self.conf_thres
        NMS_THRES = self.nms_thres
        # pdb.set_trace()
        input_size = torch.Tensor([input_size[2], input_size[3]])
        pad_data = torch.Tensor([pad_data[3], pad_data[2]])
        anchor_boxes = self._get_anchor_boxes(pad_data)
        boxes_preds_obj = []
        score_obj = []
        labels_obj = []
        obj_idx = []

        for p in range(len(anchor_boxes)):
            loc_preds[p][img_idx][:,:2] *= 0.4
            loc_preds[p][img_idx][:,2:] *= 0.8
            loc_xy_preds = loc_preds[p][img_idx][:, :2]
            loc_wh_preds = loc_preds[p][img_idx][:, 2:]

            xy_preds = loc_xy_preds * anchor_boxes[p][:, 2:].cuda() +\
                       anchor_boxes[p][:, :2].cuda()
            wh_preds = torch.exp(loc_wh_preds) * anchor_boxes[p][:,2:].cuda()
            x1y1_preds = xy_preds - wh_preds/2
            x1y1_preds_ori = torch.zeros(x1y1_preds.shape)
            x1y1_preds_ori[:,0] = x1y1_preds[:,0] * torch.Tensor([ori_img_shape[2]]).\
                             cuda() / torch.Tensor([input_size[0]]).cuda()
            x1y1_preds_ori[:,1] = x1y1_preds[:,1] * torch.Tensor([ori_img_shape[1]]).\
                             cuda() / torch.Tensor([input_size[1]]).cuda()
            x2y2_preds = xy_preds + wh_preds/2
            x2y2_preds_ori = torch.zeros(x2y2_preds.shape)
            x2y2_preds_ori[:,0] = x2y2_preds[:,0] * torch.Tensor([ori_img_shape[2]]).\
                             cuda() / torch.Tensor([input_size[0]]).cuda()
            x2y2_preds_ori[:,1] = x2y2_preds[:,1] * torch.Tensor([ori_img_shape[1]]).\
                             cuda() / torch.Tensor([input_size[1]]).cuda()
            boxes_preds = torch.cat([x1y1_preds_ori, x2y2_preds_ori], 1)

            score, labels = cls_preds[p][img_idx].sigmoid().max(1)
            if self.loss_fn == 'sigmoid':
                obj_idx_p = score > CONF_THRES
            elif self.loss_fn == 'softmax':
                obj_idx_p = torch.mul(score>CONF_THRES, labels>0)
            # if boxes_preds[obj_idx_p].shape[0] > 1000:
            #     boxes_preds_obj.append(boxes_preds[obj_idx_p][:1000])
            #     score_obj.append(score[obj_idx_p][:1000])
            #     labels_obj.append(labels[obj_idx_p][:1000])
            # else:
            boxes_preds_obj.append(boxes_preds[obj_idx_p])
            score_obj.append(score[obj_idx_p])
            labels_obj.append(labels[obj_idx_p])
            obj_idx.append(obj_idx_p)

        boxes_preds_all = torch.cat(boxes_preds_obj, 0)
        score_all = torch.cat(score_obj, 0)
        labels_all = torch.cat(labels_obj, 0)
        obj_idx_all = torch.cat(obj_idx, 0)
        if obj_idx_all.nonzero().shape[0]!=0:
            nms_boxes = box_nms(boxes_preds_all, score_all, threshold=NMS_THRES)
            return boxes_preds_all[nms_boxes], labels_all[nms_boxes], score_all[nms_boxes]
        else:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
