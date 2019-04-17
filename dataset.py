#!/usr/bin/env python4
# -*- coding: utf-8 -*-
"""
Created on 2019/04/15
author: lujie
"""

import numpy as np
import pandas as pd
import pandas
import numpy
import json
import torch.utils.data as data
import os
import torch
from IPython import embed

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

class VideoDataSet(data.Dataset):

    def __init__(self, opt, subset = "train"):
        self.temporal_scale = opt["temporal_scale"]   # 100
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = opt["mode"]                       # 
        self.feature_path    = opt["feature_path"]    # two-stream feat
        self.boundary_ratio  = opt["boundary_ratio"]  # 0.1
        self.video_info_path = opt["video_info"]      # video_info
        self.video_anno_path = opt["video_anno"]      # video_annotation
        self._getDatasetDict()

    def _getDatasetDict(self):
        ''' Match the annotations for each video from anet_anno_actio '''

        df_video_info = pd.read_csv(self.video_info_path)
        df_anno_info  = load_json(self.video_anno_path)
        self.video_dict = {}

        for i in range(len(df_video_info)):

            video_name = df_video_info.video.values[i]
            anno_info  = df_anno_info[video_name]
            video_subset = df_video_info.subset.values[i]

            if self.subset == "full":
                self.video_dict[video_name] = anno_info
            if self.subset in video_subset:
                self.video_dict[video_name] = anno_info

        self.video_list = list(self.video_dict.keys())    # video name list
        print("%s subset video numbers: %d" %(self.subset,len(self.video_list)))

    def __getitem__(self, index):
        ''' get item for DataLoader '''

        video_data, anchor_xmin, anchor_xmax = self._get_base_data(index)

        if self.mode == "train":
            match_score_action, match_score_start, match_score_end = self._get_train_label(index, anchor_xmin, anchor_xmax)
            return video_data, match_score_action, match_score_start, match_score_end
        else:
            return index, video_data, anchor_xmin, anchor_xmax

    def _get_base_data(self,index):
        ''' Load the two-stream truncated-feature ''' 

        video_name  = self.video_list[index]
        anchor_xmin = [self.temporal_gap*i for i in range(self.temporal_scale)]
        anchor_xmax = [self.temporal_gap*i for i in range(1, self.temporal_scale+1)]
        df_video = pd.read_csv(self.feature_path+ "csv_mean_"+str(self.temporal_scale)+"/"+video_name+".csv")
        video_data = df_video.values[:,:]
        video_data = torch.Tensor(video_data)
        video_data = torch.transpose(video_data, 0, 1)
        video_data.float()
        return video_data, anchor_xmin, anchor_xmax

    def _get_train_label(self, index, anchor_xmin, anchor_xmax):
        '''
        Generate label accroding anno-info
        step - 1. scale the length of video to [0, 1]
        step - 2. scale the anno-info according to start, end time
        step - 3. generate the action-interval, start and end point in unit-interval [0, 1]
        ''' 

        # step - 1
        video_name = self.video_list[index]
        video_info = self.video_dict[video_name]
        video_frame  = video_info['duration_frame']
        video_second = video_info['duration_second']
        feat_frame   = video_info['feature_frame']    # what ?
        feat_second  = float(feat_frame) / video_frame * video_second
        video_labels = video_info['annotations']      # action_info {label, segment}
        
        # step - 2
        gt_bbox = []
        for j in range(len(video_labels)):
            tmp_info  = video_labels[j]
            tmp_start = max(min(1,tmp_info['segment'][0]/feat_second),0)
            tmp_end   = max(min(1,tmp_info['segment'][1]/feat_second),0)
            gt_bbox.append([tmp_start, tmp_end])
        gt_bbox = np.array(gt_bbox)
        gt_xmins, gt_xmaxs = gt_bbox[:,0], gt_bbox[:,1]
        
        # step - 3
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)         # default : boundary_ratio : 0.1
        gt_start_bboxs = np.stack((gt_xmins-gt_len_small/2, gt_xmins+gt_len_small/2), axis=1) 
        gt_end_bboxs   = np.stack((gt_xmaxs-gt_len_small/2, gt_xmaxs+gt_len_small/2), axis=1)

        match_score_action, match_score_start, match_score_end = [], [], []
        for k in range(len(anchor_xmin)):
            match_score_action.append(np.max(self._ioa_with_anchors(anchor_xmin[k], anchor_xmax[k], gt_xmins, gt_xmaxs)))
            match_score_start.append(np.max(self._ioa_with_anchors(anchor_xmin[k], anchor_xmax[k], gt_start_bboxs[:,0], gt_start_bboxs[:,1])))
            match_score_end.append(np.max(self._ioa_with_anchors(anchor_xmin[k], anchor_xmax[k], gt_end_bboxs[:,0], gt_end_bboxs[:,1])))

        match_score_action = torch.Tensor(match_score_action)
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)

        return match_score_action, match_score_start, match_score_end

    def _ioa_with_anchors(self, anchors_min, anchors_max, box_min, box_max):
        ''' Calculate the match score '''
 
        len_anchors = anchors_max - anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        scores = np.divide(inter_len, len_anchors)

        return scores

    def __len__(self):
        return len(self.video_list)


class ProposalDataSet(data.Dataset):
    def __init__(self,opt,subset="train"):

        self.subset = subset
        self.mode = opt["mode"]
        if self.mode == "train":
            self.top_K = opt["pem_top_K"]  # 500
        else:
            self.top_K = opt["pem_top_K_inference"] # 1000

        self.video_info_path = opt["video_info"]   #  info of video
        self.video_anno_path = opt["video_anno"]   #  annotations of video
        self.video_list = None
        self._getDatasetDict()


    def _getDatasetDict(self):
        ''' Match the annotations for each video from anet_anno_action '''

        df_video_info = pd.read_csv(self.video_info_path)
        df_anno_info  = load_json(self.video_anno_path)
        self.video_dict = {}
 
        for i in range(len(df_video_info)):

            video_name = df_video_info.video.values[i]
            video_subset = df_video_info.subset.values[i]
            anno_info = df_anno_info[video_name]

            if self.subset == "full":
                self.video_dict[video_name] = anno_info
            if self.subset in video_subset:
                self.video_dict[video_name] = anno_info

        self.video_list = list(self.video_dict.keys())
        print ("%s subset video numbers: %d" %(self.subset, len(self.video_list)))

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        ''' Return video_feature and video match_iou '''
        
        video_name = self.video_list[index]
       
        # there may be multi-proposals for each video
        # bsp_feat for each proposal
        video_feature = numpy.load('./output/PGM_feature/' + video_name + '.npy')
        video_feature = video_feature[:self.top_K, :]
        video_feature = torch.Tensor(video_feature)
        
        num_prop = len(video_feature)
   
        # [start, end, start_score, end_score, score, _match_iou, _match_ioa], where score = start_score * end_score
        df_prop = pandas.read_csv('./output/PGM_proposals/' + video_name + '.csv')
        df_prop = df_prop[:self.top_K]   # BUG

        # there may be multi-proposals for each video
        # bsp_feat for each proposal
        video_feature = numpy.load('./output/PGM_feature/' + video_name + '.npy')
        video_feature = video_feature[:self.top_K, :]
        video_feature = torch.Tensor(video_feature)
    
        cache = ()
        if self.mode == 'train':
            video_match_iou = torch.Tensor(df_prop.match_iou.values[:])  
            cache = (video_feature, video_match_iou)
            #return video_feature, video_match_iou
        else:
            video_xmin = df_prop.xmin.values[:]
            video_xmax = df_prop.xmax.values[:]
            video_xmin_score = df_prop.xmin_score.values[:]
            video_xmax_score = df_prop.xmax_score.values[:]
            cache = (video_feature, video_xmin, video_xmax, video_xmin_score, video_xmax_score)
            #embed()
            #return video_feature, video_xmin, video_xmax, video_xmin_score, video_xmax_score
        return cache 
