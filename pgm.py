#!/usr/bin/env python4
# -*- coding: utf-8 -*-
"""
Created on 2019/04/15
author: lujie
"""

import json
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
from scipy.interpolate import interp1d
from IPython import embed


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    ''' Compute intersection between score a box and the anchors '''

    len_anchors = anchors_max-anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    ''' Compute jaccard score between a box and the anchors '''

    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def generateProposals(opt, video_list, video_dict):
    '''
    step - 1. choose the peak point for start, end point
    step - 2. cal the iou and ioa for each proposal
    '''

    tscale = opt["temporal_scale"]   # 100
    peak_thres = opt["pgm_threshold"] # 0.5
    tgap   = 1./tscale

    for video_name in video_list:

        pdf = pd.read_csv("./output/TEM_results/"+video_name+".csv")

        start_scores, end_scores = pdf.start.values[:], pdf.end.values[:]
        max_start, max_end = max(start_scores), max(end_scores)

        start_bins = np.zeros(len(start_scores))
        start_bins[[0,-1]] = 1  # init the start and end element
        for idx in range(1, tscale-1):
            if (start_scores[idx] > start_scores[idx+1]) and (start_scores[idx] > start_scores[idx-1]):
                start_bins[idx] = 1
            elif start_scores[idx] > (peak_thres * max_start):
                start_bins[idx] = 1

        end_bins = np.zeros(len(end_scores))
        end_bins[[0,-1]] = 1
        for idx in range(1, tscale-1):
            if (end_scores[idx] > end_scores[idx+1]) and (end_scores[idx] > end_scores[idx-1]):
                end_bins[idx] = 1
            elif end_scores[idx]>(peak_thres*max_end):
                end_bins[idx] = 1

        xmin_list, xmin_score_list = [], []
        xmax_list, xmax_score_list = [], []
        for j in range(tscale):
            if start_bins[j] == 1:
                xmin_list.append(tgap / 2 + tgap * j)
                xmin_score_list.append(start_scores[j])
            if end_bins[j] == 1:
                xmax_list.append(tgap / 2 + tgap * j)
                xmax_score_list.append(end_scores[j])

        new_props = []
        for i in range(len(xmax_list)):

            tmp_xmax, tmp_xmax_score = xmax_list[i], xmax_score_list[i]
            for j in range(len(xmin_list)):

                tmp_xmin, tmp_xmin_score = xmin_list[j], xmin_score_list[j]
                if tmp_xmin >= tmp_xmax:
                    break
                new_props.append([tmp_xmin, tmp_xmax, tmp_xmin_score, tmp_xmax_score])

        new_props = np.stack(new_props)
        col_name  = ['xmin', 'xmax', 'xmin_score', 'xmax_score']
        new_df = pd.DataFrame(new_props, columns=col_name)
        new_df['score'] = new_df.xmin_score * new_df.xmax_score
        new_df = new_df.sort_values(by='score', ascending=False)  # decending

        video_info    = video_dict[video_name]
        video_frame   = video_info['duration_frame']
        video_second  = video_info['duration_second']
        feature_frame = video_info['feature_frame']
        refine_second = float(feature_frame) / video_frame * video_second

        
        # this part just for the training of PEM
        # train and validation set has anno_info, test-set has no anno_info
        try:
            gt_xmins, gt_xmaxs = [], []
            for idx in range(len(video_info['annotations'])):
                gt_xmins.append(video_info['annotations'][idx]['segment'][0]/refine_second)
                gt_xmaxs.append(video_info['annotations'][idx]['segment'][1]/refine_second)

            new_iou_list = []
            for j in range(len(new_df)):
                tmp_new_iou = max(iou_with_anchors(new_df.xmin.values[j], new_df.xmax.values[j], gt_xmins, gt_xmaxs))
                new_iou_list.append(tmp_new_iou)
            
            new_ioa_list = []
            for j in range(len(new_df)):
                tmp_new_ioa = max(ioa_with_anchors(new_df.xmin.values[j], new_df.xmax.values[j], gt_xmins, gt_xmaxs))
                new_ioa_list.append(tmp_new_ioa)

            new_df['match_iou'], new_df['match_ioa'] = new_iou_list, new_ioa_list

        except Exception as e:
            # just for the case of test-set
            new_df['match_iou'], new_df['match_ioa'] = None, None
            print(e)
        finally: 
            new_df.to_csv("./output/PGM_proposals/"+video_name+".csv",index=False)


def PGM_proposal_generation(opt):
    ''' Proposal generation module '''

    video_dict = load_json(opt["video_anno"])

    video_list = list(video_dict.keys())
    print('len_video_list : ', len(video_list))
    num_videos = len(video_list)
    
    # single-process
    # generateProposals(opt, video_list, video_dict)
    
    num_videos_per_thread = num_videos // opt["pgm_thread"]
    processes = []
    for tid in range(opt["pgm_thread"]-1):
        tmp_video_list = video_list[(tid*num_videos_per_thread):((tid+1)*num_videos_per_thread)]   # BUG
        p = mp.Process(target=generateProposals, args=(opt,tmp_video_list,video_dict))
        p.start()
        processes.append(p)

    tmp_video_list = video_list[(opt["pgm_thread"]-1)*num_videos_per_thread:]
    p = mp.Process(target=generateProposals,args =(opt,tmp_video_list,video_dict))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()
    

def generateFeature(opt, video_list, video_dict):
    '''
    step - 1. load the temporal-estimate module result for each video
    step - 2. prepare the uniform-intervals for each video 
    step - 3. generate the bsp-fea based interp1d on uniform-intervals for each proposal
    step - 4. 
    '''
    num_sample_start    = opt["num_sample_start"]    # 8
    num_sample_end      = opt["num_sample_end"]      # 8
    num_sample_action   = opt["num_sample_action"]   # 16
    num_sample_interpld = opt["num_sample_interpld"] # 3

    for video_name in video_list:
        
        # step - 1
        adf = pd.read_csv("./output/TEM_results/" + video_name + ".csv")   # pdf of three-curves[action, start, end] of all snippets
        score_action = adf.action.values[:]
        video_len = len(adf)                     # video_scale
        video_extend = video_len // 4 + 10       # TODO

        pdf = pd.read_csv("./output/PGM_proposals/" + video_name + ".csv") # [start, end, start_prob, end_prob, prob], where prob = start_prob * end_prob
        video_subset = video_dict[video_name]['subset']

        if video_subset == "training":
            pdf = pdf[:opt["pem_top_K"]]  # 500
        else:
            pdf = pdf[:opt["pem_top_K_inference"]] # 1000
        
        # step - 2
        tmp_zeros = np.zeros([video_extend])
        score_action = np.concatenate((tmp_zeros, score_action, tmp_zeros))  # default : [35, 100, 35]
        seg_xmins, seg_xmaxs = adf.xmin.values[:], adf.xmax.values[:]
        video_gap = seg_xmaxs[0] - seg_xmins[0]     # 1 / video_scale
        part_a = [-video_gap / 2 - (video_extend-1-i) * video_gap for i in range(video_extend)]
        part_b = [video_gap / 2 + i * video_gap for i in range(video_len)]
        part_c = [video_gap / 2 + seg_xmaxs[-1] + i * video_gap for i in range(video_extend)]
        tmp_x  =  part_a + part_b + part_c
        f_action = interp1d(tmp_x, score_action, axis=0)    # linear-interpolation
        
        # step - 3
        feature_bsp = []
        for idx in range(len(pdf)):

            xmin = pdf.xmin.values[idx]
            xmax = pdf.xmax.values[idx]
            xlen = xmax - xmin
            xmin_0 = xmin - xlen * opt["bsp_boundary_ratio"]   # default 0.2
            xmin_1 = xmin + xlen * opt["bsp_boundary_ratio"]   
            xmax_0 = xmax - xlen * opt["bsp_boundary_ratio"]
            xmax_1 = xmax + xlen * opt["bsp_boundary_ratio"]

            #start
            plen_start = (xmin_1 - xmin_0) / (num_sample_start - 1)
            plen_sample = plen_start / num_sample_interpld
            tmp_x_new = [ xmin_0 - plen_start / 2 + plen_sample * i for i in range(num_sample_start * num_sample_interpld + 1)]
            tmp_y_new_start_action = f_action(tmp_x_new)
            tmp_y_new_start = [np.mean(tmp_y_new_start_action[i*num_sample_interpld:(i+1)*num_sample_interpld+1]) for i in range(num_sample_start)]

            #end
            plen_end= (xmax_1-xmax_0)/(num_sample_end-1)
            plen_sample = plen_end / num_sample_interpld
            tmp_x_new = [ xmax_0 - plen_end/2 + plen_sample * i for i in range(num_sample_end*num_sample_interpld +1 )]
            tmp_y_new_end_action = f_action(tmp_x_new)
            tmp_y_new_end = [np.mean(tmp_y_new_end_action[i*num_sample_interpld:(i+1)*num_sample_interpld+1]) for i in range(num_sample_end)]

            #action
            plen_action= (xmax-xmin)/(num_sample_action-1)
            plen_sample = plen_action / num_sample_interpld
            tmp_x_new = [ xmin - plen_action/2 + plen_sample * i for i in range(num_sample_action*num_sample_interpld +1 )]
            tmp_y_new_action = f_action(tmp_x_new)
            tmp_y_new_action = [np.mean(tmp_y_new_action[i*num_sample_interpld:(i+1)*num_sample_interpld+1]) for i in range(num_sample_action)]
            
            # concate the bsp-fea
            tmp_feature = np.concatenate([tmp_y_new_action, tmp_y_new_start, tmp_y_new_end])
            feature_bsp.append(tmp_feature)

        feature_bsp = np.array(feature_bsp)
        np.save("./output/PGM_feature/"+video_name, feature_bsp)


def getDatasetDict(opt):
    ''' Prepare the data-dict for feature of proposal '''

    df = pd.read_csv(opt["video_info"])
    anno_info = load_json(opt["video_anno"])

    video_dict = {}
    for i in range(len(df)):
        video_name = df.video.values[i]
        video_info = anno_info[video_name]
        video_new_info = {}
        video_new_info['duration_frame'] = video_info['duration_frame']
        video_new_info['duration_second'] = video_info['duration_second']
        video_new_info["feature_frame"] = video_info['feature_frame']
        video_new_info['annotations'] = video_info['annotations']
        video_new_info['subset'] = df.subset.values[i]
        video_dict[video_name] = video_new_info
    return video_dict


def PGM_feature_generation(opt):
    ''' Generate features for each proposal with multiprocessing '''

    video_dict = getDatasetDict(opt)
    video_list = list(video_dict.keys())
    num_videos_per_thread = len(video_list) // opt["pgm_thread"]
    processes = []
    for tid in range(opt["pgm_thread"]-1):
        tmp_video_list = video_list[tid*num_videos_per_thread:(tid+1)*num_videos_per_thread]
        p = mp.Process(target=generateFeature, args=(opt,tmp_video_list,video_dict))
        p.start()
        processes.append(p)

    tmp_video_list = video_list[(opt["pgm_thread"]-1)*num_videos_per_thread:]
    p = mp.Process(target = generateFeature, args =(opt, tmp_video_list, video_dict))
    p.start()
    processes.append(p)

    for p in processes:
        p.join()
