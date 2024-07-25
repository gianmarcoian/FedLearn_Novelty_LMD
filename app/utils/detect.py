import argparse
import lpips
from sklearn.metrics import roc_auc_score
import pickle
import glob
import torch
import torchvision
import torch.nn.functional as F
import tqdm
import numpy as np
import os
lpips_criterion = lpips.LPIPS(net='alex', pretrained=True, lpips=True).cuda()




def simclr_resize(x):
    result = F.interpolate(x, 224, mode="bicubic".split('_')[0])
    return result


def lpips_scaler(x):
    return x * 2. - 1.

def calculate_metric(orig, recon, metric):
    return lpips_criterion(lpips_scaler(orig.cuda()), lpips_scaler(recon.cuda())).flatten().detach().cpu().numpy()

def detect():
    reps=10
    metric='LPIPS'

    pos_files = glob.glob( 'image_in.pth')
    neg_files = glob.glob('image_out.pth')

    all_pos_eval = np.zeros((len(pos_files), reps))
    all_neg_eval = np.zeros((len(neg_files), reps))



    for idx, file in enumerate(tqdm.tqdm(pos_files, desc="Processing positive images")):
       data = torch.load(file)
       orig = data['orig']
       for r in range(reps):
           recon = data['recon'][r]
           all_pos_eval[idx, r] = calculate_metric(orig, recon, metric)

    for idx, file in enumerate(tqdm.tqdm(neg_files, desc="Processing negative images")):
       data = torch.load(file)
       orig = data['orig']
       for r in range(reps):
           recon = data['recon'][r]
           all_neg_eval[idx, r] = calculate_metric(orig, recon, metric)
    
    agg_fn = np.median
    all_pos_s = agg_fn(all_pos_eval, axis=1)
    all_neg_s = agg_fn(all_neg_eval, axis=1)

    results = np.append(all_pos_s, all_neg_s)
    return  all_neg_s[0]>all_pos_s[0]
    
