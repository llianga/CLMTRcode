import os
import time
import logging
import pickle
import pandas as pd
import numpy as np
import torch
# from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import Dataset
from functools import partial
from tqdm import tqdm
import random, math

def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = float(lon) * 0.017453292519943295
    north = float(lat) * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))

def meters2lonlat(x, y):
    semimajoraxis = 6378137.0
    lon = x / semimajoraxis / 0.017453292519943295
    t = math.exp(y / 3189068.5)
    lat = math.asin((t - 1) / (t + 1)) / 0.017453292519943295
    return lon, lat

def downsampling(st_traj, dropping_rate=0.4):
    down_traj = []
    for i in range(len(st_traj)):
        if random.random() > dropping_rate:
            down_traj.append(st_traj[i])
    if len(down_traj) == 0:
        down_traj = st_traj[::2]
    return down_traj

def distort(traj, rate = 0.2, radius=50.0):
    noisetraj = []
    for i in range(len(traj)):  
        if np.random.rand() <= rate:
            x, y = lonlat2meters(traj[i][0], traj[i][1])
            xnoise, ynoise = 2 * np.random.rand() - 1, 2 * np.random.rand() - 1
            normz = np.hypot(xnoise, ynoise)
            xnoise, ynoise = xnoise * radius / normz, ynoise * radius / normz
            lon, lat = meters2lonlat(x + xnoise, y + ynoise)
            noisetraj.append([lon, lat, traj[i][2], traj[i][3]])
        else:
            noisetraj.append(traj[i])
    return noisetraj


def downsamplingDistort(traj):
    noisetrip1 = downsampling(traj)
    noisetrip2 = distort(noisetrip1)  
    return noisetrip2

# 1) read raw pd, 2) split into 3 partitions
def read_traj_dataset(file_path):
    logging.info('[Load traj dataset] START.')
    _time = time.time()
    data = pickle.load(open(file_path, 'rb'), encoding='bytes')
    trajs = data["ori_trajs"]

    l = trajs.shape[0]
    # print(l)
    train_idx = (int(l*0), 1000) #5000
    eval_idx = (int(l*0.7), int(l*0.725))
    test_idx = (int(l*0.8), int(l*1.0))
    
    trajs_eval = []
    for i in range(eval_idx[0], eval_idx[1]):#eval_idx[1]
        trajs_eval.append(trajs[i])
    trajs_test = []
    for i in range(test_idx[0], test_idx[1]):
        trajs_test.append(trajs[i])
    
    _train = TrajDataset(trajs[train_idx[0]: train_idx[1]])
    _eval = TrajDataset(trajs_eval)
    _test = TrajDataset(trajs_test)

    logging.info('[Load traj dataset] END. @={:.0f}, #={}' \
                .format(time.time() - _time, l))
    return _train, _eval, _test

def read_postraj_dataset(file_path):
    logging.info('[Load traj dataset] START.')
    _time = time.time()
    data = pickle.load(open(file_path, 'rb'), encoding='bytes')

    l = len(data)
    
    train_idx = (int(l*0), 1000) #5000
    eval_idx = (int(l*0.7), int(l*0.725))
    test_idx = (int(l*0.8), int(l*1.0))
    
    trajs_eval = []
    for i in range(eval_idx[0], eval_idx[1]):#eval_idx[1]
        trajs_eval.append(data[i])
    trajs_test = []
    for i in range(test_idx[0], test_idx[1]):
        trajs_test.append(data[i])
    
    _train = TrajDataset(data[train_idx[0]: train_idx[1]])
    _eval = TrajDataset(trajs_eval)
    _test = TrajDataset(trajs_test)
    
    logging.info('[Load traj dataset] END. @={:.0f}, #={}' \
                .format(time.time() - _time, l))
    return _train, _eval, _test
    

class TrajDataset(Dataset): 
    def __init__(self, data):
        # data: DataFrame
        self.data = data
          
    def __getitem__(self, i):
       #返回的必须是tensor/array/list等数据类型，整个数据集中的数据长度是相等的 
        return self.data[i]

    def __len__(self):
        return len(self.data)

def generate_newsimi_test_dataset(file_path, save_path, is_vali=True):
    data = pickle.load(open(file_path, 'rb'), encoding='bytes')
    trajs = data["ori_trajs"]
    l = trajs.shape[0]  
    if is_vali:
        test_idx = (int(l*0.7), int(l*0.8))
    else:
        test_idx = (int(l*0.8), int(l*1.0)) #8000
     # using test part only
    
    n_query = 1000
    
    test_trajs = trajs[test_idx[0]: test_idx[1]]
    logging.info("Test trajs loaded.")

    # for varying db size
    
    query_lst = [] # [N, len, 2]
    db_lst = []
    i = 0
    for _, v in test_trajs.iteritems():
        if i < n_query:
            query_lst.append(np.array(v)[::2].tolist())
        db_lst.append(np.array(v)[1::2].tolist())
        i += 1

    # output_file_name = Config.dataset_file + '_newsimi_raw.pkl'
    # with open(output_file_name, 'wb') as fh:
    pickle.dump((query_lst, db_lst), open(save_path, 'wb'), protocol=2)
    #     logging.info("_raw_dataset done.")
    return

def generate_knn_test_dataset(file_path, ori_query_path, distort_query_path, db_path, is_vali=True):
    data = pickle.load(open(file_path, 'rb'), encoding='bytes')
    trajs = data["ori_trajs"]
    l = trajs.shape[0]  
    if is_vali:
        test_idx = (int(l*0.7), int(l*0.8))
    else:
        test_idx = (int(l*0.8), int(l*1.0)) #8000
     # using test part only
    
    n_query = 1000
    
    test_trajs = trajs[test_idx[0]: test_idx[1]]
    logging.info("Test trajs loaded.")

    # for varying db size
    
    query_lst = [] # [N, len, 2]
    db_lst = []
    i = 0
    for _, v in test_trajs.iteritems():
        if i < n_query:
            query_lst.append(np.array(v)[::2].tolist())
            db_lst.append(np.array(v)[1::2].tolist())
        else:
            db_lst.append(np.array(v)[::2].tolist())
            db_lst.append(np.array(v)[1::2].tolist())
        i += 1
    d_query = []
    for i in range(len(query_lst)):
        query = downsamplingDistort(query_lst[i])
        d_query.append(query)
    
    pickle.dump(query_lst, open(ori_query_path, 'wb'), protocol=2)
    pickle.dump(d_query, open(distort_query_path, 'wb'), protocol=2)
    pickle.dump(db_lst, open(db_path, 'wb'), protocol=2)
    return


if __name__ == '__main__': 
    # max_len = 30
    # batch_size = 10
    # _train, _eval, _test = read_traj_dataset('/home/xianghao/code_laq/Cons_St2vec/data/simplified_st_tdrive_4')
    # generate_newsimi_test_dataset('/home/xianghao/code_laq/Cons_St2vec/data/simplified_st_tdrive_4','/home/xianghao/code_laq/Cons_St2vec/data/test_newsimi_raw')
    # dataset = os.path.join('data', 'poi_simplified_st_tdrive')
    dataset = '/home3/xianghao/code_laq/CL_ST2Vec/data/poi_simplified_st_tdrive'
    print(dataset)
    generate_knn_test_dataset(dataset, None, None, None, True)
    