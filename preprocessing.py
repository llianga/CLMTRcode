import math
from collections import defaultdict
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
import pickle
import numpy as np
import os
import h5py, logging, time
import random
import math
from config import Config, parse
from datetime import datetime
from semantictraj_preprocessing import Vocab
import Levenshtein


UNK = 3
eps = 1e-12

def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))

def meters2lonlat(x, y):
    semimajoraxis = 6378137.0
    lon = x / semimajoraxis / 0.017453292519943295
    t = math.exp(y / 3189068.5)
    lat = math.asin((t - 1) / (t + 1)) / 0.017453292519943295
    return lon, lat

def truncated_rand(mu = 0, sigma = 0.2, factor = 0.005, bound_lo = -0.004, bound_hi = 0.004):
    # using the defaults parameters, the success rate of one-pass random number generation is ~96%
    # gauss visualization: https://www.desmos.com/calculator/jxzs8fz9qr?lang=zh-CN
    while True:
        n = random.gauss(mu, sigma) * factor
        if bound_lo <= n <= bound_hi:
            break
    return n

def time_truncated_rand(mu = 0, sigma = 10, factor = 10, bound_lo = -15, bound_hi = 15):
    while True:
        n = random.gauss(mu, sigma) * factor
        if bound_lo <= n <= bound_hi:
            break
    return n

def makemid(x1,t1,x2,t2,t):
    if (t2-t1) * (x2-x1) == 0:
        return (x1+x2) / 2
    else:
        return x1 + (t-t1) / (t2-t1) * (x2-x1)

def distance(a, b):
    return  math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def point_line_distance(point, start, end):
    # point_t = point[2] #timestamp of point
    date_object = datetime.strptime(point[2], '%Y-%m-%d %H:%M:%S')
    point_t = date_object.timestamp()
    date_s = datetime.strptime(start[2], '%Y-%m-%d %H:%M:%S')
    start_t = date_s.timestamp()
    date_t = datetime.strptime(end[2], '%Y-%m-%d %H:%M:%S')
    end_t = date_t.timestamp()
    new_x = makemid(start[0], start_t, end[0], end_t, point_t)
    new_y = makemid(start[1], start_t, end[1], end_t, point_t)
    new_p = [new_x, new_y, point_t]
    dist = distance(new_p, point)
    return dist

def rdp(points, epsilon):
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1]) #points[i]是一个点，points相当于一条轨迹
        if d > dmax:
            index = i
            dmax = d

    if dmax >= epsilon :
        results = rdp(points[:index+1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]

    return results


def straight(st_traj):
    return st_traj


def rdpsimplify(st_traj):
    # src: [[lon, lat, t], [lon, lat, t], ...]
    traj_simp_dist = 0.000008
    return rdp(st_traj, epsilon = traj_simp_dist)


def subset(st_traj,traj_subset_ratio = 0.7): #continuous sub-trajectory
    l = len(st_traj)
    max_start_idx = l - int(l * traj_subset_ratio)
    start_idx = random.randint(0, max_start_idx) #产生[0,max_start_idx)的随机数
    end_idx = start_idx + int(l * traj_subset_ratio)
    return st_traj[start_idx: end_idx]

def downsampling(st_traj, dropping_rate=0.4):
    down_traj = []
    for i in range(len(st_traj)):
        if random.random() > dropping_rate:
            down_traj.append(st_traj[i])
    if len(down_traj) == 0:
        down_traj = st_traj[::2]
    return down_traj

def distort(traj, rate = 0.2, radius=50.0, time = 200.0):
    noisetraj = []
    Vocab = pickle.load(open('data/textvocab_st_tdrive', 'rb'), encoding='bytes')
    keywords_list = list(Vocab.word2vocab.keys())
    wordset = []
    # for words in keywords_list:
    for i in range(len(keywords_list)):
        words = keywords_list[i].split()
    
        for w in words:
            w = w.strip('()')
            wordset.append(w)
        
    for i in range(len(traj)):  
        if np.random.rand() <= rate:
            x, y = lonlat2meters(traj[i][0], traj[i][1])
            xnoise, ynoise = 2 * np.random.rand() - 1, 2 * np.random.rand() - 1
            normz = np.hypot(xnoise, ynoise)
            xnoise, ynoise = xnoise * radius / normz, ynoise * radius / normz
            lon, lat = meters2lonlat(x + xnoise, y + ynoise)
            noisetraj.append([lon, lat, traj[i][2], traj[i][3]])
        elif np.random.rand() > rate and np.random.rand() <= 0.4:
            date_object = datetime.strptime(traj[i][2], '%Y-%m-%d %H:%M:%S')
            timestamp = date_object.timestamp()
            tnoise = 2 * np.random.rand() - 1
            tnoise = tnoise * time
            dt_object = datetime.fromtimestamp(timestamp+tnoise)
            date_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')
            noisetraj.append([traj[i][0], traj[i][1], date_time, traj[i][3]])
        elif np.random.rand() > 0.4 and np.random.rand() <= 0.6:  
            words = traj[i][3].split() 
            if len(words) == 1:
                noisetraj.append(traj[i])
            else:
                length = len(words)
                ind = 0
                while ind >= length-1:
                    if np.random.rand() < 0.05 and len(words) > 1: 
                        words.pop(ind)        
                        traj[i][3] = ' '.join(words)
                        length = length -1
                    elif np.random.rand() >= 0.05 and np.random.rand() < 0.1 and len(words) > 1:
                        select_i = random.randint(0, len(wordset)-1)
                        words.insert(ind, wordset[select_i])
                        traj[i][3] = ' '.join(words)
                        length = length + 1
                    elif np.random.rand() >= 0.1 and np.random.rand() < 0.15 and len(words) > 1:
                        select_i = random.randint(0, len(wordset)-1)
                        words[ind] = wordset[select_i]
                        traj[i][3] = ' '.join(words)
                    else:
                        continue
                    ind = ind + 1
                noisetraj.append([traj[i][0], traj[i][1], traj[i][2], traj[i][3]])                 
        else:
            noisetraj.append(traj[i])
    return noisetraj

def ksimplify(trajseqs, timeseqs, textseqs):
    kseg = 10
    simp_traj, simp_times, simp_texts = [], [], []
    lons, lats = [], []
    for i in range(len(trajseqs)):
        tmp_lons, tmp_lats = [], []
        for j in range(len(trajseqs[i])):
            tmp_lons.append(trajseqs[i][j][0])
            tmp_lats.append(trajseqs[i][j][1])
        lons.append(tmp_lons)
        lats.append(tmp_lats)
          
    for i in range(len(trajseqs)):
        tmp_traj = []
        seg = len(trajseqs[i]) // kseg
        for k in range(kseg):
            if k == kseg-1:
               tmp_traj.append([np.mean(lons[i][k * seg:]), np.mean(lats[i][k * seg:])]) 
            else:
                tmp_traj.append([np.mean(lons[i][k * seg:k * seg + seg]), np.mean(lats[i][k * seg:k * seg + seg])])
        simp_traj.append(tmp_traj)  
    simp_traj = np.array(simp_traj)
    simp_traj = simp_traj.reshape(-1, kseg*2)
    
    for i in range(len(timeseqs)):
        tmp_time = []
        seg = len(timeseqs[i]) // kseg
        for k in range(kseg):
            if k == kseg-1:
               tmp_time.append(np.mean(timeseqs[i][k * seg:])) 
            else:
                tmp_time.append(np.mean(timeseqs[i][k * seg:k * seg + seg]))
        simp_times.append(tmp_time)  
    simp_times = np.array(simp_times)
    
    for i in range(len(textseqs)):
        tmp_text = []
        seg = len(textseqs[i]) // kseg
        for k in range(kseg):
            if k == kseg-1:
               tmp_text.append(np.mean(textseqs[i][k * seg:])) 
            else:
                tmp_text.append(np.mean(textseqs[i][k * seg:k * seg + seg]))
        simp_texts.append(tmp_text)  
    simp_texts = np.array(simp_texts)
    
    print("the first simptraj: ", simp_traj[0])
    print(simp_times[0])
    print(simp_texts[0])
    print(simp_traj.shape)
    print(simp_times.shape)
    print(simp_texts.shape)
    # kseg_trajs = np.concatenate((simp_traj, simp_times, simp_texts), axis=1)
    st_kseg_trajs = np.concatenate((simp_traj, simp_times), axis=1)
    return st_kseg_trajs #kseg_trajs

def traj_mdl_comp(points, start_index, curr_index, typed):
    length = distance(points[start_index], points[curr_index])
    h = 0
    lh = 0
    if typed == 'simp':
        if length > eps:
            h = math.log2(length)
    for i in range(start_index, curr_index, 1):
        if typed == 'simp':
            t = points[i][2]
            new_x = makemid(points[start_index][0], points[start_index][2], points[curr_index][0], points[curr_index][2], points[i][2])
            new_y = makemid(points[start_index][1], points[start_index][2], points[curr_index][1], points[curr_index][2], points[i][2])
            new_p = [new_x, new_y, t]
            lh += distance(points[i], new_p)
        elif typed == 'orign':
            d = distance(points[i], points[i+1])
            # h += d
            if d > eps:
                h += math.log2(d)
    if typed == 'simp':
        if lh > eps:
            h += math.log2(lh)
        return h
    else:
        return h


def mdlsimplify(st_traj):
    simp_traj = []
    start_index = 0
    length = 1
    simp_traj.append(st_traj[start_index])
    while start_index + length < len(st_traj):
        curr_index = start_index + length
        cost_simp = traj_mdl_comp(st_traj, start_index, curr_index, 'simp')
        cost_origin = traj_mdl_comp(st_traj, start_index, curr_index, 'orign')
        if cost_simp > cost_origin:
            simp_traj.append(st_traj[curr_index])
            start_index = curr_index
            length = 1
        else:
            length += 1
    if not (simp_traj[-1][0] == st_traj[-1][0] and simp_traj[-1][1] == st_traj[-1][1] and simp_traj[-1][2] == simp_traj[-1][2]):
        simp_traj.append(st_traj[-1])
    return simp_traj

def get_aug_fn(name: str):
    return {'straight': straight, 'rdpsimplify': rdpsimplify, 'distort': distort,
            'ksimplify': ksimplify, 'subset': subset, 'downsampling': downsampling, 'mdlsimplify': mdlsimplify}.get(name, None)


def time_frechet_dis(list_a = [], list_b = []):
    tr1, tr2 = np.array(list_a), np.array(list_b)
    M, N = len(tr1), len(tr2)
    c = np.zeros((M + 1, N + 1))
    c[0, 0] = abs(tr1[0]-tr2[0])
    for i in range(1, M):
        temp = abs(tr1[i]-tr2[0])
        if temp > c[i - 1][0]:
            c[i][0] = temp
        else:
            c[i][0] = c[i - 1][0]
    for i in range(1, N):
        temp = abs(tr2[i]-tr1[0])
        if temp > c[0][i - 1]:
            c[0][i] = temp
        else:
            c[0][i] = c[0][i - 1]
    for i in range(1, M):
        for j in range(1, N):
            c[i, j] = max(abs(tr1[i]-tr2[j]), min(c[i - 1][j - 1], c[i - 1][j], c[i][j - 1]))

    return int(c[M - 1, N - 1])

def spatial_frechet_dis(list_a = [], list_b = []):
    tr1, tr2 = np.array(list_a), np.array(list_b)
    M, N = len(tr1), len(tr2)
    c = np.zeros((M + 1, N + 1))  
    c[0, 0] = np.linalg.norm(tr1[0]-tr2[0])
    for i in range(1, M):
        temp = np.linalg.norm(tr1[i]-tr2[0])
        if temp > c[i - 1][0]:
            c[i][0] = temp
        else:
            c[i][0] = c[i - 1][0]
    for i in range(1, N):
        temp = np.linalg.norm(tr2[i]-tr1[0])
        if temp > c[0][i - 1]:
            c[0][i] = temp
        else:
            c[0][i] = c[0][i - 1]
    for i in range(1, M):
        for j in range(1, N):
            c[i, j] = max(np.linalg.norm(tr1[i]-tr2[j]), min(c[i - 1][j - 1], c[i - 1][j], c[i][j - 1]))

    return c[M - 1, N - 1]

# def edit_dis(tr1, tr2):
#     M, N = len(tr1), len(tr2)
#     c = np.zeros((M + 1, N + 1))
#     dist0 = Levenshtein.distance(tr1[0], tr2[0])
#     c[0, 0] = abs(dist0)
#     for i in range(1, M):
#         distm =  Levenshtein.distance(tr1[i], tr2[0])
#         temp = abs(distm)
#         if temp > c[i - 1][0]:
#             c[i][0] = temp
#         else:
#             c[i][0] = c[i - 1][0]
#     for i in range(1, N):
#         distn = Levenshtein.distance(tr2[i], tr1[0])
#         temp = abs(distn)
#         if temp > c[0][i - 1]:
#             c[0][i] = temp
#         else:
#             c[0][i] = c[0][i - 1]
#     for i in range(1, M):
#         for j in range(1, N):
#             distj = Levenshtein.distance(tr1[i], tr2[j])
#             c[i, j] = max(abs(distj), min(c[i - 1][j - 1], c[i - 1][j], c[i][j - 1]))

#     return int(c[M - 1, N - 1])


def getsimpsttraj(data):
    trajdata = data["spatial_seqs"]
    timedata = data["time_seqs"]
    textdata = data["keyword_seqs"]
    new_timedata, new_textdata = [], []
    
    Vocab = pickle.load(open('data/textvocab_st_tdrive', 'rb'), encoding='bytes')
     
    
    for i in range(len(timedata)):
        tmp_time = []
        for j in range(len(timedata[i])):
            timevalue = timedata[i][j]
            date_object = datetime.strptime(timevalue, '%Y-%m-%d %H:%M:%S')
            timestamp = date_object.timestamp()
            tmp_time.append(int(timestamp))
        new_timedata.append(tmp_time) 
    
    for i in range(len(textdata)):
        tmp_text = []
        for j in range(len(textdata[i])):
            vocab_id = Vocab.word2vocab[textdata[i][j]] 
            tmp_text.append(vocab_id)
        new_textdata.append(tmp_text)
    kseg_trajs = ksimplify(trajdata, new_timedata, new_textdata)      
    # pickle.dump(kseg_trajs, open("data/simptrajs", 'wb'), protocol=2)
    pickle.dump(kseg_trajs, open("data/st_simptrajs", 'wb'), protocol=2)
    
def getneighbor(trajs, data, trajdata, k=3):
    spatialdata, timedata, textdata = data["spatial_seqs"], data["time_seqs"], data["keyword_seqs"]
    timestampdata = []
    for i in range(len(timedata)):
        tmp = []
        for j in range(len(timedata[i])):
            date_object = datetime.strptime(timedata[i][j], '%Y-%m-%d %H:%M:%S')
            timestamp = date_object.timestamp() 
            tmp.append(timestamp)
        timestampdata.append(tmp)
            
    # ball_tree = BallTree(trajs)
    kd_tree = KDTree(trajs, leaf_size=2)
    positivetrajs = []
    knnid_list = []
    for i in range(len(trajs)):
        # tmp_trajpair = []
        ori_spatial_seq, ori_time_seq, ori_text_seq = spatialdata[i], timestampdata[i], textdata[i]
        # dist, index = ball_tree.query([trajs[i]], k) 
        dist, index = kd_tree.query([trajs[i]], k) 
        min_dist = 10000000
        min_id = i
        if i in knnid_list:
            continue
        print(i)
        # knnid_list.append(index[0])
        # print(index[0])
        for ind in index[0]:
            # if i==ind:
            #     continue
            spatial_seq, time_seq, text_seq = spatialdata[ind], timestampdata[ind], textdata[ind]
            # print('spatial_seqs: ', ori_spatial_seq)
            # print(spatial_seq)
            sp_dist = spatial_frechet_dis(spatial_seq, ori_spatial_seq)
            time_dist = time_frechet_dis(time_seq, ori_time_seq)
            text_dist = Levenshtein.distance(ori_text_seq, text_seq)
            # text_dist = edit_dis(ori_text_seq, text_seq)
            # print('distances: ', sp_dist, time_dist, text_dist)
            total_dist = 10*sp_dist + 0.00005* time_dist + 0.001*text_dist
            # print("---total_dist---: ", total_dist)
            if total_dist < min_dist:
                min_id = ind
                min_dist = total_dist
        # print("i and min_id: ", i, min_id)
        knnid_list.append(min_id)
        positivetrajs.append([trajdata[i], trajdata[min_id]])
         
        # positivetrajs.append(tmp_trajpair)   
        # print("the nearest id: ", min_id)
    # print(len(positivetrajs[0]))
    # print(len(positivetrajs[0][0]), len(positivetrajs[0][1]))
    # print("knn id list: ", knnid_list)
    # pickle.dump(positivetrajs, open("data/vali_positivetrajs", 'wb'), protocol=2)
    # pickle.dump(knnid_list, open("data/vali_knnid_list", 'wb'), protocol=2)
    return positivetrajs

def get_augmented_postrajs(positivetrajs):
    aug1 = get_aug_fn('downsampling')
    aug2 = get_aug_fn('distort')
    aug_trajs = []
    for i in range(len(positivetrajs)):
        traj1 = aug1(positivetrajs[i][0])
        traj2 = aug2(positivetrajs[i][1])
        # print(len(traj2))
        aug_trajs.append([traj1, traj2])
    return aug_trajs
       
        
if __name__ == '__main__':
    args = parse.get_args()
    Config.merge_from_file(args.config_file) 
    trajs_file = os.path.join('data', Config.DATASETS.dataset)
    data = pickle.load(open(trajs_file, 'rb'), encoding='bytes')
    # getsimpsttraj(data)
    trajdata = data["ori_trajs"]
    
    simptrajs = pickle.load(open("data/st_simptrajs", 'rb'), encoding='bytes')
    # print(len(simptrajs))
    simptrajsdata = simptrajs[:10000]
    # print(len(simptrajsdata))
    positivetrajs = getneighbor(simptrajsdata, data, trajdata)
    aug_trajs = get_augmented_postrajs(positivetrajs)
    print(len(aug_trajs))
    print(len(aug_trajs[0]))
    pickle.dump(aug_trajs, open("data/augmented_trajs_v3", 'wb'), protocol=2)
    
    
    
    
    


