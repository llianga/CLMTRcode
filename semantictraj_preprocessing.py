import math
from collections import defaultdict
from sklearn.neighbors import KDTree
import pickle
import numpy as np
import os
import h5py, logging, time, json
import random
from config import Config, parse
import torch
import torch.backends.cudnn
from node2vec import train_node2vec
from d2vecModel import train_d2vec
import datetime
from bertmodel import get_bert_embeddings, train_text2Vec
from data_utils.data_loader import generate_newsimi_test_dataset, generate_knn_test_dataset

UNK = 3
eps = 1e-12

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

class Grid():
    def __init__(self, maxlon, minlon, maxlat, minlat, minfreq, xstep, ystep, maxvocab_size, vocab_start, k, name):
        self.maxlon = maxlon
        self.minlon = minlon
        self.maxlat = maxlat
        self.minlat = minlat
        self.minfreq = minfreq
        self.xstep = xstep
        self.ystep = ystep
        self.maxvocab_size = maxvocab_size
        self.vocab_start = vocab_start
        self.k = k
        self.name = name
        self.minx, self.miny = lonlat2meters(self.minlon, self.minlat)
        self.maxx, self.maxy = lonlat2meters(self.maxlon, self.maxlat)
        numx = round(self.maxx - self.minx) / self.xstep
        self.numx = int(math.ceil(numx))
        numy = round(self.maxy - self.miny) / self.ystep
        self.numy = int(math.ceil(numy))
    
    
    def coord2cell(self, x, y):
        xoffset = round(x - self.minx) / self.xstep
        yoffset = round(y - self.miny) / self.ystep
        xoffset = int(math.floor(xoffset))
        yoffset = int(math.floor(yoffset))
        cell_id = yoffset * self.numx + xoffset
        return cell_id

    
    def cell2coord(self, cell_id):
        yoffset = cell_id // self.numx
        xoffset = cell_id % self.numx
        y = self.miny + (yoffset + 0.5) * self.ystep
        x = self.minx + (xoffset + 0.5) * self.xstep
        return x, y

    
    def gps2cell(self, lon, lat):
        x, y = lonlat2meters(lon, lat)
        cell_id = self.coord2cell(x, y)
        return cell_id
    
    def cell2gps(self, cell_id):
        x, y = self.cell2coord(cell_id)
        lon, lat = meters2lonlat(x, y)
        return lon, lat
    
    def gps2offset(self, lon, lat):
        x, y = lonlat2meters(lon, lat)
        xoffset = round(x - self.minx) / self.xstep
        yoffset = round(y - self.miny) / self.ystep
        return xoffset, yoffset
    
    def coordingrid(self, lon, lat):
        if float(lon) >= self.minlon and float(lon) <= self.maxlon and float(lat) >= self.minlat and float(lat) <= self.maxlat:
            return True
        else:
            return False 
    
    
    def trajingrid(self, traj):
        for i in range(len(traj)):
            if not self.coordingrid(traj[i][0], traj[i][1]):
                return False
        return True

   
    def makeVocab(self, trajdata):
        self.cellcount = defaultdict(list)
        num_out_region = 0
        for i in range(len(trajdata)):
            for j in range(len(trajdata[i])):
                lon, lat = trajdata[i][j][0], trajdata[i][j][1]
                if not self.coordingrid(lon, lat):
                    num_out_region += 1
                else:
                    cell_id = self.gps2cell(lon, lat)
                    if not self.cellcount[cell_id]:
                        self.cellcount[cell_id] = 1
                    else:
                        self.cellcount[cell_id] += 1 
        self.max_num_hotcells = min(self.maxvocab_size, len(self.cellcount))
        self.topcellcount = dict(sorted(self.cellcount.items(), key=lambda d: d[1], reverse=True))
        self.hotcell = []
        for key in self.topcellcount.keys():
            if self.topcellcount[key] >= self.minfreq:
                self.hotcell.append(key)
        self.hotcell2vocab = defaultdict(list)
        for (i, cell) in enumerate(self.hotcell):
            self.hotcell2vocab[cell] = i+self.vocab_start
        self.vocab2hotcell = {value:key for key,value in self.hotcell2vocab.items()}
        self.vocab_size = self.vocab_start + len(self.hotcell)
        coord = []
        for cell in self.hotcell:
            x, y = self.cell2coord(cell)
            coord.append([x,y])
        self.hotcell_kdtree = KDTree(coord, leaf_size=2) 
        self.built = True

    def knearestHotcells(self, cell, k):
        coord = []
        x, y = self.cell2coord(cell)
        coord.append([x,y])
        dists, idxs = self.hotcell_kdtree.query(coord, k)
        kcells = []
        for i in range(len(idxs[0])):
            kcells.append(self.hotcell[idxs[0][i]])       
        return kcells, dists
    
    def nearestHotcell(self, cell): 
        kcells, _ = self.knearestHotcells(cell, 1)
        return kcells[0]
    
    
    def saveKNearestVocabs(self):
        V = np.zeros((self.vocab_size, self.k))
        D = np.zeros((self.vocab_size, self.k))
        for vocab in range(self.vocab_start):
            V[vocab, :] = vocab
            D[vocab, :] = 0.0
        for vocab in range(self.vocab_start, self.vocab_size):
            cell = self.vocab2hotcell[vocab]
            kcells, dists = self.knearestHotcells(cell, self.k)
            kvocabs = []
            for i in range(len(kcells)):
                kvocabs.append(self.hotcell2vocab[kcells[i]])
            V[vocab, :] = kvocabs
            D[vocab, :] = dists
        cellsize = int(self.xstep)
        filename = os.path.join('data', "{}-minfreq-{}-vocab-dist-cell{}.h5".format(self.name,self.minfreq,cellsize))
        f = h5py.File(filename,'w')
        f["V"], f["D"] = V, D
        f.close()
    
    def cell2vocab(self, cell):
        if self.hotcell2vocab[cell]:
            return self.hotcell2vocab[cell]
        else:
            hotcell = self.nearestHotcell(cell)
            return self.hotcell2vocab[hotcell]
    
    def gps2vocab(self, lon, lat):
        if not self.coordingrid(lon, lat):
            return UNK
        cell_id = self.gps2cell(lon, lat)
        vocab_id = self.cell2vocab(cell_id)
        return vocab_id
    
    def traj2seq(self, traj):
        seq = []
        for i in range(len(traj)):
            lon, lat = traj[i][0], traj[i][1]
            vocab_id = self.gps2vocab(lon, lat)
            seq.append(vocab_id)
        return seq
    
    def seq2traj(self, seq):
        traj = []
        for i in range(len(seq)):
            if self.vocab2hotcell[seq[i]]:
                cell_id = self.vocab2hotcell[seq[i]]
            else:
                print("{} is out of vocabulary".format(seq[i]))
                cell_id = -1
            lon, lat = self.cell2gps(cell_id)
            traj.append([lon, lat])
        return traj

    def trajmeta(self, traj):
        xs = []
        ys = []
        for i in range(len(traj)):
            xs.append(traj[i][0])
            ys.append(traj[i][1])
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        x_mid = x_min + (x_max - x_min) / 2
        y_mid = y_min + (y_max - y_min)/2
        xoffset, yoffset = self.gps2offset(x_mid, y_mid)
        return xoffset, yoffset
    
    def get_cellid_by_xyidx(self, i_x: int, i_y: int):
        return i_x * self.numy + i_y
    
    def k_neighbours_cell_pairs_permutated(self): 
        
        all_cell_knnpairs = []
        all_vocab_knnpairs_id = []

        for vocab in range(self.vocab_start, self.vocab_size):
            cell = self.vocab2hotcell[vocab]
            kcells, dists = self.knearestHotcells(cell, 10)
            kvocabs = []
            for i in range(len(kcells)):
                kvocab = self.hotcell2vocab[kcells[i]]
                if kvocab == vocab:
                    continue
                all_cell_knnpairs.append((cell, kcells[i])) 
                all_vocab_knnpairs_id.append((vocab, kvocab))
        return all_cell_knnpairs, all_vocab_knnpairs_id

def init_grid(_device,trajdata):
    with open('data_utils/tdrive.json', 'r') as traj_param_file:
        traj_params = json.load(traj_param_file)
    maxlon = traj_params["poi_tdrive_v1"]["maxlon"]
    minlon = traj_params["poi_tdrive_v1"]["minlon"]
    maxlat = traj_params["poi_tdrive_v1"]["maxlat"]
    minlat = traj_params["poi_tdrive_v1"]["minlat"]
    minfreq = traj_params["poi_tdrive_v1"]["minfreq"]
    xstep = traj_params["poi_tdrive_v1"]["xstep"]
    ystep = traj_params["poi_tdrive_v1"]["ystep"]
    maxvocab_size = traj_params["poi_tdrive_v1"]["maxvocab_size"]
    vocab_start =  traj_params["poi_tdrive_v1"]["vocab_start"]
    k =  traj_params["poi_tdrive_v1"]["k"]
    name = traj_params["poi_tdrive_v1"]["cityname"]
    grid = Grid(maxlon, minlon, maxlat, minlat, minfreq, xstep, ystep, maxvocab_size, vocab_start, k, name)
    
    grid.makeVocab(trajdata) 
    pickle.dump(grid, open('data/poi_grid_cellsize_100_minfreq_5', 'wb'), protocol=2)
    grid = pickle.load(open('data/poi_grid_cellsize_100_minfreq_5', 'rb'), encoding='bytes')
    _, all_vocab_knnpairs_id = grid.k_neighbours_cell_pairs_permutated()
    all_vocab_knnpairs_id.append((0,0))
    all_vocab_knnpairs_id.append((1,1))
    all_vocab_knnpairs_id.append((2,2))
    all_vocab_knnpairs_id.append((3,3))
    edge_index = torch.tensor(all_vocab_knnpairs_id, dtype = torch.long, device = _device).T
    train_node2vec(edge_index)
    return

def transtimes(timedata):
    times = []
    for i in range(len(timedata)): 
        for j in range(len(timedata[i])):
            timestamp = timedata[i][j]
            date_obj = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            t = datetime.datetime.fromtimestamp(date_obj.timestamp())
            t = [t.hour, t.minute, t.second, t.year, t.month, t.day]
            times.append(t)
    return times

def getkeywords(textdata):
    keywords = []
    for i in range(len(textdata)):
        for j in range(len(textdata[i])):
            keywords.append(textdata[i][j])
    keywords = list(set(keywords))
    return keywords

class Interval():
    def __init__(self, mintime, maxtime, mintimefreq, timestep, maxvocab_size, vocab_start, k, name):
        self.mintime = mintime
        self.maxtime = maxtime
        self.mintimefreq = mintimefreq
        self.timestep = timestep
        self.maxvocab_size = maxvocab_size
        self.vocab_start = vocab_start
        self.k = k
        self.name = name
        num = round(self.maxtime - self.mintime) / self.timestep
        self.num = int(math.ceil(num))
    
    def time2cell(self, t):
        toffset = round(t - self.mintime) / self.timestep
        toffset = int(math.floor(toffset))
        return toffset
    
    def cell2time(self, cell_id):  
        t = self.mintime + cell_id * self.timestep + 0.5 * self.timestep
        return t
    
    def timeinterval(self, t):
        if t >= self.mintime and t <= self.maxtime:
            return True
        else:
            return False 
        
    def timeseqinterval(self, timeseq):
        for i in range(len(timeseq)):
            if not self.timeinterval(timeseq[i]):
                return False
        return True
    
    def makeVocab(self, timedata):
        self.cellcount = defaultdict(list)
        num_out_time = 0
        for i in range(len(timedata)):
            t = timedata[i]
            if not self.timeinterval(t):
                num_out_time += 1
            else:
                cell_id = self.time2cell(t)
                if not self.cellcount[cell_id]:
                    self.cellcount[cell_id] = 1
                else:
                    self.cellcount[cell_id] += 1 
        self.max_num_hotcells = min(self.maxvocab_size, len(self.cellcount))
        self.topcellcount = dict(sorted(self.cellcount.items(), key=lambda d: d[1], reverse=True))
        self.hotcell = []
        for key in self.topcellcount.keys():
            if self.topcellcount[key] >= self.mintimefreq:
                self.hotcell.append(key)
        self.hotcell2vocab = defaultdict(list)
        for (i, cell) in enumerate(self.hotcell):
            self.hotcell2vocab[cell] = i+self.vocab_start
        self.vocab2hotcell = {value:key for key,value in self.hotcell2vocab.items()}
        self.vocab_size = self.vocab_start + len(self.hotcell)
        y = list(np.zeros(len(self.hotcell)))
        self.D2_hotcell = list(zip(self.hotcell, y))
        self.hotcell_kdtree = KDTree(self.D2_hotcell, leaf_size=2)

    def knearestHotcells(self, cell, k):
        coord = []
        t = self.cell2time(cell)
        coord.append([t,0])
        dists, idxs = self.hotcell_kdtree.query(coord, k)
        kcells = []
        for i in range(len(idxs[0])):
            kcells.append(self.hotcell[idxs[0][i]])       
        return kcells, dists

    
    def nearestHotcell(self, cell): 
        kcells, _ = self.knearestHotcells(cell, 1)
        return kcells[0]
    
    
    def saveKNearestVocabs(self):
        V = np.zeros((self.vocab_size, self.k))
        D = np.zeros((self.vocab_size, self.k))
        for vocab in range(self.vocab_start):
            V[vocab, :] = vocab
            D[vocab, :] = 0.0
        for vocab in range(self.vocab_start, self.vocab_size):
            cell = self.vocab2hotcell[vocab]
            kcells, dists = self.knearestHotcells(cell, self.k)
            kvocabs = []
            for i in range(len(kcells)):
                kvocabs.append(self.hotcell2vocab[kcells[i]])
            V[vocab, :] = kvocabs
            D[vocab, :] = dists
        cellsize = int(self.timestep)
        filename = os.path.join('data', "{}-vocab-timedist-cell{}.h5".format(self.name,cellsize))
        f = h5py.File(filename,'w')
        f["V"], f["D"] = V, D
        f.close()

    def cell2vocab(self, cell):
        if self.hotcell2vocab[cell]:
            return self.hotcell2vocab[cell]
        else:
            hotcell = self.nearestHotcell(cell)
            return self.hotcell2vocab[hotcell]
    
    def time2vocab(self, t):
        if not self.timeinterval(t):
            return UNK
        cell_id = self.time2cell(t)
        vocab_id = self.cell2vocab(cell_id)
        return vocab_id


    def time2seq(self, timeseq):
        seq = []
        for i in range(len(timeseq)):
            t = timeseq[i]
            vocab_id = self.time2vocab(t)
            seq.append(vocab_id)
        return seq

class Vocab():
    def __init__(self, vocab_start):
        self.vocab_start = vocab_start
    
    def keyword2vocab(self, wordset):
        self.word2vocab = {}
        
        for (i, word) in enumerate(wordset):
            self.word2vocab[word] = i+self.vocab_start
        self.vocab_size = self.vocab_start + len(self.word2vocab)
        self.vocab2word = {value:key for key,value in self.word2vocab.items()}

def initvocab():
    vocab = Vocab(4)
    textembeddingfile = os.path.join('data', Config.DATASETS.textembeddings_file)
    output_vecs = pickle.load(open(textembeddingfile, 'rb'), encoding='bytes')
    keywords = list(output_vecs.keys())
    vocab.keyword2vocab(keywords)
    textvocab = os.path.join('data', Config.DATASETS.textvocab_file)
    pickle.dump(vocab, open(textvocab, 'wb'), protocol=2)

if __name__ == '__main__':
    args = parse.get_args()
    Config.merge_from_file(args.config_file) 
    device = torch.device('cpu')
    if Config.SOLVER.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    
    dataset = os.path.join('data', Config.DATASETS.dataset)
    print(dataset)
    data = pickle.load(open(dataset, 'rb'), encoding='bytes')
    trajdata = data["spatial_seqs"]
    timedata = data["time_seqs"]
    textdata = data["keyword_seqs"]
    
    init_grid(device, trajdata) 
    
    random.shuffle(timedata)
    random_numbers = random.sample(range(len(timedata)), 10000)
    times = [timedata[num] for num in random_numbers]
    times = transtimes(times)
    print(Config.time_embedding_size)
    train_d2vec(device, times)
    
    keywords = getkeywords(textdata)
    keywords.append('bos')
    keywords.append('eos')
    output_vecs = get_bert_embeddings(device, keywords)
    textembeddingfile = os.path.join('data', Config.DATASETS.textembeddings_file)
    pickle.dump(output_vecs, open(textembeddingfile, 'wb'), protocol=2)
    
    output_vecs = pickle.load(open(textembeddingfile, 'rb'), encoding='bytes')
    training_embeddings = []
    for i in output_vecs.keys():
        training_embeddings.append(output_vecs[i])
    train_text2Vec(device, training_embeddings)
    initvocab()
    testfile = os.path.join('data', Config.DATASETS.poi_st_tdrive_test_similar_file)
    valifile = os.path.join('data', Config.DATASETS.poi_st_tdrive_vali_similar_file)
    generate_newsimi_test_dataset(dataset, testfile, is_vali=False)
    generate_newsimi_test_dataset(dataset, valifile, is_vali=True)
    testori = os.path.join('data', Config.DATASETS.poi_st_tdrive_test_knn_query)
    testcq = os.path.join('data', Config.DATASETS.poi_st_tdrive_test_knn_changedquery)
    testdb = os.path.join('data', Config.DATASETS.poi_st_tdrive_test_knn_db)
    valitori = os.path.join('data', Config.DATASETS.poi_st_tdrive_vali_knn_query)
    valicq = os.path.join('data', Config.DATASETS.poi_st_tdrive_vali_knn_changedquery)
    validb = os.path.join('data', Config.DATASETS.poi_st_tdrive_vali_knn_db)
    generate_knn_test_dataset(dataset, valitori, valicq, validb, True)
   

    