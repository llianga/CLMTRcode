import torch
from torch import nn
import torch.backends.cudnn
from config import Config, parse
from torch.utils.data.dataloader import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import pickle
import datetime
from tqdm import tqdm
from utils.checkpoint import CheckPointer
import time
from functools import partial
import random

class TimeDataset(Dataset): 
    def __init__(self, data):
        # data: DataFrame
        self.data = data
          
    def __getitem__(self, i):
       #返回的必须是tensor/array/list等数据类型，整个数据集中的数据长度是相等的 
        return self.data[i]

    def __len__(self):
        return len(self.data)

def downsampling(st_traj, dropping_rate=0.4):
    down_traj = []
    for i in range(len(st_traj)):
        if random.random() > dropping_rate:
            down_traj.append(st_traj[i])
    if len(down_traj) == 0:
        down_traj = st_traj[::2]
    return down_traj

def collate_and_augment(times):
    # times: list of ['2012-12-15 00:13:02', ...] 时间序列集
    times_seq = [] 
    dropping_rate=0.4 #test uneven sampling 
    for i in range(len(times)):  
        t = times[i]
        x = torch.Tensor(t).float()
        # temp_time_seq.append(trajs[i][j][2])  
        times_seq.append(x)  
        # if random.random() > dropping_rate: #test uneven sampling
        #     # print(111)
        #     times_seq.append(x)
    # print('times_seq:', times_seq)  
    
    # time_lens = torch.tensor(list(map(len, times_seq)), dtype = torch.long)  
    times_seq = pad_sequence(times_seq, batch_first = True)
    
    # print('times_seq.shape----------', times_seq.shape)
    return times_seq

class Date2VecConvert:
    def __init__(self, model_path="d2vec_checkpoints/epoch_90.pt"):
        state_dict = torch.load(model_path, map_location='cpu')
        # print(state_dict.keys())
        self.model = Date2Vec(k=Config.time_embedding_size)
        # print('---Date2VecConvert Config.time_embedding_size---', Config.time_embedding_size)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        # print(2)
    
    def __call__(self, x):
        with torch.no_grad():
            return self.model.encode(torch.Tensor(x).unsqueeze(0)).squeeze(0).cpu()

class Date2Vec(nn.Module):
    def __init__(self, k=Config.time_embedding_size, act="sin"):
        super(Date2Vec, self).__init__()
        # print('---Date2Vec Config.time_embedding_size---', k)
        
        if k % 2 == 0:
            k1 = k // 2
            k2 = k // 2
        else:
            k1 = k // 2
            k2 = k // 2 + 1
        
        self.fc1 = nn.Linear(6, k1)

        self.fc2 = nn.Linear(6, k2)
        self.d2 = nn.Dropout(0.3)
 
        if act == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.cos

        self.fc3 = nn.Linear(k, k // 2)
        self.d3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(k // 2, 6)
        
        self.fc5 = torch.nn.Linear(6, 6)

    def forward(self, x):
        # print("---x---", x)
        out1 = self.fc1(x)
        out2 = self.d2(self.activation(self.fc2(x)))
        out = torch.cat([out1, out2], 1)
        out = self.d3(self.fc3(out))
        out = self.fc4(out)
        out = self.fc5(out)
        return out

    def encode(self, x):
        out1 = self.fc1(x)
        # out2 = self.activation(self.fc2(x))
        out2 = self.fc2(x)
        out = torch.cat([out1, out2], 1)
        return out
    
    def loss(self, out, x):
        loss = nn.MSELoss(out,x)
        return loss


def train_d2vec(_device, times):
    model = Date2Vec()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.SOLVER.BASE_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SOLVER.LR_STEP, gamma=Config.SOLVER.LR_GAMMA)
    
    criterion = nn.MSELoss()
    train_dataset = TimeDataset(times)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=256, 
        shuffle=True,
        collate_fn=collate_and_augment)
    model = model.to(_device)
    model.train()
    epoch_train_loss_best = 1000000000.0
    epoch_best = 0
    total_epoch = 100
    epoch_patience = 30
    epoch_worse_count = 0
    print("Training time embedding...")

    for i in range(total_epoch):
        total_loss = 0
        time_ep = time.time()
        for input in tqdm(train_dataloader): #遍历数据集中的每个batch
            batch_size = len(input)
            
            input = input.to(_device)
            output = model(input)
            loss = criterion(output.to(_device), input.to(_device)) #每个batch的损失
            # loss.backward()
            # optimizer.step()
            total_loss += loss.item()  #为了得到每个epoch的损失
            loss = loss / batch_size #每个数据的损失
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("[time2vec] i_ep={}, loss={:.4f} @={}".format(i, total_loss, time.time()-time_ep))
        scheduler.step()
        
        if total_loss < epoch_train_loss_best:
            epoch_best = i
            epoch_train_loss_best = total_loss
            epoch_worse_count = 0
            torch.save(model.state_dict(), f"d2vec_checkpoints/d2vec_256_epoch_{i}_loss_{total_loss}.pt")
            
        else:
            epoch_worse_count += 1
            if epoch_worse_count >= epoch_patience:
                break

    
    '''save_embeddings'''#保存的是最后一个model的参数吗
    # embs = model()
    # embs_file = 'data' + '_time' + '_embdim' + str(Config.time_embedding_size) + '_embs.pkl'
    # embs_file = os.path.join('models', embs_file)
    # pickle.dump(embs, open(embs_file, 'wb'), protocol=2)
    print("[time2vec], best_ep={}".format(epoch_best))
    


if __name__ == "__main__":
    args = parse.get_args()
    Config.merge_from_file(args.config_file) 
    device = torch.device('cpu')
    if Config.SOLVER.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    dataset = os.path.join('data', Config.DATASETS.dataset)
    
    data = pickle.load(open(dataset, 'rb'), encoding='bytes')
    
    timedata = data["time_seqs"]
    # train(device, timedata)
    times = []
    for i in range(len(timedata)): 
        for j in range(len(timedata[i])):
            timestamp = timedata[i][j]
            date_obj = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            t = datetime.datetime.fromtimestamp(date_obj.timestamp())
            # t = datetime.datetime.fromtimestamp(timestamp)
            t = [t.hour, t.minute, t.second, t.year, t.month, t.day]
            # times.append(t)
            dropping_rate = 0.4
            if random.random() > dropping_rate: #test uneven sampling
                times.append(t)
    print('len times:', len(times))
    train_d2vec(device, times)