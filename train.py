import logging
import os
import shutil
import time
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn
from functools import partial
import pickle
import datetime
from data_utils.data_loader import read_traj_dataset, read_postraj_dataset
from config import Config, parse
from utils import util
from utils.checkpoint import CheckPointer
from augments import get_aug_fn, downsamplingDistort
from model import Date2vec, Text2Vec, MultiModal_2vec
from semantictraj_preprocessing import Grid, Vocab
import constants

def collate_and_augment(trajs):
    trajs1_seq, trajs2_seq = [], []
    for i in range(len(trajs)):
        trajs1_seq.append(trajs[i][0])
        trajs2_seq.append(trajs[i][1])
    trgtrajs_seq = trajs1_seq
    srctrajs_seq = trajs1_seq
    
    trajs1_lens = torch.tensor(list(map(len, trajs1_seq)), dtype = torch.long)
    trajs2_lens = torch.tensor(list(map(len, trajs2_seq)), dtype = torch.long)
    
    srctrajs_lens = torch.tensor(list(map(len, srctrajs_seq)), dtype = torch.long)
    trgtrajs_lens = torch.tensor(list(map(len, trgtrajs_seq)), dtype = torch.long)
    
    return srctrajs_seq, srctrajs_lens, trgtrajs_seq, trgtrajs_lens, trajs1_seq, trajs1_lens, trajs2_seq, trajs2_lens

def collate_for_test(trajs):
    trgtrajs_seq = trajs
    trgtrajs_lens = torch.tensor(list(map(len, trgtrajs_seq)), dtype = torch.long)
    return trgtrajs_seq, trgtrajs_lens

def save_checkpoint(state, is_best):
    torch.save(state, Config.checkpoint)
    if is_best:
        shutil.copyfile(Config.checkpoint, os.path.join(
            Config.data_path, 'best_model.pt'))

def load_checkpoint(model, optimizer):
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    checkpoint = torch.load(Config.checkpoint)
    start_iteration = checkpoint["iteration"]
    best_prec_loss = checkpoint["best_prec_loss"]
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["model_optimizer"])
    model.to(device)
    optimizer.to(device)
def get_timelist(traj_seq):
    timelist = []
    for i in range(len(traj_seq)):
        temp_time = []
        for j in range(len(traj_seq[i])):
            temp_time.append(traj_seq[i][j][2]) 
        timelist.append(temp_time)
    return timelist

def get_trgtimelist(traj_seq):
    time_seq = get_timelist(traj_seq)
    print(len(time_seq))
    all_list = []
    for one_seq in time_seq:
        one_list = []
        for timestamp in one_seq:
            date_obj = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            t = datetime.datetime.fromtimestamp(date_obj.timestamp())
            t = [t.hour, t.minute, t.second, t.year, t.month, t.day]
            x = t
            one_list.append(x)
        all_list.append(torch.Tensor(one_list))
    all_list = pad_sequence(all_list, batch_first=True)
    return all_list

def getTimeEmbedding(traj_seq):
    d2vec = Date2vec()
    timelist = get_timelist(traj_seq)
    for i in range(len(timelist)):
        if len(timelist[i]) == 0:
            print('---------is null-------',i)
    d2v = d2vec(timelist)
    return d2v #[batch_size, max_seqlen, time_embedding_size]

def get_trajvocabs(traj_seq):
    grid_file = os.path.join('data', Config.DATASETS.grid_file)
    Grid = pickle.load(open(grid_file, 'rb'), encoding='bytes')
    vocablist = []
    for i in range(len(traj_seq)):
        temp_vocab = []
        for j in range(len(traj_seq[i])):
            vocabid = Grid.gps2vocab(traj_seq[i][j][0], traj_seq[i][j][1])
            temp_vocab.append(vocabid)         
        vocablist.append(temp_vocab)
    return vocablist

def get_trgSpatialvocab(traj_seq):
    vocablist = get_trajvocabs(traj_seq)
    trajs_vocab_seq = [[constants.BOS]+t+[constants.EOS] for t in vocablist]
    
    return trajs_vocab_seq

def getSpatialEmbedding(traj_seq, flag='encoder'):
    embs = pickle.load(open('models/cell50_embdim256_datasetpoi_simplified_st_tdrive_embs.pkl', 'rb')).to('cpu').detach()  #models/cell50_embdim128_datasetpoi_simplified_st_tdrive_embs.pkl
    if flag == 'decoder':
        trajs_emb_cell = [embs[list(t)] for t in traj_seq]
    else:
        vocablist = get_trajvocabs(traj_seq)
        trajs_emb_cell = [embs[list(t)] for t in vocablist] 
    
    trajs_emb_cell = pad_sequence(trajs_emb_cell, batch_first = True)
    return trajs_emb_cell #[batch_size, max_seqlen, spatial_embedding_size]

def get_textvocabs(traj_seq):
    vocab_file = os.path.join('data', Config.DATASETS.textvocab_file)
    Vocab = pickle.load(open(vocab_file, 'rb'), encoding='bytes')
    vocablist = []
    for i in range(len(traj_seq)):
        temp_vocab = []
        for j in range(len(traj_seq[i])): 
            vocabid = Vocab.word2vocab[traj_seq[i][j][3]]
            temp_vocab.append(vocabid)         
        vocablist.append(temp_vocab)
    return vocablist

def get_trgTextvocab(traj_seq):
    vocablist = get_textvocabs(traj_seq)
    trajs_vocab_seq = [torch.tensor(t) for t in vocablist]
    trajs_vocab_seq = pad_sequence(trajs_vocab_seq, batch_first=True)
    return trajs_vocab_seq

def bert_embedding(traj_seq):
    textembeddingfile = os.path.join('data', Config.DATASETS.textembeddings_file)
    text_vecs = pickle.load(open(textembeddingfile, 'rb'), encoding='bytes')
    trajs_bert_embs = []
    for i in range(len(traj_seq)):
        temp_text = []
        for j in range(len(traj_seq[i])):
            keyword = traj_seq[i][j][3]
            emb = text_vecs[keyword]
            temp_text.append(emb)
        trajs_bert_embs.append(temp_text)
    return trajs_bert_embs

def wordvocab(traj_seq):
    trajs_word_vocabs = []
    for i in range(len(traj_seq)):
        temp_text = []
        for j in range(len(traj_seq[i])):
            keyword = traj_seq[i][j][3] 
            temp_text.append(keyword)
        trajs_word_vocabs.append(temp_text)
    return trajs_word_vocabs

def getTextEmbedding(traj_seq):
    text2vec = Text2Vec()
    trajs_bert_embs = bert_embedding(traj_seq)
    t2v = text2vec(trajs_bert_embs)
    return t2v #[batch_size, max_seqlen, text_embedding_size]

def validate(model):
    model.eval()
    total_loss = 0
    trajs_file = os.path.join('data', Config.DATASETS.dataset)
    _, eval_dataset, _ = read_postraj_dataset(trajs_file)
   
    eval_dataloader = DataLoader(eval_dataset, 
                                batch_size = Config.SOLVER.val_batchsize, 
                                shuffle = False, 
                                num_workers = 0, 
                                drop_last = False, 
                                collate_fn = collate_and_augment)
  
    with torch.no_grad():
        for i_batch, (srctrajs_seq, srctrajs_lens, trgtrajs_seq, trgtrajs_lens, 
                      trajs1_seq, trajs1_lens, trajs2_seq, trajs2_lens ) in tqdm(enumerate(eval_dataloader)):
            trajs1_time_seq = getTimeEmbedding(trajs1_seq)
            trajs1_spatial_seq = getSpatialEmbedding(trajs1_seq)
            trajs1_text_seq = getTextEmbedding(trajs1_seq)
            trajs2_time_seq = getTimeEmbedding(trajs2_seq)
            trajs2_spatial_seq = getSpatialEmbedding(trajs2_seq)
            trajs2_text_seq = getTextEmbedding(trajs2_seq)
            
            trgtraj_spatial_seq = getSpatialEmbedding(trgtrajs_seq)
            trgtraj_time_seq = getTimeEmbedding(trgtrajs_seq)
            trgtraj_text_seq = getTextEmbedding(trgtrajs_seq)
           
            logits, labels, inter_logits, inter_targets,  = model(trgtraj_spatial_seq, trgtraj_time_seq, trgtraj_text_seq, trgtrajs_lens,
                trajs1_spatial_seq, trajs1_time_seq, trajs1_text_seq, trajs1_lens, 
                trajs2_spatial_seq, trajs2_time_seq, trajs2_text_seq, trajs2_lens)
            loss, intra_loss, inter_loss = model.loss_fn_interintra(logits, labels, inter_logits, inter_targets)
            total_loss += loss.item()
    
    trajs_file = os.path.join('data', Config.DATASETS.poi_st_tdrive_vali_similar_file)
    rank = test_model(model, trajs_file)
    model.train()
   
    return rank, total_loss


def test_model(model,testtrajs_file):
    query_lst, db_lst = pickle.load(open(testtrajs_file, 'rb'), encoding='bytes') # query_lst:(1000, 不定长, 3) db_lst(8000, 不定长, 3)
    results = []
    querys, databases = test_trajs_to_embs(model, query_lst, db_lst)
    
    dists = torch.cdist(querys, databases, p = 1) 
    targets = torch.diag(dists)
    
    rank = torch.sum(torch.le(dists.T, targets)).item() / querys.shape[0]
    results.append(rank)
    
    return rank

def test_trajs_to_embs(model, query_lst, db_lst):
    querys = []
    databases = []
    num_query = len(query_lst)
    num_database = 4000  
    batch_size = num_query
    test_batch = Config.SOLVER.test_batchsize
    if num_query <= test_batch:
        query_lst = query_lst[:num_query]
        trajseq, trajlens = collate_for_test(query_lst)
        traj_spatial_seq = getSpatialEmbedding(trajseq)
        traj_time_seq = getTimeEmbedding(trajseq)
        traj_text_seq = getTextEmbedding(trajseq)
        trajs1_emb = model.interpret(traj_spatial_seq, traj_time_seq, traj_text_seq, trajlens)
        querys.append(trajs1_emb.detach().cpu().numpy())
    else:
        i = 0
        while i < num_query:
            trajseq, trajlens = collate_for_test(query_lst[i:i+test_batch])
            traj_spatial_seq = getSpatialEmbedding(trajseq)
            traj_time_seq = getTimeEmbedding(trajseq)
            traj_text_seq = getTextEmbedding(trajseq)
            trajs1_emb = model.interpret(traj_spatial_seq, traj_time_seq, traj_text_seq, trajlens)
            querys.append(trajs1_emb.detach().cpu().numpy())
            
            i += test_batch
    
    if num_database <= test_batch:
        db_lst = db_lst[:num_database]
        trajseq, trajlens = collate_for_test(db_lst)
        traj_spatial_seq = getSpatialEmbedding(trajseq)
        traj_time_seq = getTimeEmbedding(trajseq)
        traj_text_seq = getTextEmbedding(trajseq)
        trajs2_emb = model.interpret(traj_spatial_seq, traj_time_seq, traj_text_seq, trajlens)
        databases.append(trajs2_emb.detach().cpu().numpy())
    else:
        i = 0
        while i < num_database:
            trajseq, trajlens = collate_for_test(db_lst[i:i+test_batch])
            traj_spatial_seq = getSpatialEmbedding(trajseq)
            traj_time_seq = getTimeEmbedding(trajseq)
            traj_text_seq = getTextEmbedding(trajseq)
            trajs2_emb = model.interpret(traj_spatial_seq, traj_time_seq, traj_text_seq, trajlens)     
            databases.append(trajs2_emb.detach().cpu().numpy())
            i += test_batch
    import numpy as np
    
    querys = torch.from_numpy(np.concatenate(querys)).to('cuda')
    databases = torch.from_numpy(np.concatenate(databases)).to('cuda')
   
    return querys, databases

def test():
    device = torch.device('cpu')
    if Config.SOLVER.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    model = MultiModal_2vec(Config.spatial_embedding_size, 
                            Config.text_embedding_size, 
                            Config.trans_hidden_dim, 
                            Config.trans_attention_head, 
                            Config.trans_attention_layer, 
                            Config.trans_attention_dropout, 
                            Config.trans_pos_encoder_dropout, device)
    results = []
    model = model.to(device)
    testtrajs_file = os.path.join('data', Config.DATASETS.poi_st_tdrive_test_similar_file)
    rank = test_model(model, testtrajs_file)
   
    results.append(rank)
    return rank

def train(load_model=None, load_optimizer=None):
    logger = logging.getLogger('CL')
    logger.info(f"Start training for {Config.SOLVER.epochs} epochs.")
    device = torch.device('cpu')
    if Config.SOLVER.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    model = MultiModal_2vec(Config.spatial_embedding_size, 
                            Config.text_embedding_size, 
                            Config.trans_hidden_dim, 
                            Config.trans_attention_head, 
                            Config.trans_attention_layer, 
                            Config.trans_attention_dropout, 
                            Config.trans_pos_encoder_dropout, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.SOLVER.BASE_LR, weight_decay=Config.SOLVER.WEIGHT_DECAY)
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SOLVER.LR_STEP, gamma=Config.SOLVER.LR_GAMMA)
    
    checkpointer = CheckPointer(model, optimizer, save_dir='checkpoints')
    trajs_file = os.path.join('data', Config.DATASETS.dataset)
    print(trajs_file)
    train_dataset, _, _ = read_postraj_dataset(trajs_file)
    train_dataloader = DataLoader(train_dataset, 
                                    batch_size = Config.SOLVER.train_batchsize, 
                                    shuffle = False, 
                                    num_workers = 0, 
                                    drop_last = False, 
                                    collate_fn = collate_and_augment)
    model = model.to(device)
    n_iter = 0
    for i_ep in range(Config.SOLVER.epochs):
        model.train()
        _time_batch_start = time.time()
        for i_batch, batch in tqdm(enumerate(train_dataloader)):
            srctrajs_seq, srctrajs_lens, trgtrajs_seq, trgtrajs_lens, trajs1_seq, trajs1_lens, trajs2_seq, trajs2_lens = batch
            
            trajs1_time_seq = getTimeEmbedding(trajs1_seq)
            trajs1_spatial_seq = getSpatialEmbedding(trajs1_seq)
            trajs1_text_seq = getTextEmbedding(trajs1_seq)
            
            trajs2_time_seq = getTimeEmbedding(trajs2_seq)
            trajs2_spatial_seq = getSpatialEmbedding(trajs2_seq)
            trajs2_text_seq = getTextEmbedding(trajs2_seq) 
            
            
            trgtraj_spatial_seq = getSpatialEmbedding(trgtrajs_seq)
            trgtraj_time_seq = getTimeEmbedding(trgtrajs_seq)
            trgtraj_text_seq = getTextEmbedding(trgtrajs_seq) 
           
            
            logits, labels, inter_logits, inter_targets = model(trgtraj_spatial_seq, trgtraj_time_seq, trgtraj_text_seq, trgtrajs_lens,
                trajs1_spatial_seq, trajs1_time_seq, trajs1_text_seq, trajs1_lens, 
                trajs2_spatial_seq, trajs2_time_seq, trajs2_text_seq, trajs2_lens)
            loss, intra_loss, inter_loss = model.loss_fn_interintra(logits, labels, inter_logits, inter_targets)
            
            loss = loss / Config.SOLVER.train_batchsize
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_iter += 1
            if i_batch % Config.SOLVER.print_freq == 0 and i_batch:
                
                logger.info("[Training intra+inter] ep-batch={}-{}, total_loss={:.3f},intra_loss={:.3f},inter_loss={:.3f}, @={:.3f}"
                            .format(i_ep, i_batch, loss,intra_loss.item(), inter_loss.item(), time.time() - _time_batch_start))
        
        scheduler.step()
        logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        if i_ep % Config.SOLVER.save_freq == 0 and i_ep: 
            rank, vali_loss = validate(model)
                  
            logger.info(
                f"\nvalidata model with loss {vali_loss} at epoch {i_ep} \n"
                f"rank : {rank} \n")
            print(
                f"\nvalidata model with loss {vali_loss} at epoch {i_ep} \n"
                f"rank : {rank} \n")
            
            checkpointer.save(i_ep, rank, vali_loss)
    logger.info("Training has finished.")


if __name__ == '__main__':
    args = parse.get_args()
    Config.merge_from_file(args.config_file) 
    Config.logdir = os.path.join(
        'logs', time.strftime("%Y%m%d%H%M%S", time.localtime())) 
    if not os.path.exists(Config.logdir):
        os.mkdir(Config.logdir)
    Config.freeze() 
    logger = util.setup_logger('CL', Config.logdir)
    logger.info(f"Loaded configuration file {args.config_file}")
    logger.info(f"Runing with config:\n{Config}")
    train()
    