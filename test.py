import torch
from config import Config, parse
from model import Date2vec, Text2Vec, MultiModal_2vec
import os
import pickle
import numpy as np
import time
from train import getSpatialEmbedding, getTimeEmbedding, getTextEmbedding, test_model
from utils import util
from semantictraj_preprocessing import Grid, Vocab
 
def collate_for_test(trajs):
    trgtrajs_seq = trajs
    trgtrajs_lens = torch.tensor(list(map(len, trgtrajs_seq)), dtype = torch.long)
    return trgtrajs_seq, trgtrajs_lens

def test_knn(fname):
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
    model.load_state_dict(torch.load(fname)['model'])
    results = []
    model = model.to(device)
    query_file = os.path.join('data', Config.DATASETS.poi_st_tdrive_test_knn_query)
    distortquery_file = os.path.join('data', Config.DATASETS.poi_st_tdrive_test_knn_changedquery)
    db_file = os.path.join('data', Config.DATASETS.poi_st_tdrive_test_knn_db)
    recall_1, recall_5, recall_10, recall_10_50 = test_knn_model(model, query_file, distortquery_file, db_file)
    logger.info(f"recall_1 :{recall_1}\n"
                f"recall_5 : {recall_5} \n"
                f"recall_10 : {recall_10} \n"
                f"recall_10_50 : {recall_10_50} \n")
    print('------recall_1, recall_5, recall_10, recall_10_50------', recall_1, recall_5, recall_10, recall_10_50)
    return recall_1, recall_5, recall_10_50

def test_knn_model(model,testtrajs_file, distortquery_file, datrajs_file):
    query_lst = pickle.load(open(testtrajs_file, 'rb'), encoding='bytes') 
    distort_query_lst = pickle.load(open(distortquery_file, 'rb'), encoding='bytes')
    db_lst = pickle.load(open(datrajs_file, 'rb'), encoding='bytes') 
    results = []
    querys = test_trajs_to_embs_knn(model, query_lst, 1000)
    distort_querys = test_trajs_to_embs_knn(model, distort_query_lst, 1000)
    databases = test_trajs_to_embs_knn(model, db_lst, 4000)
    
    
    dists = torch.cdist(querys, databases, p = 1) 
    distort_dists = torch.cdist(distort_querys, databases, p = 1) 
    l_recall_1 = 0
    l_recall_5 = 0
    l_recall_10 = 0
    l_recall_10_50 = 0
    f_num = 0
    for i in range(len(dists)):
        input_r = np.array(dists[i].cpu())
        one_index = []
        for idx, value in enumerate(input_r):
            one_index.append(idx)
        input_r = input_r[one_index]
        input_r = input_r[:1000]
        input_r5 = np.argsort(input_r)[:5]
        input_r1 = input_r5[:1]
        input_r50 = np.argsort(input_r)[:50]
        input_r10 = input_r50[:10]

        embed_r = np.array(distort_dists[i].cpu())
        embed_r = embed_r[one_index]
        embed_r = embed_r[:1000]

        embed_r5 = np.argsort(embed_r)[:5]
        embed_r1 = embed_r5[:1]
        embed_r50 = np.argsort(embed_r)[:50]
        embed_r10 = embed_r50[:10]

        f_num += 1
        l_recall_1 += len(list(set(input_r1).intersection(set(embed_r1))))
        l_recall_5 += len(list(set(input_r5).intersection(set(embed_r5))))
        l_recall_10 += len(list(set(input_r10).intersection(set(embed_r10))))
        l_recall_10_50 += len(list(set(input_r50).intersection(set(embed_r10))))
    print('f_num: ', f_num)
    recall_1 = float(l_recall_1) / f_num
    recall_5 = float(l_recall_5) / (5 * f_num)
    recall_10 = float(l_recall_10) / (10 * f_num)
    recall_10_50 = float(l_recall_10_50) / (10 * f_num)

    return recall_1, recall_5, recall_10, recall_10_50

def test_trajs_to_embs_knn(model, data_lst, num_query): 
    data = []
    batch_size = num_query
    test_batch = Config.SOLVER.test_batchsize
    if num_query <= test_batch:
        data_lst = data_lst[:num_query]
        trajseq, trajlens = collate_for_test(data_lst)
        traj_spatial_seq = getSpatialEmbedding(trajseq)
        traj_time_seq = getTimeEmbedding(trajseq)
        traj_text_seq = getTextEmbedding(trajseq)
        trajs1_emb = model.interpret(traj_spatial_seq, traj_time_seq, traj_text_seq, trajlens)
        data.append(trajs1_emb.detach().cpu().numpy())
    else:
        i = 0
        while i < num_query:
            trajseq, trajlens = collate_for_test(data_lst[i:i+test_batch])
            traj_spatial_seq = getSpatialEmbedding(trajseq)
            traj_time_seq = getTimeEmbedding(trajseq)
            traj_text_seq = getTextEmbedding(trajseq)
            trajs1_emb = model.interpret(traj_spatial_seq, traj_time_seq, traj_text_seq, trajlens)
            data.append(trajs1_emb.detach().cpu().numpy())           
            i += test_batch
    data = torch.from_numpy(np.concatenate(data)).to('cuda')
    return data

def test_most_similar(fname):
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
    model.load_state_dict(torch.load(fname)['model'])
    # print(list(model.named_parameters())[0])
    results = []
    model = model.to(device)
    testtrajs_file = os.path.join('data', Config.DATASETS.poi_st_tdrive_test_similar_file)
    rank = test_model(model, testtrajs_file)
    results.append(rank)
    logger.info(f"rank :{rank}\n"
                )
    print('------rank------', rank)
    return rank


if __name__ == '__main__':
    args = parse.get_args()
    Config.merge_from_file(args.config_file) #使yaml文件中的值覆盖config的默认值
    Config.logdir = os.path.join(
        'test_logs', time.strftime("%Y%m%d%H%M%S", time.localtime())) #测试时的logs
    if not os.path.exists(Config.logdir):
        os.mkdir(Config.logdir)
    Config.freeze() 

    logger = util.setup_logger('CL', Config.logdir)
    logger.info(f"Loaded configuration file {args.config_file}")
    logger.info(f"Runing with config:\n{Config}")
    
    test_knn('/root/autodl-tmp/code_laq/CONS_ST2Vec/checkpoints_simsub/poi_simplified_st_tdrive_frechet_epoch_106_rank_4.12_Loss_43.06245231628418.pt')
    test_most_similar('/root/autodl-tmp/code_laq/CONS_ST2Vec/checkpoints_embed_64/poi_simplified_st_tdrive_frechet_epoch_148_rank_1.215_Loss_42.616127014160156.pt')
    
    







