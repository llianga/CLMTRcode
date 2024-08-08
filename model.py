import datetime
import torch
import torch.nn as nn
from config import Config
from d2vecModel import Date2VecConvert
from bertmodel import Text2VecConvert
from moco import MoCo
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import math
import constants

class Date2vec(nn.Module):
    def __init__(self):
        super(Date2vec, self).__init__()
        self.d2v = Date2VecConvert(model_path="d2vec_checkpoints/d2vec_256_epoch_99_loss_3426278.12890625.pt") #d2vec_checkpoints/d2vec_epoch_93_loss_11496.215072989464.pt
    def forward(self, time_seq):
        all_list = []
        for one_seq in time_seq:
            one_list = []
            for timestamp in one_seq:
                date_obj = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                t = datetime.datetime.fromtimestamp(date_obj.timestamp())
                t = [t.hour, t.minute, t.second, t.year, t.month, t.day]
                x = torch.Tensor(t).float()
                embed = self.d2v(x)
                one_list.append(embed)
            one_list = torch.cat(one_list, dim=0)
            one_list = one_list.view(-1, Config.time_embedding_size)
            all_list.append(one_list.clone().detach())
        all_list = pad_sequence(all_list, batch_first = True) 
        return all_list
    
class Text2Vec(nn.Module):
    def __init__(self, k=Config.text_embedding_size):
        super(Text2Vec, self).__init__()
        self.t2v = Text2VecConvert(model_path="text2vec_checkpoints/epoch_90_total_loss_3.2165612019598484.pt") #text2vec_checkpoints/epoch_90_total_loss_3.2165612019598484.pt
    def forward(self, text_seq):      
        all_list = []    
        for one_embs_list in text_seq:
            one_list = []
            for emb in one_embs_list:
                emb = torch.Tensor(emb)
                embed = self.t2v(emb)
                one_list.append(embed)
            one_list = torch.cat(one_list, dim=0)
            one_list = one_list.view(-1, Config.text_embedding_size)
            all_list.append(one_list.clone().detach())
        all_list = pad_sequence(all_list, batch_first = True) 
        return all_list 
    
class textvectors(nn.Module):
    def __init__(self, vocab_size, text_embedding_size, device):
        super(textvectors, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = text_embedding_size
        self.device = device
        self.model = nn.Embedding(self.vocab_size, self.embedding_size,
                                      padding_idx=constants.PAD).to(device)
       
    def forward(self, text_seqs):  
        batch_size = len(text_seqs)
        seq_lengths = list(map(len, text_seqs))
        all_list = []
        for text_one in text_seqs:
            text_one += [0]*(max(seq_lengths)-len(text_one))
           
        text_seqs = [torch.tensor(t) for t in text_seqs]
        text_seqs = pad_sequence(text_seqs, batch_first=True)
        text_seqs = text_seqs.to(self.device)
        s_input = self.model(text_seqs)
        return s_input
    
class Fusion_Layer(nn.Module):
    def __init__(self, dim):
        super(Fusion_Layer, self).__init__()
        self.Wq = nn.Linear(dim, int(dim*0.5), bias=False)
        self.Wk = nn.Linear(dim, int(dim*0.5), bias=False)
        self.Wv = nn.Linear(dim, int(dim*0.5), bias=False)
        self.temperature = dim ** 0.5
        self.FFN = nn.Sequential(
            nn.Linear(int(dim*0.5), int(dim*0.5)),
            nn.ReLU(),
            nn.Linear(int(dim*0.5), int(dim*0.5)),
            nn.Dropout(0.1)
        )
        self.layer_norm = nn.LayerNorm(int(dim*0.5), eps=1e-6)
    def forward(self, seq_s, seq_t, src_len): 
        h = torch.stack([seq_s, seq_t], 2)  
        q = self.Wq(h)
        k = self.Wk(h)
        v = self.Wv(h)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = F.softmax(attn, dim=-1)
        attn_h = torch.matmul(attn, v)
        attn_o = self.FFN(attn_h) + attn_h
        attn_o = self.layer_norm(attn_o)
        att_s = attn_o[:, :, 0, :]
        att_t = attn_o[:, :, 1, :]
        output = torch.cat((att_s, att_t), dim=2) #output: [batch_size, seq_len, seq_embedding_dim]
        rtn = torch.sum(output, 1)
        rtn = rtn / src_len.unsqueeze(-1).expand(rtn.shape) #rtn:[batch_size, seq_embedding_dim]
        return output, rtn  
    
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int= 300): 
        super(PositionalEncoding, self).__init__()
        den = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000)) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den) 
        pos_embedding[:, 1::2] = torch.cos(pos * den) 
        pos_embedding = pos_embedding.unsqueeze(0) 
        self.dropout = nn.Dropout(dropout) 
        self.register_buffer('pos_embedding', pos_embedding) 
        
    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0),:token_embedding.size(1), :])

class ST_Transformer(nn.Module):
    def __init__(self, embedding_size, ninput, nhidden, nhead, nlayer, attn_dropout, pos_droput, device): #embedding_size是时间和文本嵌入的embedding大小，max_len是该batch中的max_seqlen，即第1维的大小
        super(ST_Transformer, self).__init__()
        self.device = device
        self.ninput = ninput
        self.nhidden = nhidden
        self.nhead = nhead
        self.fusionlayer = Fusion_Layer(embedding_size).to(device)
        self.pos_encoder = PositionalEncoding(ninput, pos_droput).to(device)
        
        structural_attn_layers = nn.TransformerEncoderLayer(ninput, nhead, nhidden, attn_dropout, batch_first=True).to(device)
        self.structural_attn = nn.TransformerEncoder(structural_attn_layers, nlayer).to(device) #batch_first参数
        
        self.gamma_param = nn.Parameter(data = torch.tensor(0.5), requires_grad = True).to(device)
    
    def forward(self, traj_seqs, time_seqs, attn_mask, src_padding_mask, src_len):
        traj_seqs = traj_seqs.to(self.device)
        time_seqs = time_seqs.to(self.device)
        src_len = src_len.to(self.device)
        output, rtn = self.fusionlayer(traj_seqs, time_seqs, src_len)
        src = self.pos_encoder(output) 
        output = self.structural_attn(src, None, src_padding_mask) 
       
        mask = 1 - src_padding_mask.unsqueeze(-1).expand(output.shape).float() 
        
        res = mask * output
       
        rtn = torch.sum(mask * output, 1)
        rtn = rtn / src_len.unsqueeze(-1).expand(rtn.shape)
        return output, rtn 
    
class Text_Transformer(nn.Module):
    def __init__(self, ninput, nhidden, nhead, nlayer, attn_dropout, pos_droput, device): 
        super(Text_Transformer, self).__init__()
        self.device = device
        self.ninput = ninput
        self.nhidden = nhidden
        self.nhead = nhead
        self.pos_encoder = PositionalEncoding(ninput, pos_droput).to(device)
        
        structural_attn_layers = nn.TransformerEncoderLayer(ninput, nhead, nhidden, attn_dropout, batch_first=True).to(device)
        self.structural_attn = nn.TransformerEncoder(structural_attn_layers, nlayer).to(device) 
        
        self.gamma_param = nn.Parameter(data = torch.tensor(0.5), requires_grad = True).to(device)
    
    def forward(self, text_seqs, attn_mask, src_padding_mask, src_len):
        text_seqs = text_seqs.to(self.device)
        src_len = src_len.to(self.device) 
        src = self.pos_encoder(text_seqs)
        output = self.structural_attn(src, None, src_padding_mask) 
       
        mask = 1 - src_padding_mask.unsqueeze(-1).expand(output.shape).float() 
        res = mask * output
       
        rtn = torch.sum(mask * output, 1)
        
        rtn = rtn / src_len.unsqueeze(-1).expand(rtn.shape)
        return output, rtn 

    
class Sttext_Traj_Encoder(nn.Module):
    def __init__(self,timeembedding_size,ninput, nhidden, nhead, nlayer, attn_dropout, pos_droput, device):
        super(Sttext_Traj_Encoder, self).__init__()
        self.device = device
        self.stencoder = ST_Transformer(timeembedding_size, ninput, nhidden, nhead, nlayer, attn_dropout, pos_droput, device)
        self.textencoder = Text_Transformer(ninput, nhidden, nhead, nlayer, attn_dropout, pos_droput, device) 
        self.fusionlayer = Fusion_Layer(ninput).to(device)
    
    def forward(self, spatial_seq, time_seq, text_seq, trajs_lens):
        max_trgtrajs_len = trajs_lens.max().item() # in essense --  
        src_padding_mask = torch.arange(max_trgtrajs_len)[None, :] >= trajs_lens[:, None] #[batch_size, seq_len] 值为true的位置会被mask，长度小于该batch中的max_len的部分会被mask
        src_padding_mask = src_padding_mask.to(self.device) 
        trajs_lens = trajs_lens.to(self.device)
        st_output, _ = self.stencoder(spatial_seq, time_seq, None, src_padding_mask, trajs_lens) #st_output: [batch_size, seq_len, spatial_embedding_size+time_embedding_size]
        text_output, _ = self.textencoder(text_seq, None, src_padding_mask, trajs_lens) #[batch_size, seq_len, text_embedding_size]
        output, fusion_vec = self.fusionlayer(st_output, text_output, trajs_lens)
        return output, fusion_vec
    
def NLLcriterion(vocab_size):
    weight = torch.ones(vocab_size)
    weight[constants.PAD] = 0
    criterion = nn.NLLLoss(weight, reduction='sum')
    return criterion


class MultiModal_2vec(nn.Module):
    def __init__(self,spatial_embedding_size, text_embedding_size, nhidden, nhead, nlayer, attn_dropout, pos_droput, device):
        super(MultiModal_2vec, self).__init__()
        self.device = device
        self.encoder_q = Sttext_Traj_Encoder(spatial_embedding_size, text_embedding_size, nhidden,
                                             nhead, nlayer, attn_dropout, pos_droput, device)
        self.encoder_k = Sttext_Traj_Encoder(spatial_embedding_size, text_embedding_size, nhidden,
                                             nhead, nlayer, attn_dropout, pos_droput, device)
        self.st_predictor = nn.Sequential(nn.Linear(text_embedding_size, text_embedding_size//2),
            nn.ReLU(),nn.Linear(text_embedding_size//2, text_embedding_size//2)).to(device) 
        self.text_predictor = nn.Sequential(nn.Linear(text_embedding_size, text_embedding_size//2),
            nn.ReLU(),nn.Linear(text_embedding_size//2, text_embedding_size//2)).to(device) 
        self.clmodel = MoCo(self.encoder_q, self.encoder_k, Config.seq_embedding_dim,
                            Config.moco_proj_dim, Config.moco_nqueue, temperature = Config.moco_temperature).to(device) 
        
    def intra_loss(self, logits, labels):
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        loss = criterion(logits, labels) 
        return loss  
    
    def mocoloss(self, logits, targets):
        return self.clmodel.loss(logits, targets)
    
    def nll_loss(self, vocab_size, target, output):
        target = target.to(self.device)
        criterion = NLLcriterion(vocab_size).to(self.device)
        loss = criterion(output, target)
        return loss
    
    def mseloss(self, output, target):
        target = target.to(self.device)
        loss_fn = nn.MSELoss()
        loss = loss_fn(output, target)
        return loss
    
    def loss_fn(self, logits, labels, inter_logits, inter_targets, spatial_output, tgt_s_vocab, time_output, trg_time_seq, text_output, textvocabs):
        loss1 = self.intra_loss(logits, labels)
        loss2 = self.mocoloss(inter_logits, inter_targets)
        loss3 = self.nll_loss(Config.vocab_size, tgt_s_vocab, spatial_output)
        loss4 = self.nll_loss(Config.textvocab_size, textvocabs, text_output)
        loss5 = self.mseloss(time_output, trg_time_seq)
        loss6 = loss3 + loss4 + loss5
        return Config.SOLVER.eta1*loss1+Config.SOLVER.eta2*loss2+Config.SOLVER.eta3*loss6, loss1, loss2, loss3, loss4, loss5
    
    def loss_fn_interintra(self, logits, labels, inter_logits, inter_targets):
        loss1 = self.intra_loss(logits, labels)
        loss2 = self.mocoloss(inter_logits, inter_targets)
        return loss1 + loss2, loss1, loss2
    
    def interpret(self, trajs_spatial_seq, trajs_time_seq, trajs_text_seq, trajs_lens):
        _, traj_embs = self.encoder_q(**{'spatial_seq': trajs_spatial_seq, 'time_seq': trajs_time_seq, 'text_seq': trajs_text_seq, 'trajs_lens': trajs_lens})
        return traj_embs #[batch_size, seq_embedding_dim]
    
    def forward(self, trgtraj_spatial_seq, trgtraj_time_seq, trgtraj_text_seq, trgtrajs_lens,
                trajs1_spatial_seq, trajs1_time_seq, trajs1_text_seq, trajs1_lens, 
                trajs2_spatial_seq, trajs2_time_seq, trajs2_text_seq, trajs2_lens):
        
        inter_logits, inter_targets = self.clmodel({'spatial_seq': trajs1_spatial_seq, 'time_seq': trajs1_time_seq, 'text_seq': trajs1_text_seq, 'trajs_lens': trajs1_lens},  
                {'spatial_seq': trajs2_spatial_seq, 'time_seq': trajs2_time_seq, 'text_seq': trajs2_text_seq, 'trajs_lens': trajs2_lens})
        
        max_trgtrajs_len = trgtrajs_lens.max().item()    
        src_padding_mask = torch.arange(max_trgtrajs_len)[None, :] >= trgtrajs_lens[:, None] 
        src_padding_mask = src_padding_mask.to(self.device)
        _, st_vec = self.encoder_q.stencoder(trgtraj_spatial_seq, trgtraj_time_seq, None, src_padding_mask, trgtrajs_lens) #[batch_size, spatial_embedding_size+time_embedding_size]
        _, text_vec = self.encoder_q.textencoder(trgtraj_text_seq, None, src_padding_mask, trgtrajs_lens) #[batch_size, text_embedding_size]
        st_feature = self.st_predictor(st_vec)
        text_feature = self.text_predictor(text_vec)
        
        pos_neg_samples = torch.stack((st_feature, text_feature), dim=1)
        pos_neg_samples = pos_neg_samples.view(-1, st_feature.shape[-1]) #[2*batch_size, spatial_embedding_size]
        
        batch_size = st_feature.shape[0]
        labels = torch.cat([torch.tensor([i]*2) for i in range(batch_size)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)
        features = F.normalize(pos_neg_samples, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)       
        
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)
        logits = logits / Config.temperature
        return logits, labels, inter_logits, inter_targets 