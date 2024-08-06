import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.backends.cudnn
from config import Config, parse
from torch.utils.data.dataloader import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
import time
from tqdm import tqdm
from collections import defaultdict


class TextDataset(Dataset): 
    def __init__(self, data):
        # data: DataFrame
        self.data = data
          
    def __getitem__(self, i):
       #返回的必须是tensor/array/list等数据类型，整个数据集中的数据长度是相等的 
        return self.data[i]

    def __len__(self):
        return len(self.data)
    
def collate_and_augment(text_embeddings):
    # 对短语进行分词和编码
    embedding_seq = []
    for i in range(len(text_embeddings)):
        embedding_seq.append(torch.tensor(text_embeddings[i])) 
    embedding_seq = pad_sequence(embedding_seq, batch_first = True)
    # print('embedding_seq.shape----------', embedding_seq.shape)
    return embedding_seq

class Text2VecConvert:
    def __init__(self, model_path="d2vec_checkpoints/epoch_90.pt"):
        state_dict = torch.load(model_path, map_location='cpu')
        self.model = Text2Vec(k=Config.text_embedding_size)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        # print(2)
    
    def __call__(self, x):
        with torch.no_grad(): 
            return self.model.encode(torch.Tensor(x).unsqueeze(0)).squeeze(0).cpu()
    
class Text2Vec(nn.Module):
    def __init__(self, k=Config.text_embedding_size):
        super(Text2Vec, self).__init__()
  
        self.fc1 = nn.Linear(768, 2*k)
        self.fc2 = nn.ReLU()
        
        self.d2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(2*k, k)
        self.fc4 = nn.Linear(k, 768)
        
        self.fc5 = torch.nn.Linear(768, 768)

    def forward(self, x):
        out = self.fc1(x)
        # print('-------------out1.shape:------------- ', out.shape)
        out = self.d2(self.fc2(out))
        # print('-------------out.shape:------------- ', out.shape)
        
        out = self.fc3(out)
        # print('-------------fc3: out.shape:------------- ', out.shape)
        out = self.fc4(out)
        # print('-------------fc4: out.shape:------------- ', out.shape)
        out = self.fc5(out)
        # print('-------------fc5: out.shape:------------- ', out.shape)
        return out

    def encode(self, x):
        out1 = self.fc1(x)
        # out2 = self.fc2(out1)
        # out = self.fc3(out2)
        out = self.fc3(out1)
        return out
    
    # def loss(self, out, x):
    #     loss = nn.MSELoss(out,x)
    #     return loss

def train_text2Vec(device, texts):
    model = Text2Vec()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.SOLVER.BASE_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SOLVER.LR_STEP, gamma=Config.SOLVER.LR_GAMMA)
    criterion = nn.MSELoss()
    #获取训练数据

    train_dataset = TextDataset(texts)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_and_augment)
    # print('------------train_dataloader-------------',train_dataloader)
    model = model.to(device)
    model.train()
    epoch_train_loss_best = 10000000.0
    epoch_best = 0
    total_epoch = 100
    epoch_patience = 30
    epoch_worse_count = 0
    print("Training text embedding...")
    for i in range(total_epoch):
        total_loss = 0
        time_ep = time.time()
        for input in tqdm(train_dataloader): #遍历数据集中的每个batch
            batch_size = len(input)
            input = input.to(device)
            # print('input.shape: ', input.shape)
            output = model(input)
            # print('-----------output.shape-------------', output.shape)
            loss = criterion(output.to(device), input.to(device))
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
            torch.save(model.state_dict(), f"embed_64_text2vec_checkpoints/epoch_{i}_total_loss_{total_loss}.pt")
            
        else:
            epoch_worse_count += 1
            if epoch_worse_count >= epoch_patience:
                break
        
    print("[time2vec], best_ep={}".format(epoch_best))

@torch.no_grad()
def get_bert_embeddings(device, texts):
    # 加载预训练的BERT模型和分词器
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)
    bert_model = bert_model.to(device)

    output_vecs = defaultdict(list)

    for i in range(len(texts)):  
        token_ids = []
        attention_masks = []
        tokens = tokenizer.tokenize(texts[i])
        token_ids.append(tokenizer.convert_tokens_to_ids(tokens))
        attention_masks.append([1] * len(tokens))
        # 对序列进行填充和掩码处理
        max_length = max(len(ids) for ids in token_ids)
        padded_token_ids = [ids + [0] * (max_length - len(ids)) for ids in token_ids]
        attention_masks = [masks + [0] * (max_length - len(masks)) for masks in attention_masks]
        # 转换为张量
        input_ids = torch.tensor(padded_token_ids)
        attention_masks = torch.tensor(attention_masks)
        sequence_lengths = torch.tensor([len(ids) for ids in token_ids]).unsqueeze(-1).float()
        sequence_lengths = sequence_lengths.to(device)
        # 获取BERT模型的输出
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        # print(i, texts[i])
        outputs = bert_model(input_ids, attention_mask=attention_masks)
        # print('--------texts i-------:', i)
        # 获取最后一层的隐藏状态作为嵌入
        last_hidden_state = outputs.last_hidden_state
        # print('----last_hidden_state----', last_hidden_state.shape)
        # 使用掩码矩阵计算平均池化向量
        mask = attention_masks.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked_hidden_state = last_hidden_state * mask
        summed_hidden_state = torch.sum(masked_hidden_state, dim=1)
        average_hidden_state = summed_hidden_state / sequence_lengths
        # print('----average_hidden_state----', average_hidden_state.shape)
        output_vecs[texts[i]] = average_hidden_state.squeeze().tolist()
    # print('len(outputs): ',len(output_vecs))
    # print('outputs[0]: ', output_vecs[0].shape)
    return output_vecs         

if __name__ == "__main__":
    args = parse.get_args()
    Config.merge_from_file(args.config_file) 
    dataset = os.path.join('data', Config.DATASETS.dataset)
    
    device = torch.device('cpu')
    if Config.SOLVER.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True

    data = pickle.load(open(dataset, 'rb'), encoding='bytes')
    
    textdata = data["keyword_seqs"]
    keywords = []
    for i in range(len(textdata)):
        for j in range(len(textdata[i])):
            keywords.append(textdata[i][j])
    keywords = list(set(keywords))
    output_vecs = get_bert_embeddings(device, keywords)
    print('---------------len(output_vecs)-----------', len(output_vecs))
    # first_pair = next(iter(output_vecs.items()))
    # print(first_pair)
    
    pickle.dump(output_vecs, open('data/test_output_vecs', 'wb'), protocol=2)
    # print('--------output_vecs-----------', len(output_vecs))
    # print(len(output_vecs[0]))
    output_vecs = pickle.load(open('data/test_output_vecs', 'rb'), encoding='bytes')
    train_text2Vec(device, output_vecs)
