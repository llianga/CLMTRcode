# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# https://github.com/facebookresearch/moco

# Modified by: yanchuan

import torch
import torch.nn as nn

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, encoder_q, encoder_k, nemb, nout,
                queue_size, mmt = 0.999, temperature = 0.07):
        super(MoCo, self).__init__()

        self.queue_size = queue_size
        self.mmt = mmt
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss()

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        self.mlp_q = Projector(nemb, nout)
        self.mlp_k = Projector(nemb, nout)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  
            param_k.requires_grad = False  

        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data.copy_(param_q.data)  
            param_k.requires_grad = False 

        # create the queue
        self.register_buffer("queue", torch.randn(nout, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim = 0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.mmt + param_q.data * (1. - self.mmt)
        
        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data = param_k.data * self.mmt + param_q.data * (1. - self.mmt)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            self.queue[:, ptr:self.queue_size] = keys.T[:, 0:self.queue_size-ptr]
            self.queue[:, 0:batch_size-self.queue_size+ptr] = keys.T[:, self.queue_size-ptr:]

        ptr = (ptr + batch_size) % self.queue_size  

        self.queue_ptr[0] = ptr

 
    def forward(self, kwargs_q, kwargs_k): 
        output, rtn = self.encoder_q(**kwargs_q)
        q = self.mlp_q(rtn)  
        q = nn.functional.normalize(q, dim=1) 

        with torch.no_grad():
            self._momentum_update_key_encoder()  
            output, rtn = self.encoder_k(**kwargs_k)
            k = self.mlp_k(rtn) 
            k = nn.functional.normalize(k, dim=1)

       
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) 
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()]) 
        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.temperature
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda() 
        self._dequeue_and_enqueue(k)

        return logits, labels

    def loss(self, logit, target):
        return self.criterion(logit, target)


class Projector(nn.Module):
    def __init__(self, nin, nout):
        super(Projector, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(nin, nin), 
                                        nn.ReLU(), 
                                        nn.Linear(nin, nout))
        self.reset_parameter()

    def forward(self, x):
        return self.mlp(x)

    def reset_parameter(self):
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=1.414)
                torch.nn.init.zeros_(m.bias)
        
        self.mlp.apply(_weights_init)
        


