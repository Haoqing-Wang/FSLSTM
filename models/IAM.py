import torch
import torch.nn as nn
import numpy as np

def one_hot(indices, depth):
    """
    Inputs:
       indices:  a (n_batch, m) Tensor or (m) Tensor.
       depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """
    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda() #(n_batch, m, depth) or (m, depth)
    index = indices.view(indices.size()+torch.Size([1])) #(n_batch, m, 1) or (m, 1)
    if len(indices.size())<2:
        encoded_indicies = encoded_indicies.scatter_(1,index,1)
    else:
        encoded_indicies = encoded_indicies.scatter_(2, index, 1)
    return encoded_indicies

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, Y):  #q:(b, sup, d)  k:(b, que, d)  v:(b, que, d)  Y:(b, sup)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)  # (b, sup, que)
        Y_one_hot = one_hot(Y, 5)  # (b, sup, 5)
        attn = torch.bmm(Y_one_hot.transpose(1,2), attn)  # (b, 5, que)
        attn = attn.div(Y_one_hot.transpose(1,2).sum(dim=2, keepdim=True).expand_as(attn))  # (b, 5, que)
        attn = torch.bmm(Y_one_hot, attn)  # (b, sup, que)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output

class InverseAttentionModule(nn.Module):
    ''' Inverse Attention Module '''
    def __init__(self, d_model, reduction=8, dropout=0.1):
        super().__init__()
        self.Q = nn.Sequential(nn.Linear(d_model, d_model//reduction),
                               nn.ReLU(inplace=True),
                               nn.Linear(d_model//reduction, d_model))
        self.K = nn.Sequential(nn.Linear(d_model, d_model//reduction),
                               nn.ReLU(inplace=True),
                               nn.Linear(d_model//reduction, d_model))
        self.V = nn.Sequential(nn.Linear(d_model, d_model//reduction),
                               nn.ReLU(inplace=True),
                               nn.Linear(d_model//reduction, d_model))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_model, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(nn.Linear(d_model, d_model//reduction),
                                nn.ReLU(inplace=True),
                                nn.Linear(d_model//reduction, d_model))
        self.dropout = nn.Dropout(dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
        
    def forward(self, q, k, v, Y):
        residual = q
        q = self.Q(q)
        k = self.K(k)
        v = self.V(v)
        output = self.attention(q, k, v, Y)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output