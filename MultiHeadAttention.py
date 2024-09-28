import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 注意力层
"""
torch.matmul() 是 PyTorch 中用于执行矩阵乘法的函数。它可以用于多种情况，包括：

二维张量的矩阵乘法：进行标准的矩阵乘法。
一维和二维张量的乘法：将一维张量视为列向量与二维张量相乘。
多维张量的广播：对于高维张量，torch.matmul() 会在最后两个维度上进行矩阵乘法，并进行广播以匹配其他维度。
"""


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // heads  # bert 768 // 12 = 64
        self.h = heads  # 头数

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # 上下文单词所对应的权重得分，形状是 seq_len, d_model × d_model, seq_len = seq_len, seq_len
        # 掩盖掉那些为了填补长度增加的单元，使其通过 softmax 计算后为 0
        '''
        if mask is not None:这行代码检查 mask 是否为 None，只有在掩码存在的情况下才会进行以下操作。这是为了避免在没有掩码时出现不必要的错误或计算。
        mask = mask.unsqueeze(1): 通过 unsqueeze(1) 增加一个维度，使得掩码的形状从 (batch_size, seq_len) 变为 (batch_size, 1, seq_len)。如此一来，掩码可以在后续计算中与得分矩阵（scores）进行广播，从而更方便的进行元素级的操作。
        scores = scores.masked_fill(mask == 0, -1e9):
        这行代码的作用是将 scores 张量中，掩码为零的位置（即 mask == 0）的值替换为 -1e9。这样做的目的是在计算 softmax 时，将这些位置的得分极大地降低，从而使得 softmax 结果趋向于零，达到忽略填充（padding）部分的效果。
        '''
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        
        # 进行线性操作划分为 h 个头， batch_size, seq_len, d_model -> batch_size, seq_len, h, d_k
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)  
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # 矩阵转置  batch_size, seq_len, h, d_k -> batch_size, h, seq_len, d_k
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # 计算 attention
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)
        
        # 连接多个头并输入到最后的线性层 (bs, h, seq_len, d_k) 转换为 (bs, seq_len, h, d_k)
        # .contiguous() 用于确保内存的连续性，方便后续的操作。
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


