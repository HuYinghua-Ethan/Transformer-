import torch
import torch.nn as nn
import math

# 嵌入表示层



"""
self.register_buffer('pe', pe) 和 self.pe = pe 在功能上是有区别的，它们的作用和效果不同：
self.register_buffer('pe', pe)
1.持久化存储：register_buffer 会将 pe 作为模型的缓冲区注册，这样在调用 model.state_dict() 时，pe 会被包含在内。这意味着，你可以在保存和加载模型时，它的值会被持久化。

2.非可训练参数：注册的缓冲区不会被视为模型参数，也就是说它不参与优化，因此不会被更新（不计算梯度）。它在反向传播中不会计算梯度，也不能通过优化过程进行更新。

3.设备管理：通过 register_buffer 注册的缓冲区会随模型自动迁移到 GPU 或 CPU，这样你在调用 .to(device) 或 .cuda() 时，不需要手动迁移这个缓冲区。

self.pe = pe
1.普通属性：简单地将 pe 赋值给 self.pe 只是在模型中创建了一个普通的属性。这不会影响其在 state_dict() 中的存在，除非你自己处理保存和加载。

2.仍可训练：如果你没有特别指明它是非可训练的，self.pe 会被视为一个普通的属性，可以是可训练的（如果后续代码将其卷积更新）。

3.不自动管理设备：使用普通属性时，需要手动确保在 GPU 和 CPU 之间迁移时保持一致性。
总结
使用 register_buffer：你通常会在需要保持模型中常数或固定值的时候使用它，例如位置编码、一些常量矩阵等。
使用普通属性：一般用于存储状态或需要动态更新的变量。
所以，在你的情况下，使用 register_buffer 是更合适的选择，因为你希望注册一个固定的、在模型训练过程中不需要更新的值。
"""


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super(PositionalEncoder, self).__init__()
        self.embedding = nn.Embedding(10, d_model, padding_idx=0)
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos][i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos][i + 1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))
                
        pe = pe.unsqueeze(0)
        # print(pe)
        # print(pe.shape) # torch.Size([1, 80, 512])
        # input()
        self.register_buffer('pe',pe)

    def forward(self, x):
        x = self.embedding(x)    # x.shape : batch_size, seq_len, d_model
        # print(x.shape)  # torch.Size([1, 4, 512])
        # input()
        # 使得单词嵌入表示相对大一些
        x = x * math.sqrt(self.d_model)
        # 增加位置常量到单词嵌入表示中
        seq_len = x.size(1)
        # print(self.pe)
        # print(self.pe[:,:seq_len].shape) # torch.Size([1, 4, 512])
        # input()
        # 这里的 requires_grad=False 表示在反向传播中不计算该变量的梯度，因为位置编码是固定的，不需要更新。
        x = (x + torch.tensor(self.pe[:,:seq_len], requires_grad=False)).cuda()
        # print(x.shape)  # torch.Size([1, 4, 512])
        # input()
        return x



if __name__ == "__main__":
    x = [2, 4, 7, 8]
    x = torch.tensor([x])
    model = PositionalEncoder(d_model=512, max_seq_len=80)
    x = model.forward(x)
    print(x)
    
    

