"""
https://blog.csdn.net/weixin_53598445/article/details/125009686?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166574963816782388033881%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166574963816782388033881&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-4-125009686-null-null.142^v56^control,201^v3^control_1&utm_term=%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6&spm=1018.2226.3001.4187
测试：
features = torch.rand((32, 20, 10))      # --> x
attention = selfAttention(2, 10, 20)    # --> self, num_attention_heads, input_size, hidden_size
result = attention.forward(features)    # --> 
print(result.shape)
结果：
torch.size([32, 30, 30])
"""
import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F


class selfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(selfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[: -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim=-1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[: -2] + (self.all_head_size,)
        context = context.view(*new_size)
        return context


if __name__ == "__main__":
    features = torch.rand((32, 20, 10))     # --> x
    attention = selfAttention(2, 10, 20)    # --> self, num_attention_heads, input_size, hidden_size
    result = attention.forward(features)    # -->
    print(result.shape)
