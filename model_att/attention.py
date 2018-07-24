import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class Attention(nn.Module):
    def __init__(self, input_size, attention_size, use_cuda):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.attention_size = attention_size
        self.linear_att = nn.Linear(self.input_size*2, self.attention_size, bias=True)
        self.u = Parameter(torch.randn(attention_size, 1))
        self.use_cuda = use_cuda

    def forward(self, h_i, h_t):
        # h_i: variable (len, two_hidden_size)
        # h_t: variable (1, two_hidden_size)
        m_combine = torch.cat([h_i, torch.cat([h_t for _ in range(h_i.size(0))], 0)], 1)
        # m_combine: variable (len, 2*two_hidden_size)
        m_combine = F.tanh(self.linear_att(m_combine))
        # m_combine: variable (len, attention_size)
        # self.u: variable (attention_size, 1)
        beta = torch.mm(m_combine, self.u)  # beta: variable (len, 1)
        beta = torch.t(beta)  # beta: variable (1, len)

        if self.use_cuda:
            alpha = F.softmax(beta, dim=1)  # alpha: variable (1, len)
        else:
            alpha = F.softmax(beta)         # alpha: variable (1, len)
        s = torch.mm(alpha, h_i)
        # alpha: variable (1, len), h_i: variable (len, two_hidden_size)
        # s: variable (1, two_hidden_size)
        return s