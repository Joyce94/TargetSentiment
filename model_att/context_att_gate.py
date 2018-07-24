import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model_att.attention import Attention


class Context_att_gate(nn.Module):
    def __init__(self, config, params):
        super(Context_att_gate, self).__init__()
        self.word_num = params.word_num
        self.label_num = params.label_num
        self.char_num = params.char_num

        self.id2word = params.word_alphabet.id2word
        self.word2id = params.word_alphabet.word2id
        self.padID = params.word_alphabet.word2id['<pad>']
        self.unkID = params.word_alphabet.word2id['<unk>']

        self.use_cuda = params.use_cuda
        self.add_char = params.add_char
        self.static = params.static

        self.feature_count = config.shrink_feature_thresholds
        self.word_dims = config.word_dims
        self.char_dims = config.char_dims

        self.lstm_hiddens = config.lstm_hiddens
        self.attention_size = config.attention_size

        self.dropout_emb = nn.Dropout(p=config.dropout_emb)
        self.dropout_lstm = nn.Dropout(p=config.dropout_lstm)

        self.lstm_layers = config.lstm_layers
        self.batch_size = config.train_batch_size

        self.embedding = nn.Embedding(self.word_num, self.word_dims)
        self.embedding.weight.requires_grad = True
        if self.static:
            self.embedding_static = nn.Embedding(self.word_num, self.word_dims)
            self.embedding_static.weight.requires_grad = False

        if params.pretrain_word_embedding is not None:
            # pretrain_weight = np.array(params.pretrain_word_embedding)
            # self.embedding.weight.data.copy_(torch.from_numpy(pretrain_weight))
            # pretrain_weight = np.array(params.pretrain_embed)
            pretrain_weight = torch.FloatTensor(params.pretrain_word_embedding)
            self.embedding.weight.data.copy_(pretrain_weight)

        # for id in range(self.word_dims):
        #     self.embedding.weight.data[self.eofID][id] = 0

        if params.static:
            self.lstm = nn.LSTM(self.word_dims * 2, self.lstm_hiddens // 2, num_layers=self.lstm_layers, bidirectional=True, dropout=config.dropout_lstm)
        else:
            self.lstm = nn.LSTM(self.word_dims, self.lstm_hiddens // 2, num_layers=self.lstm_layers, bidirectional=True, dropout=config.dropout_lstm)

        self.hidden2label = nn.Linear(self.lstm_hiddens, self.label_num)

        self.hidden = self.init_hidden(self.batch_size, self.lstm_layers)

        self.attention = Attention(self.lstm_hiddens, self.attention_size, self.use_cuda)
        self.attention_l = Attention(self.lstm_hiddens, self.attention_size, self.use_cuda)
        self.attention_r = Attention(self.lstm_hiddens, self.attention_size, self.use_cuda)

        self.w1 = Parameter(torch.randn(self.lstm_hiddens, self.lstm_hiddens))
        self.w2 = Parameter(torch.randn(self.lstm_hiddens, self.lstm_hiddens))
        self.w3 = Parameter(torch.randn(self.lstm_hiddens, self.lstm_hiddens))
        self.u1 = Parameter(torch.randn(self.lstm_hiddens, self.lstm_hiddens))
        self.u2 = Parameter(torch.randn(self.lstm_hiddens, self.lstm_hiddens))
        self.u3 = Parameter(torch.randn(self.lstm_hiddens, self.lstm_hiddens))
        self.b1 = Parameter(torch.randn(self.lstm_hiddens, 1))
        self.b2 = Parameter(torch.randn(self.lstm_hiddens, 1))
        self.b3 = Parameter(torch.randn(self.lstm_hiddens, 1))

        nn.init.xavier_uniform(self.w1)
        nn.init.xavier_uniform(self.w2)
        nn.init.xavier_uniform(self.w3)
        nn.init.xavier_uniform(self.u1)
        nn.init.xavier_uniform(self.u2)
        nn.init.xavier_uniform(self.u3)
        nn.init.xavier_uniform(self.b1)
        nn.init.xavier_uniform(self.b2)
        nn.init.xavier_uniform(self.b3)

        self.linear_2 = nn.Linear(self.lstm_hiddens, self.label_num, bias=True)
        nn.init.xavier_uniform(self.linear_2.weight)


    def init_hidden(self, batch_size, num_layers):
        if self.use_cuda:
            return (Variable(torch.zeros(2 * num_layers, batch_size, self.lstm_hiddens // 2)).cuda(),
                     Variable(torch.zeros(2 * num_layers, batch_size, self.lstm_hiddens // 2)).cuda())
        else:
            return (Variable(torch.zeros(2 * num_layers, batch_size, self.lstm_hiddens // 2)),
                     Variable(torch.zeros(2 * num_layers, batch_size, self.lstm_hiddens // 2)))


    def forward(self, fea_v, length, target_start, target_end):
        if self.add_char:
            word_v = fea_v[0]
            char_v = fea_v[1]
        else: word_v = fea_v
        batch_size = word_v.size(0)
        seq_length = word_v.size(1)

        word_emb = self.embedding(word_v)
        word_emb = self.dropout_emb(word_emb)
        if self.static:
            word_static = self.embedding_static(word_v)
            word_static = self.dropout_emb(word_static)
            word_emb = torch.cat([word_emb, word_static], 2)

        x = torch.transpose(word_emb, 0, 1)
        packed_words = pack_padded_sequence(x, length)
        lstm_out, self.hidden = self.lstm(packed_words, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        ##### lstm_out: (seq_len, batch_size, hidden_size)
        lstm_out = self.dropout_lstm(lstm_out)

        ##### no_batch
        x = torch.squeeze(lstm_out, 1)
        # x: variable (seq_len, hidden_size)
        # print(target_start)
        start = target_start.data.tolist()[0]
        end = target_end.data.tolist()[0]

        if self.use_cuda:
            indices_left = None
            if start != 0:
                indices_left = Variable(torch.cuda.LongTensor([i for i in range(0, start)]))
            indices_target = Variable(torch.cuda.LongTensor([i for i in range(start, end + 1)]))
            indices_right = None
            if end != x.size(0) - 1:
                indices_right = Variable(torch.cuda.LongTensor([i for i in range(end + 1, x.size(0))]))
        else:
            indices_left = None
            if start != 0:
                indices_left = Variable(torch.LongTensor([i for i in range(0, start)]))
            indices_target = Variable(torch.LongTensor([i for i in range(start, end + 1)]))
            indices_right = None
            if end != x.size(0) - 1:
                indices_right = Variable(torch.LongTensor([i for i in range(end + 1, x.size(0))]))

        left = None
        if indices_left is not None:
            left = torch.index_select(x, 0, indices_left)       # left: variable (left_len, two_hidden_size)
        target = torch.index_select(x, 0, indices_target)       # target: variable (target_len, two_hidden_size)
        average_target = torch.mean(target, 0)
        average_target = torch.unsqueeze(average_target, 0)     # average_target: variable (1, two_hidden_size)
        right = None
        if indices_right is not None:
            right = torch.index_select(x, 0, indices_right)     # right: variable (right_len, two_hidden_size)

        s = self.attention(x, average_target)
        s_l = None
        if left is not None:
            s_l = self.attention_l(left, average_target)
        s_r = None
        if right is not None:
            s_r = self.attention_r(right, average_target)

        z = torch.exp(torch.mm(self.w1, torch.t(s)) + torch.mm(self.u1, torch.t(average_target)) + self.b1)
        z_all = z
        if s_l is not None:
            z_l = torch.exp(torch.mm(self.w2, torch.t(s_l)) + torch.mm(self.u2, torch.t(average_target)) + self.b2)
            z_all = torch.cat([z_all, z_l], 1)
        if s_r is not None:
            z_r = torch.exp(torch.mm(self.w3, torch.t(s_r)) + torch.mm(self.u3, torch.t(average_target)) + self.b3)
            z_all = torch.cat([z_all, z_r], 1)

        # z_all: variable (two_hidden_size, label_num)

        if self.use_cuda:
            z_all = F.softmax(z_all, dim=1)
        else:
            z_all = F.softmax(z_all)
        z_all = torch.t(z_all)
        z = torch.unsqueeze(z_all[0], 0)
        if left is not None:
            z_l = torch.unsqueeze(z_all[1], 0)
            if right is not None:
                z_r = torch.unsqueeze(z_all[2], 0)
        else:
            if right is not None:
                z_r = torch.unsqueeze(z_all[1], 0)

        ss = torch.mul(z, s)
        if left is not None:
             ss = torch.add(ss, torch.mul(z_l, s_l))
        if right is not None:
             ss = torch.add(ss, torch.mul(z_r, s_r))

        logit = self.linear_2(ss)
        return logit











