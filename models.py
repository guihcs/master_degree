import torch
import torch.nn as nn
from transformers import BertModel

import struct


class Finbank:

    def __init__(self, wp, ep):
        super(Finbank, self).__init__()
        self.wp = wp
        self.ep = ep

        with open(self.wp, 'r') as f:
            names = f.readlines()
            self.wm = {w[:-1]: i for i, w in enumerate(names)}

        self.embs = dict()

    def sim(self, w1, w2):
        return torch.cosine_similarity(self.get_emb(w1).unsqueeze(0), self.get_emb(w2).unsqueeze(0)).item()

    def get_emb(self, w1):
        if w1 not in self.wm:
            return torch.zeros(200)
        if self.wm[w1] not in self.embs:

            with open(self.ep, 'rb') as f:
                f.seek(self.wm[w1] * 800)
                data = f.read(800)

                res = []
                for i in range(0, len(data) - 3, 4):
                    res.append(struct.unpack('>f', data[i:i + 4])[0])

                self.embs[self.wm[w1]] = torch.Tensor(res)

        return self.embs[self.wm[w1]]


class Model1(nn.Module):

    def __init__(self, classes):
        super(Model1, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.tanh = nn.Tanh()

        self.dff = nn.Sequential(
            nn.Linear(768, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, a):
        embs = self.bert(input_ids=x, attention_mask=a)['last_hidden_state']
        embs *= a.unsqueeze(2)
        out = embs.sum(dim=1) / a.sum(dim=1, keepdims=True)
        return self.dff(self.tanh(out))


class Model2(nn.Module):

    def __init__(self, glove, vocab_size, classes):
        super(Model2, self).__init__()
        self.glove = glove
        self.emb = nn.Embedding(vocab_size + 1, 64, padding_idx=0)
        self.ln1 = nn.Sequential(
            nn.Linear(300, 64),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(batch_first=True, input_size=64, hidden_size=64, bidirectional=True)
        self.ln2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.dff = nn.Sequential(
            nn.Linear(128, classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, l, x):
        embs1 = self.glove(l)
        pool1 = embs1.sum(dim=1)
        cf = torch.sum(l != 0, dim=1)
        cf[cf == 0] = 1
        pool1 /= cf.unsqueeze(1)

        label = self.ln1(pool1)

        embs2 = self.emb(x)

        out, (hn, cn) = self.lstm(embs2)

        xmask = (x != 0).float().unsqueeze(2)
        pool2 = torch.sum(out * xmask, dim=1)

        cf = torch.sum(xmask, dim=1)
        cf[cf == 0] = 1
        pool2 /= cf
        context = self.ln2(pool2)

        return self.dff(torch.cat([label, context], dim=1))
