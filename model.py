import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch
from model_utils import SupConLoss

def custom_replace(tensor, on_zero, on_non_zero):
    # 1 is unethical, zero is ethical
    res = torch.randn(len(tensor), len(on_zero), device="cuda:3")
    for i in range(len(tensor)):
        if tensor[i] == 0:
            res[i] = on_zero 
        else:
            res[i] = on_non_zero
    return res

class BertEthicsFinetuner(pl.LightningModule):

    def __init__(self, lr=None):
        super().__init__()
        # self.bert = SentenceTransformer('distilbert-base-nli-mean-tokens')
        self.bert = SentenceTransformer('/data5/shashank2000/distilbert', device='cuda')
        # freeze this and only learn the good bad stuff
        for param in self.bert.parameters():
            param.requires_grad = False        
        
        self.input_dim = 768
        self.hidden_dim = 500
        self.output_dim = 100
        # self.model = nn.Sequential(
        #     nn.Linear(self.input_dim, self.hidden_dim),
        #     nn.BatchNorm1d(self.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_dim, self.output_dim))
        self.lin1 = nn.Linear(768, 100)
        self.good = torch.randn(100, device="cuda:3")
        self.bad = torch.randn(100, device="cuda:3")
        # insert margin value?
        self.loss = nn.CosineEmbeddingLoss()
        self.lr = lr
        self.loss_vec = torch.ones(16, device="cuda:3") # batch size
        self.one_elem = -torch.ones(1, device="cuda:3")
        
    def forward(self, x):
        # pass through BERT + linear layer + projection head
        x = self.bert.encode(x, convert_to_tensor=True).cuda()
        x = self.model(x)
        return x
        
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train_loss', loss)
        return loss

    def shared_step(self, batch, train=True):
        sentence, label = batch
        s_vec = self(sentence)
        tlabel = custom_replace(label, self.good, self.bad)
        loss = self.loss(s_vec, tlabel, self.loss_vec)
        loss += self.loss(self.good.unsqueeze(0), self.bad.unsqueeze(0), self.one_elem) # minimizing the similarity leads to minimum loss 
        bad_labels = custom_replace(label, self.bad, self.good)
        loss += self.loss(s_vec, bad_labels, -1*self.loss_vec) # minimizing the similarity leads to minimum loss
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)