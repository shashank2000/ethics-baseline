import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch
from model_utils import SupConLoss
from model import BertEthicsFinetuner 

class EthicsEval(pl.LightningModule):

    def __init__(self, lr, pretrained_model_path):
        super().__init__()
        self.model = BertEthicsFinetuner.load_from_checkpoint(pretrained_model_path)
        self.model.freeze()
        self.bert = self.model.bert
        
        # another one
        self.lin1 = nn.Linear(768, 100)
        # gotta make sure this is correct orientation
        self.good = F.normalize(self.model.good, dim=0)
        self.bad = F.normalize(self.model.bad, dim=0)
        # self.good = self.model.good
        # self.bad = self.model.bad
        self.qual_vec = torch.stack((self.good, self.bad)).T

        # self.qual_vec should still be frozen
        self.qual_vec.requires_grad = False
        
        self.lr = lr

        self.train_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy() 
        self.val_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        # pass through BERT + linear layer + projection head
        x = self.bert.encode(x, convert_to_tensor=True).cuda()
        x = self.lin1(x)
        return x

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, 2)
        self.log('test_loss', loss)
        return loss

    # lower bound on train accuracy
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, 0)
        self.log('val_loss', loss)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train_loss', loss)
        return loss

    def shared_step(self, batch, train=1):
        # baseline
        sentence, label = batch
        s_vec = self(sentence)
        label = torch.tensor(label.cuda())
        logits = s_vec @ self.qual_vec # shouldn't I be instead calculating the cosine similarity?
        # divide each entry by the product of the magnitudes
        loss = F.cross_entropy(logits, label)
        
        # can try cosine embedding loss with always increasing similarity
        if train == 1:
            acc = self.train_accuracy(logits, label)
            self.log("train_acc", acc)
        elif train == 2:
            acc = self.test_accuracy(logits, label)
            self.log("test_acc", acc)
        elif train == 0:
            acc = self.val_accuracy(logits, label)
            self.log("val_acc", acc)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)