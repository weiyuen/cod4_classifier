import torch
import pytorch_lightning as pl
import torchmetrics
import torchvision

from torch import nn

class PLModel(pl.LightningModule):
    def __init__(self, lr=0.0001):
        super().__init__()
        self.lr = lr
        self.loss = nn.BCEWithLogitsLoss()
        self.acc = torchmetrics.Accuracy()
        
        self.model = torchvision.models.resnext50_32x4d(pretrained=True)
        self.fc = nn.Linear(2048, 1)
        self.model.fc = self.fc

        
    def forward(self, x):
        x = self.model(x)
        return x
    
                
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self(x).squeeze()
        loss = self.loss(logits, y.float())
        pred = torch.round(torch.sigmoid(logits))
        acc = self.acc(pred, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc, on_step=True, on_epoch=False)
        return loss
    
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self(x).squeeze()
        loss = self.loss(logits, y.float())
        pred = torch.round(torch.sigmoid(logits))
        acc = self.acc(pred, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        
        
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self(x).squeeze()
        loss = self.loss(logits, y.float())
        pred = torch.round(torch.sigmoid(logits))
        acc = self.acc(pred, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        
        
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr = self.lr
        )