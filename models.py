import pytorch_lightning as pl
import torch
from torch import nn
from swin.video_swin_transformer import SwinTransformer3D
import torch.nn.functional as F
from torchmetrics import Accuracy
from collections import OrderedDict
from torchvision.models.video import r2plus1d_18
from torchvision.models import efficientnet_b7, mobilenet_v3_large


class VideoSwin(pl.LightningModule):
    def __init__(self, checkpoint_path, num_classes, label_smoothing, **kwargs):
        super(VideoSwin, self).__init__()
        base_model = SwinTransformer3D(**kwargs)
        
        checkpoint = torch.load(checkpoint_path) # load the checkpoint from the provided path (from the original authors)
        
        new_state_dict = OrderedDict() # change weights names so there is no mismatch
        for k, v in checkpoint['state_dict'].items():
            if 'backbone' in k:
                name = k[9:]
                new_state_dict[name] = v 
                
        base_model.load_state_dict(new_state_dict)
        
        self.label_smoothing = label_smoothing
        # add classification head
        self.model = nn.Sequential(base_model, nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(1024, num_classes))
        self.train_accuracy = Accuracy()
        self.valid_accuracy = Accuracy()
        
    def forward(self, x):
        logits = self.model(x)
        
        return logits
    
    def configure_optimizers(self):        
        return [torch.optim.AdamW(self.parameters(), lr=1e-05)]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        
        y_pred = torch.softmax(logits, 1).argmax(1)
        self.train_accuracy(y_pred, y)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        
        y_pred = torch.softmax(logits, 1).argmax(1)
        self.valid_accuracy(y_pred, y)
        
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", self.valid_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
        return loss
    
    
class R2Plus1D(pl.LightningModule):
    def __init__(self, num_classes, label_smoothing):
        super(R2Plus1D, self).__init__()
        
        self.label_smoothing = label_smoothing
        
        base_model = r2plus1d_18(pretrained=True)
        
        fc_in_features = base_model.fc.in_features
        base_model.fc = torch.nn.Linear(fc_in_features, num_classes)
        
        self.base_model = base_model
        
        self.train_accuracy = Accuracy()
        self.valid_accuracy = Accuracy()
        
    def forward(self, x):
        return self.base_model(x)
    
    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=1e-5)]

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        
        y_pred = torch.softmax(logits, 1).argmax(1)
        self.train_accuracy(y_pred, y)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        
        y_pred = torch.softmax(logits, 1).argmax(1)
        self.valid_accuracy(y_pred, y)
        
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", self.valid_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
        return loss
    
    
class CnnGru(pl.LightningModule):
    def __init__(self, num_classes, label_smoothing, rnn_hidden_size=512, rnn_num_layers=1, dropout_rate=0.2):
        super(CnnGru, self).__init__()
        
        self.label_smoothing = label_smoothing
        
        # initialize both mobilenet and gru and align their features
        self.base_model = mobilenet_v3_large(pretrained=True)
        fc_in_features = self.base_model.classifier[0].in_features
        self.base_model.classifier = torch.nn.Identity()
        
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.rnn = torch.nn.GRU(fc_in_features, rnn_hidden_size, rnn_num_layers, batch_first=True)
        self.fc = torch.nn.Linear(rnn_hidden_size, num_classes)
        
        self.train_accuracy = Accuracy()
        self.valid_accuracy = Accuracy()
        
    def forward(self, x):
        batch_size, c, num_frames, h, w = x.shape
        
        # get mobilenet features for first frame
        i = 0
        y = self.base_model(x[:, :, i])

        # feed them into rnn
        out, hidden_state = self.rnn(y.unsqueeze(1))
        
        # do it for each frame
        for i in range(1, num_frames):
            y = self.base_model(x[:, :, i])
            out, hidden_state = self.rnn(y.unsqueeze(1), hidden_state)
            
        out = self.dropout(out[:, -1])
        out = self.fc(out)
        
        return out
    
    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=1e-5)]

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        
        y_pred = torch.softmax(logits, 1).argmax(1)
        self.train_accuracy(y_pred, y)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        
        y_pred = torch.softmax(logits, 1).argmax(1)
        self.valid_accuracy(y_pred, y)
        
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", self.valid_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
        return loss