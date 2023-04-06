import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModel, get_linear_schedule_with_warmup, AutoConfig
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F 
import os
import sys 


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
ROOT_PATH = os.path.abspath(os.curdir)
PHOBERT_CONFIG = AutoConfig.from_pretrained("vinai/phobert-base")
PHOBERT = AutoModel.from_pretrained("vinai/phobert-base", PHOBERT_CONFIG)


def print_progress(index, total, fi="", last=""):
    percent = ("{0:.1f}").format(100 * ((index) / total))
    fill = int(50 * (index / total))
    spec_char = ["\u001b[38;5;255m╺\u001b[0m", "\u001b[38;5;198m━\u001b[0m", "\u001b[38;5;234m━\u001b[0m"]
    bar = spec_char[1]*(fill-1) + spec_char[0] + spec_char[2]*(50-fill)
    if fill == 50:
        bar = spec_char[1]*fill
        
    percent = " "*(5-len(str(percent))) + str(percent)
    
    if index == total:
        print(f"{fi} {bar} {percent}% {last}")
    else:
        print(f"{fi} {bar} {percent}% {last}", end="\r")


class EmbeddingModel(nn.Module):
    
    def __init__(self):
        super(EmbeddingModel, self).__init__()
        self.phobert = PHOBERT
        self.__freeze_model()
        
        
    def __freeze_model(self):
        for param in self.phobert.parameters():
            param.requires_grad = False
        
    
    def forward(self, X, X_mask):
        _, y = self.phobert(input_ids=X, attention_mask=X_mask, return_dict=False)
        return y


class Embedder:
    
    def __init__(self, device):

        self.device = device
        self.model = EmbeddingModel().to(self.device)


    def forward(self, X):
        self.model.eval()
        X = X.to(self.device)
        X_mask = torch.where(X != 1, 1, 0).to(self.device)
        predict = self.model(X, X_mask)
        return predict


if __name__ == "__main__":
    train_data = torch.load(ROOT_PATH + "/dataset/train_data_v3.pt")
    train_label = torch.load(ROOT_PATH + "/dataset/train_label_v3.pt")
    valid_data = torch.load(ROOT_PATH + "/dataset/valid_data_v3.pt")
    valid_label = torch.load(ROOT_PATH + "/dataset/valid_label_v3.pt")

    train_loader = DataLoader(
        TensorDataset(train_data, train_label),
        batch_size=64,
        shuffle=True,
        drop_last=False
    )
    valid_loader = DataLoader(
        TensorDataset(valid_data, valid_label),
        batch_size=64,
        shuffle=True,
        drop_last=False
    )    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    emb = Embedder(device)
    train_tensor = torch.Tensor().to(device)
    train_label_tensor = torch.Tensor().to(device)
    for idx, (X, y) in enumerate(train_loader):
        print_progress(idx+1, len(train_loader), fi="Train embedding")
        out = emb.forward(X)
        y = y.to(device)
        train_tensor = torch.cat((train_tensor, out), dim = 0)
        train_label_tensor = torch.cat((train_label_tensor, y), dim = 0)

    valid_tensor = torch.Tensor().to(device)
    valid_label_tensor = torch.Tensor().to(device)
    for idx, (X, y) in enumerate(valid_loader):
        print_progress(idx+1, len(valid_loader), fi ="Valid Embedding")
        out = emb.forward(X)
        y = y.to(device)
        valid_tensor = torch.cat((valid_tensor, out), dim = 0)
        valid_label_tensor = torch.cat((valid_label_tensor, y), dim = 0)

    torch.save(train_tensor, ROOT_PATH+"/dataset/train_embedding_v3.pt")
    torch.save(train_label_tensor, ROOT_PATH+"/dataset/train_label_embedding_v3.pt")
    torch.save(valid_tensor, ROOT_PATH+"/dataset/valid_embedding_v3.pt")
    torch.save(valid_label_tensor, ROOT_PATH+"/dataset/valid_label_embedding_v3.pt")