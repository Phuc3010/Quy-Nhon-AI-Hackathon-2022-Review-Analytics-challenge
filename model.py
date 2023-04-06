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


class ClassifyModel(nn.Module):
    
    def __init__(self):
        super(ClassifyModel, self).__init__()

        self.cls_layer1 = nn.Sequential(
            nn.Linear(768, 1, bias=True),
            nn.Sigmoid()
        )
        self.cls_layer2 = nn.Sequential(
            nn.Linear(768, 1, bias=True),
            nn.Sigmoid()
        )
        self.cls_layer3 = nn.Sequential(
            nn.Linear(768, 1, bias=True),
            nn.Sigmoid()
        )
        self.cls_layer4 = nn.Sequential(
            nn.Linear(768, 1, bias=True),
            nn.Sigmoid()
        )
        self.cls_layer5 = nn.Sequential(
            nn.Linear(768, 1, bias=True),
            nn.Sigmoid()
        )
        self.cls_layer6 = nn.Sequential(
            nn.Linear(768, 1, bias=True),
            nn.Sigmoid()
        )
        self.reg_layer1 = nn.Sequential(
            nn.Linear(768, 5, bias=True),
            nn.Softmax(dim = 1)
        )
        self.reg_layer2 = nn.Sequential(
            nn.Linear(768, 5, bias=True),
            nn.Softmax(dim = 1)
        )
        self.reg_layer3 = nn.Sequential(
            nn.Linear(768, 5, bias=True),
            nn.Softmax(dim = 1)
        )
        self.reg_layer4 = nn.Sequential(
            nn.Linear(768, 5, bias=True),
            nn.Softmax(dim = 1)
        )
        self.reg_layer5 = nn.Sequential(
            nn.Linear(768, 5, bias=True),
            nn.Softmax(dim = 1)
        )
        self.reg_layer6 = nn.Sequential(
            nn.Linear(768, 5, bias=True),
            nn.Softmax(dim = 1)
        )
        
    
    def forward(self, X):
        y_cls1 = torch.where(self.cls_layer1(X) >= 0.5, 1, 0)
        y_cls2 = torch.where(self.cls_layer2(X) >= 0.5, 1, 0)
        y_cls3 = torch.where(self.cls_layer3(X) >= 0.5, 1, 0)
        y_cls4 = torch.where(self.cls_layer4(X) >= 0.5, 1, 0)
        y_cls5 = torch.where(self.cls_layer5(X) >= 0.5, 1, 0)
        y_cls6 = torch.where(self.cls_layer6(X) >= 0.5, 1, 0)

        y_reg1 = self.reg_layer1(X).argmax(dim=1, keepdim=True) + 1
        y_reg2 = self.reg_layer2(X).argmax(dim=1, keepdim=True) + 1
        y_reg3 = self.reg_layer3(X).argmax(dim=1, keepdim=True) + 1
        y_reg4 = self.reg_layer4(X).argmax(dim=1, keepdim=True) + 1
        y_reg5 = self.reg_layer5(X).argmax(dim=1, keepdim=True) + 1
        y_reg6 = self.reg_layer6(X).argmax(dim=1, keepdim=True) + 1

        y1 = y_cls1*y_reg1
        y2 = y_cls2*y_reg2 
        y3 = y_cls3*y_reg3
        y4 = y_cls4*y_reg4 
        y5 = y_cls5*y_reg5
        y6 = y_cls6*y_reg6 

        output = torch.cat((y1,y2,y3,y4,y5,y6), dim = 1).float().requires_grad_(True)
        return output


class Trainer:
    
    def __init__(self, train_loader, valid_loader, n_classes):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_classes = n_classes
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.epochs = 100
        
        self.model = ClassifyModel().to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=0, 
            num_training_steps=len(train_loader)*self.epochs
        )
        # self.criterion = F.cross_entropy
        self.criterion = nn.MSELoss()
        self.train_loss = []
        self.valid_loss = []


    def load_model(self, path):
        if path[0] != "/":
            path = "/" + path 
        model_state_dict = torch.load(ROOT_PATH+path)
        self.model.load_state_dict(model_state_dict)
        
    
    def save_model(self, path):
        if path[0] != "/":
            path = "/" + path
        torch.save(self.model.state_dict(), ROOT_PATH+path)


    def accuracy_calc(self, y_pred, y_true):
        pred_mask_f1 = torch.where(y_pred == 0.0, 0, 1).long()
        true_mask_f1 = torch.where(y_true == 0.0, 0, 1).long()
        final_score = 0
        for col in range(6):
            tp = sum([1 for idx in range(y_pred.size(0)) if pred_mask_f1[idx][col]==1 and pred_mask_f1[idx][col]==true_mask_f1[idx][col]])
            if tp == 0:
                continue
            precision_denom = pred_mask_f1[:, col].sum().item()
            recall_denom = true_mask_f1[:, col].sum().item()

            precision = tp/precision_denom
            recall = tp/recall_denom

            f1_score = (2*precision*recall)/(precision+recall)
            rss = ((y_true[:, col] - y_pred[:, col])**2).sum().item()
            k = 16*y_pred.size(0)

            r2_score = 1 - rss/k
            final_score += f1_score*r2_score
        return final_score/6


    def train_step(self):
        self.model.train()
        loss_his = []
        acc_his = []
        for idx, (X_batch, y_batch) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            predict = self.model(X_batch)
            acc = round(self.accuracy_calc(predict, y_batch), 5)
            acc_his.append(acc)
            _loss = self.criterion(predict, y_batch)
            _loss.backward()
            loss_his.append(_loss.item())

            self.optimizer.step()
            self.scheduler.step()

            print_progress(idx+1, len(self.train_loader), last=f"Loss: {round(_loss.item(), 5)} Acc: {acc}", fi=f"Train batch {idx+1}/{len(self.train_loader)}")
            
        loss_his = torch.tensor(loss_his)
        acc_his = torch.tensor(acc_his)
        mean_loss = loss_his.mean(dim = 0)
        mean_acc = acc_his.mean(dim=0)
        self.train_loss.append(mean_loss.item())
        return mean_loss.item(), mean_acc.item()


    def valid_step(self):
        self.model.eval()
        loss_his = []
        acc_his = []
        
        for idx, (X_batch, y_batch) in enumerate(self.valid_loader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            predict = self.model(X_batch)
            acc = round(self.accuracy_calc(predict, y_batch), 5)
            acc_his.append(acc)

            _loss = self.criterion(predict, y_batch)
            loss_his.append(_loss.item())

            print_progress(idx+1, len(self.valid_loader), last=f"Loss: {round(_loss.item(), 5)} Acc: {acc}", fi=f"Valid batch {idx+1}/{len(self.valid_loader)}")
            
        loss_his = torch.tensor(loss_his)
        acc_his = torch.tensor(acc_his)
        mean_loss = loss_his.mean(dim = 0)
        mean_acc = acc_his.mean(dim=0)
        self.valid_loss.append(mean_loss.item())
        return mean_loss.item(), mean_acc.item()


    def predict(self, X):
        self.model.eval()
        X = X.to(self.device)
        predict = self.model(X)
        return predict
        
        
    def fit(self):
        for epoch in range(self.epochs):
            print("="*100)
            print("Epoch:", epoch+1)
            try:
                train_loss, train_acc = self.train_step()
                valid_loss, valid_acc = self.valid_step()
            except KeyboardInterrupt:
                self.save_model("/dataset/model_v3.pt")
                sys.exit()

            print(f"Train loss: {round(train_loss, 5)} - Valid loss: {round(valid_loss, 5)} - Train acc: {round(train_acc, 5)} - Valid acc: {round(valid_acc, 5)}")
            print("="*100)


def get_loader():
    train_data = torch.load(ROOT_PATH + "/dataset/train_embedding_v3.pt")
    train_label = torch.load(ROOT_PATH + "/dataset/train_label_embedding_v3.pt")
    valid_data = torch.load(ROOT_PATH + "/dataset/valid_embedding_v3.pt")
    valid_label = torch.load(ROOT_PATH + "/dataset/valid_label_embedding_v3.pt")

    train_loader = DataLoader(
        TensorDataset(train_data, train_label),
        batch_size=64,
        shuffle=False,
        drop_last=False
    )
    valid_loader = DataLoader(
        TensorDataset(valid_data, valid_label),
        batch_size=64,
        shuffle=False,
        drop_last=False
    )    
    return train_loader, valid_loader 


if __name__ == "__main__":
    train_loader, valid_loader = get_loader()
    trainer = Trainer(train_loader, valid_loader, 6)
    trainer.fit()

    # pred = torch.tensor([[1, 0, 3, 4, 2, 0], [0, 2, 2, 5, 1, 0]])
    # target = torch.tensor([[1, 1, 3, 5, 3, 1], [0, 0, 3, 4, 0, 0]])
    # print(trainer.accuracy_calc(pred, target))
    # trainer.save_model("/dataset/model_v3.pt")
    
    # trainer.load_model("/dataset/model_v3.pt")

    # for idx, (X_batch, y_batch) in enumerate(train_loader):
    #     pred = trainer.predict(X_batch[0].view(1, -1))
    #     print("Prob predict:",pred.size())
    #     print(pred)
    #     print("True label:",y_batch[0])
    #     break