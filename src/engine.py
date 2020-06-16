import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


def weighted_nae(inp, targ):
    W = torch.FloatTensor([0.3, 0.175, 0.175, 0.175, 0.175])
    return torch.mean(torch.matmul(torch.abs(inp - targ), W.cuda()/torch.mean(targ, axis=0)))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


def loss_fn(logits, targets):
    loss_fct = RMSELoss()
    loss = loss_fct(logits, targets)
    return loss


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    
    y_true = []
    y_pred = []
    for bi, d in enumerate(tk0):

        features = d["features"].to(device, dtype=torch.float32)
        targets = d["targets"].to(device, dtype=torch.float32).view(-1, 5)

        model.zero_grad()
        outputs = model(features)

        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        # outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        # targets = targets.float().cpu().detach().numpy()
        cv = weighted_nae(targets, outputs)
        y_true.append(targets)
        y_pred.append(outputs)

        losses.update(loss.item(), features.size(0))
        tk0.set_postfix(loss=losses.avg, cv=cv.cpu().detach().numpy())

    y_true_cat = torch.cat(y_true, 0)
    y_pred_cat = torch.cat(y_pred, 0)
    score = weighted_nae(y_true_cat, y_pred_cat)
    print()
    print(f'train score : {score}')


def eval_fn(data_loader, model, device):
    model.eval()
    losses = AverageMeter()
    y_true = []
    y_pred = []
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            features = d["features"].to(device, dtype=torch.float32)
            targets = d["targets"].to(device, dtype=torch.float32).view(-1, 5)
            outputs = model(features)
            loss = loss_fn(outputs, targets)
            # outputs = torch.sigmoid(outputs).cpu().detach().numpy()
            # targets = targets.float().cpu().detach().numpy()
            cv = weighted_nae(targets, outputs)
            y_true.append(targets)
            y_pred.append(outputs)
            losses.update(loss.item(), features.size(0))
            tk0.set_postfix(loss=losses.avg, cv=cv.cpu().detach().numpy())
        y_true_cat = torch.cat(y_true, 0)
        y_pred_cat = torch.cat(y_pred, 0)
        score = weighted_nae(y_true_cat, y_pred_cat)
        print(f'valid score : {score}')
    return score, losses.avg        
