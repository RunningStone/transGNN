"""
Loss functions from different algorithms
"""
import torch
import numpy as np
import torch.nn as nn



#损失函数基于
#https://github.com/TencentAILabHealthcare/MLA-GNN/blob/main/model_GAT_v4.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CoxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, survtime, censor, hazard_pred):
        current_batch_len = survtime.shape[0]
        R_mat = (survtime.unsqueeze(-1) <= survtime).float().to(survtime.device)

        theta = hazard_pred.view(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
        return loss_cox

class RegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model):
        loss_reg = sum(torch.abs(W).sum() for W in model.parameters())
        return loss_reg

class MixedCoxLossWithReg(nn.Module):
    def __init__(self, lambda_cox, lambda_nll, lambda_reg):
        super().__init__()
        self.cox_loss = CoxLoss()
        self.reg_loss = RegLoss()
        self.lambda_cox = lambda_cox
        self.lambda_nll = lambda_nll
        self.lambda_reg = lambda_reg
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, preds, model,grad_batch_labels, surv_batch_labels, censor_batch_labels, ):
        loss_cox = self.cox_loss(surv_batch_labels, censor_batch_labels, preds)
        loss_reg = self.reg_loss(model)
        grad_loss = self.ce_loss(preds, grad_batch_labels)
        loss = self.lambda_cox * loss_cox + self.lambda_nll * grad_loss + self.lambda_reg * loss_reg
        return loss

class CELossWithReg(nn.Module):
    def __init__(self, lambda_nll, lambda_reg):
        super().__init__()
        self.reg_loss = RegLoss()
        self.lambda_nll = lambda_nll
        self.lambda_reg = lambda_reg
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, model, pred, label):
        loss_reg = self.reg_loss(model)
        loss_grad = self.ce_loss(pred, label)
        loss = self.lambda_nll * loss_grad + self.lambda_reg * loss_reg
        return loss





def reg_loss_fn(model):
    for W in model.parameters():
        loss_reg = torch.abs(W).sum()
    return loss_reg

def Mixed_CoxLoss_with_reg_fn(
                        # for survival loss
                        surv_batch_labels,censor_batch_labels,
                        # grad prediction and labels
                        grad_batch_labels,preds,
                        # model for loss reg
                        model,
                        #-----------> parameters
                        lambda_cox,lambda_nll,lambda_reg):
    loss_cox = CoxLoss(surv_batch_labels, censor_batch_labels, preds) 
    loss_reg = reg_loss_fn(model)
    loss_func = nn.CrossEntropyLoss()
    grad_loss = loss_func(preds, grad_batch_labels)
    loss = lambda_cox * loss_cox + lambda_nll * grad_loss + lambda_reg * loss_reg
    return loss

def CELoss_with_reg_fn(model,pred,label,
                    lambda_nll,lambda_reg):
    loss_reg = reg_loss_fn(model)
    loss_func = nn.CrossEntropyLoss()
    loss_grad = loss_func(pred, label)
    loss = lambda_nll * loss_grad + lambda_reg * loss_reg
    return loss


def CoxLoss_fn(survtime, censor, hazard_pred):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    # print("R mat shape:", R_mat.shape)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i, j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).cuda()
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    # print("censor and theta shape:", censor.shape, theta.shape)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox