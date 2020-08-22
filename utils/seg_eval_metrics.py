# MAE, AUC_PR, AUC_ROC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy
import sklearn.metrics as metrics

class compute_mae(Function):
    def forward(self, input, target):
        eps = 1e-8
        threshold = 0.5
        input = (input > 0.5).float()
        h, w = input.shape[1], input.shape[2]
        sumError = torch.sum(torch.abs(input.view(-1) - target.view(-1)))
        maeError = sumError / (float(h) * float(w) + eps)

        return maeError.cpu()


class compute_auc_roc(Function):
    def forward(self, input, target):
        y_true = target.cpu().view(-1).detach().numpy()
        y_pred = input.cpu().view(-1).detach().numpy()

        try:
            auc_roc = metrics.roc_auc_score(y_true, y_pred)
        except:
            auc_roc = 0

        return auc_roc


class compute_auc_pr(Function):
    def forward(self, input, target):
        y_true = target.cpu().view(-1).detach().numpy()
        y_pred = input.cpu().view(-1).detach().numpy()

        if y_true.max() == 1:
            precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
            auc_pr = metrics.auc(recall, precision)
        else:
            auc_pr = 0

        return auc_pr


def seg_metrics(input, target, pos_count):

    mae = 0
    auc_roc = 0
    auc_pr = 0
    for i, c in enumerate(zip(input, target)):
        mae = mae + compute_mae().forward(c[0], c[1])
        auc_roc = auc_roc + compute_auc_roc().forward(c[0], c[1])
        auc_pr = auc_pr + compute_auc_pr().forward(c[0], c[1])

    if pos_count > 0:
        return mae / (i + 1), auc_roc / pos_count, auc_pr / pos_count
    else:
        return mae / (i + 1), auc_roc, auc_pr