""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import numpy as np


class AverageMeter:
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def calc_tp_fp_fn_tn(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    confusion_vector = pred / target
    #cone is 0 other is 1 -> true and false swiithced
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_negatives = torch.sum(confusion_vector == 1).item() / batch_size
    false_negatives = torch.sum(confusion_vector == float('inf')).item() / batch_size
    true_positives = torch.sum(torch.isnan(confusion_vector)).item() / batch_size
    false_positives = torch.sum(confusion_vector == 0).item() / batch_size

    return true_positives, false_positives, false_negatives, true_negatives


def calc_confusion_matrix(output, target, num_classes):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)

    confusion = np.zeros((num_classes, num_classes))

    for i in range(batch_size):
        confusion[pred[i], target[i]] += 1

    return batch_size, confusion




