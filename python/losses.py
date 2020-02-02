import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        numerator = 2 * torch.sum(pred * target)
        denominator = torch.sum(pred + target)
        return 1 - (numerator + 1) / (denominator + 1)


class NormalLoss(nn.Module):
    def __init__(self):
        super(NormalLoss, self).__init__()

    def forward(self, pred, target, mask):
        pred = pred.permute(0, 2, 3, 1)[mask, :]
        target = target.permute(0, 2, 3, 1)[mask, :]
        # Calculate loss : average cosine value between predicted/actual normals at each pixel
        # theta = arccos((P dot Q) / (|P|*|Q|)) -> cos(theta) = (P dot Q) / (|P|*|Q|)
        # Both the predicted and ground truth normals normalized to be between -1 and 1
        preds_norm = nn.functional.normalize(pred, p=2, dim=1)
        truths_norm = nn.functional.normalize(target, p=2, dim=1)
        # make negative so function decreases (cos -> 1 if angles same)
        loss = 1-torch.sum(preds_norm * truths_norm, dim=1)
        return loss.mean()


class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()

    def forward(self, pred, target, masks):
        pred = pred.permute(0, 2, 3, 1)[masks,:]
        target = target.permute(0, 2, 3, 1)[masks,:]

        loss = torch.sqrt(torch.mean(torch.abs(torch.log(target) - torch.log(pred)) ** 2))
        return loss

