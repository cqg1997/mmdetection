import torch
import torch.nn.functional as F


def sigmoid_focal_loss(pred,
                       target,
                       weight,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='elementwise_mean'):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    return F.binary_cross_entropy_with_logits(
        pred, target, weight, reduction=reduction)


def py_bin_sigmoid_focal_loss(pred,
                              target,
                              weight,
                              gamma=2.0,
                              alpha=0.25,
                              avg_factor=None,
                              num_classes=80):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6
    return sigmoid_focal_loss(
        pred, target, weight, gamma=gamma, alpha=alpha,
        reduction='sum')[None] / avg_factor
