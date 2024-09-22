import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion_triplet = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        self.weight = weight

    def forward(self, input):
        anchor_cls = input["anchor_cls"]
        positive_cls = input["positive_cls"]
        negative_cls = input["negative_cls"]
      
        return self.criterion_triplet(anchor_cls, positive_cls, negative_cls)


class DiceLoss(nn.Module):
    def __init__(self, weight, smooth=1, p=2, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.smooth = smooth
        self.p = p
        self.reduction = reduction


    def forward(self, input):
        gt_masks = input["mask"]
        pred_mask_logit = input['ref_logit'].sigmoid()

        predict = pred_mask_logit.contiguous().view(pred_mask_logit.shape[0], -1)
        target = gt_masks.contiguous().view(gt_masks.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class CELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion_ce = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, input):
        class_logits = input["class_logit"]
        class_labels = input["class_label"]

        return self.criterion_ce(class_logits, class_labels)


class FeatureMSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        rec_feats = input["rec_feat"]
        target_feats = input["target_feat"]
        return self.criterion_mse(rec_feats, target_feats)


def build_criterion(config):
    loss_dict = {}
    for i in range(len(config)):
        cfg = config[i]
        loss_name = cfg["name"]
        loss_dict[loss_name] = globals()[cfg["type"]](**cfg["kwargs"])
    return loss_dict
