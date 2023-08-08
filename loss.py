import torch
import torch.nn.functional as F


def bce_loss(preds, targets, pos_weight=None):
    """Computes the weighted binary cross-entropy loss.
    Args:
        targets: a tensor of shape [B, 1, H, W].
        preds: a tensor of shape [B, 1, H, W]
    Returns:
        bce_loss: the weighted binary cross-entropy loss.
    """
    bce_loss = F.binary_cross_entropy_with_logits(
        preds.float(),
        targets.float(),
        pos_weight=pos_weight,
    )
    return bce_loss


def ce_loss(preds, targets, ignore=255):
    """Computes the weighted multi-class cross-entropy loss.
    Args:
        targets: a tensor of shape [B, H, W].
        preds: a tensor of shape [B, C, H, W]. 
        ignore: the class index to ignore.
    Returns:
        ce_loss: the weighted multi-class cross-entropy loss.
    """
    ce_loss = F.cross_entropy(
        preds.float(),
        targets.long(),    # [B, H, W]
        ignore_index=ignore,
    )
    return ce_loss


def dice_loss(preds, targets, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Args:
    preds(logits) a tensor of shape [B, C, H, W]
    targets: a tensor of shape [B, 1, H, W].
    eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = preds.shape[1]
    true_1_hot = F.one_hot(targets.squeeze(1), num_classes=num_classes)   # (B, 1, H, W) to (B, H, W, C)
    true_1_hot = true_1_hot.permute(0, 3, 1, 2)                        # (B, H, W, C) to (B, C, H, W)
    probas = F.softmax(preds, dim=1)
    true_1_hot = true_1_hot.type(preds.type()).contiguous()
    dims = (0,) + tuple(range(2, targets.ndimension()))        # dims = (0, 2, 3)
    intersection = torch.sum(probas * true_1_hot, dims)     # intersection w.r.t. the class
    cardinality = torch.sum(probas + true_1_hot, dims)      # cardinality w.r.t. the class
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def jaccard_loss(preds, targets, eps=1e-7):
    """Computes the Jaccard loss.
    Args:
    preds(logits) a tensor of shape [B, C, H, W]
    targets: a tensor of shape [B, 1, H, W].
    eps: added to the denominator for numerical stability.
    Returns:
        Jaccard loss
    """
    num_classes = preds.shape[1]
    true_1_hot = F.one_hot(targets.squeeze(1), num_classes=num_classes)  # (B, 1, H, W) to (B, H, W, C)
    true_1_hot = true_1_hot.permute(0, 3, 1, 2)  # (B, H, W, C) to (B, C, H, W)
    probas = F.softmax(preds, dim=1)
    true_1_hot = true_1_hot.type(preds.type()).contiguous()
    dims = (0,) + tuple(range(2, targets.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return (1 - jacc_loss)


