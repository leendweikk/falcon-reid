import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """Batch-hard triplet loss. margin=None -> soft-margin version (never exactly zero)."""

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, feats, labels):
        feats = feats.float()                                   # distances in float32
        dist = torch.cdist(feats, feats)                        # (B, B) euclidean distances
        same = labels.unsqueeze(0) == labels.unsqueeze(1)       # (B, B) True where same car

        hardest_pos = (dist * same).max(dim=1).values           # farthest same-car photo
        hardest_neg = (dist + same * 1e6).min(dim=1).values     # closest different-car photo

        if self.margin is None:                                 # soft margin: smooth, keeps teaching
            return F.softplus(hardest_pos - hardest_neg).mean()
        return F.relu(hardest_pos - hardest_neg + self.margin).mean()


class ReIDLoss(nn.Module):
    """Total loss = ID loss (with label smoothing) + triplet loss."""

    def __init__(self, margin=0.3, smoothing=0.1):
        super().__init__()
        self.id_loss = nn.CrossEntropyLoss(label_smoothing=smoothing)
        self.triplet = TripletLoss(margin)

    def forward(self, feat, logits, labels):
        id_l = self.id_loss(logits.float(), labels)
        tri_l = self.triplet(feat, labels)
        return id_l + tri_l, id_l.item(), tri_l.item()


if __name__ == "__main__":
    labels = torch.arange(4).repeat_interleave(4)
    centers = torch.randn(4, 768) * 10
    good_feat = centers[labels] + 0.01 * torch.randn(16, 768)
    for m in [0.3, None]:
        tri = TripletLoss(margin=m)(good_feat, labels)
        print(f"margin={m}: triplet on perfect features = {tri:.6f}")