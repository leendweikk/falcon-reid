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


class RelationalKD(nn.Module):
    """Distillation of the ensemble into one model (docs/PLAN.md A11), at the level of RELATIONS:
    inside a batch, each photo's similarity profile to the other photos (a softmax over cosine
    similarities / tau) should match the teacher's profile. Works although the student (768-d) and
    the teacher (Base+ViT, 1792-d) have different sizes, and needs no teacher classifier."""

    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau

    def forward(self, student, teacher):
        s = F.normalize(student.float(), dim=1)
        t = F.normalize(teacher.float(), dim=1)
        eye = torch.eye(len(s), dtype=torch.bool, device=s.device)
        s_logits = (s @ s.T / self.tau).masked_fill(eye, -1e4)      # a photo is not compared with itself
        t_logits = (t @ t.T / self.tau).masked_fill(eye, -1e4)
        return F.kl_div(F.log_softmax(s_logits, dim=1), F.log_softmax(t_logits, dim=1),
                        reduction="batchmean", log_target=True)


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