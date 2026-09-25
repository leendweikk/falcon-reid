import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReIDModel(nn.Module):
    def __init__(self, num_classes, backbone="convnext_base.dinov3_lvd1689m",
                 head="linear", scale=30.0, margin=0.25, pretrained=True, pool="avg"):
        super().__init__()
        # pretrained=False -> build the empty architecture offline (no internet), then load our weights
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.backbone.set_grad_checkpointing(True)   # trade compute for memory (training only)
        dim = self.backbone.num_features

        self.bnneck = nn.BatchNorm1d(dim)
        self.bnneck.bias.requires_grad_(False)       # frozen bias (BNNeck trick)

        self.classifier = nn.Linear(dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

        self.head, self.scale, self.margin = head, scale, margin

        # pooling: "avg" (timm default) or "gem" (generalized mean: strong small details count more).
        # GeM is only defined for ConvNeXt feature maps (B, C, H, W).
        self.pool = pool
        if pool == "gem":
            assert hasattr(self.backbone, "head") and hasattr(self.backbone.head, "norm"), "GeM: ConvNeXt only"
            self.gem_p = nn.Parameter(torch.ones(1) * 3.0)   # learnable exponent, starts at 3

    def embed(self, x):
        if self.pool != "gem":
            return self.backbone(x)
        f = self.backbone.forward_features(x)                  # (B, C, H, W)
        p = self.gem_p.clamp(min=1.0)
        f = f.float().clamp(min=1e-6).pow(p).mean(dim=(2, 3), keepdim=True).pow(1.0 / p)
        f = self.backbone.head.norm(f)                          # same LayerNorm as timm's own head
        return f.flatten(1)

    def forward(self, x, labels=None):
        feat = self.embed(x)                         # raw fingerprint -> triplet loss
        feat_bn = self.bnneck(feat)                  # normalized fingerprint
        if not self.training:
            return feat_bn                           # test time: embedding only

        if self.head == "cosface":
            # only DIRECTION matters (like cosine at test time), plus a margin for the right car
            cos = F.linear(F.normalize(feat_bn.float()), F.normalize(self.classifier.weight.float()))
            if labels is not None:
                cos = cos - self.margin * F.one_hot(labels, cos.size(1)).float()
            logits = self.scale * cos
        else:
            logits = self.classifier(feat_bn)
        return feat, logits


def load_reid(path, backbone, device="cpu"):
    """Load our trained weights offline. Returns (model, input_size).
    Weights saved by train_final.py carry their own input size; older files are 256."""
    state = torch.load(path, map_location="cpu")
    size = int(state.pop("input_size", torch.tensor(256)).item())
    pool = "gem" if "gem_p" in state else "avg"
    model = ReIDModel(num_classes=state["classifier.weight"].shape[0], backbone=backbone,
                      pretrained=False, pool=pool)
    model.load_state_dict(state)
    return model.to(device).eval(), size


if __name__ == "__main__":
    for head in ["linear", "cosface"]:
        model = ReIDModel(num_classes=1241, head=head).cuda().train()
        x = torch.randn(8, 3, 256, 256, device="cuda")
        labels = torch.randint(0, 1241, (8,), device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            feat, logits = model(x, labels)
        print(f"{head}: feat {tuple(feat.shape)}  logits {tuple(logits.shape)}  "
              f"logit range [{logits.min().item():.1f}, {logits.max().item():.1f}]")