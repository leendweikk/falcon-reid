import re


def param_groups(model, lr_backbone, lr_head, llrd=1.0):
    """AdamW parameter groups.
    llrd < 1 (ViT only): layer-wise learning-rate decay. The last block learns at lr_backbone,
    each earlier block at llrd times the one after it, so the pretrained early layers change gently."""
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]   # bnneck, classifier, gem_p
    blocks = getattr(model.backbone, "blocks", None)
    if llrd >= 1.0 or blocks is None:
        return [{"params": list(model.backbone.parameters()), "lr": lr_backbone},
                {"params": head, "lr": lr_head}]

    n = len(blocks)
    groups = {}
    for name, p in model.backbone.named_parameters():
        m = re.match(r"blocks\.(\d+)\.", name)
        layer = int(m.group(1)) + 1 if m else (n + 1 if name.startswith("norm") else 0)
        groups.setdefault(layer, []).append(p)
    out = [{"params": ps, "lr": lr_backbone * llrd ** (n + 1 - layer)} for layer, ps in sorted(groups.items())]
    return out + [{"params": head, "lr": lr_head}]
