"""
Gauge normalization/fixing functions to remove the gauge freedom in the neuralign model.
These functions help with scaling and sign ambiguities in the factors, making them more interpretable
"""

import torch

@torch.no_grad()
def gauge_fix_model(model, eps=1e-12, sign_fix=False):

    # Normalize V and absorb its component scales into the CP weights.
    scale = torch.linalg.norm(model.V, dim=0)
    scale = torch.where(scale > eps, scale, torch.ones_like(scale))
    model.V.div_(scale)
    model.w.mul_(scale)

    # Normalize H while preserving H @ diag(S_n).
    scale = torch.linalg.norm(model.H, dim=0)
    scale = torch.where(scale > eps, scale, torch.ones_like(scale))
    model.H.div_(scale)
    for S in model.S_n:
        S.mul_(scale)

    # Normalize U with one component scale shared equally across sessions and
    # absorb that scale into the CP weights.
    scale = torch.sqrt(
        torch.stack([torch.mean(U.square(), dim=0) for U in model.U]).mean(dim=0)
    )
    scale = torch.where(scale > eps, scale, torch.ones_like(scale))
    for U in model.U:
        U.div_(scale)
    
    # normalize condition centroids if they exist
    if hasattr(model, "C") and model.C is not None:
        model.C.div_(scale)
        
    model.w.mul_(scale)
        

    if sign_fix:
        # Make the largest-magnitude entry of each V column positive. Flipping
        # every corresponding U column preserves the CP reconstruction.
        indices = torch.argmax(torch.abs(model.V), dim=0, keepdim=True)
        anchors = torch.gather(model.V, 0, indices).squeeze(0)
        signs = torch.where(
            torch.abs(anchors) > eps,
            torch.sign(anchors),
            torch.ones_like(anchors),
        )
        model.V.mul_(signs)
        for U in model.U:
            U.mul_(signs)
        if hasattr(model, "C") and model.C is not None:
            model.C.mul_(signs)

        # Apply the same convention to H and compensate in every S_n so that
        # H @ diag(S_n) is unchanged.
        indices = torch.argmax(torch.abs(model.H), dim=0, keepdim=True)
        anchors = torch.gather(model.H, 0, indices).squeeze(0)
        signs = torch.where(
            torch.abs(anchors) > eps,
            torch.sign(anchors),
            torch.ones_like(anchors),
        )
        model.H.mul_(signs)
        for S in model.S_n:
            S.mul_(signs)
