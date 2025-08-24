import numpy as np
import torch.nn.functional as F

def calc(pred, tgt, bsz, mode="per_feature"):
    """
    Calculate correlation and cosine similarity between pred and tgt.

    Args:
        pred (torch.Tensor): Predictions (bsz, n_features)
        tgt (torch.Tensor): Ground truth (bsz, n_features)
        bsz (int): batch size
        mode (str): "global" or "per_feature"
            - "global": flatten everything and compute one correlation
            - "per_feature": compute correlation per feature and return the average

    Returns:
        corr (float): correlation value
        cos (float): cosine similarity
    """
    if pred is None or tgt is None:
        return 0.0, 0.0

    # reshape to (bsz, n_features_flat)
    p = pred.view(bsz, -1)
    t = tgt.view(bsz, -1)

    if mode == "global":
        # Flatten across all samples and features
        p_flat = p.reshape(-1).cpu().numpy()
        t_flat = t.reshape(-1).cpu().numpy()
        corr = np.corrcoef(p_flat, t_flat)[0, 1]

    elif mode == "per_feature":
        # Compute correlation for each feature across samples
        p_np, t_np = p.cpu().numpy(), t.cpu().numpy()
        n_features = p_np.shape[1]
        corrs = []
        for j in range(n_features):
            if np.std(p_np[:, j]) == 0 or np.std(t_np[:, j]) == 0:
                corrs.append(np.nan)  # avoid NaNs
            else:
                corrs.append(np.corrcoef(p_np[:, j], t_np[:, j])[0, 1])
        corr = np.nanmean(corrs)

    else:
        raise ValueError("mode must be 'global' or 'per_feature'")

    # Cosine similarity (always global)
    cos = F.cosine_similarity(p.flatten().unsqueeze(0), t.flatten().unsqueeze(0), dim=1).item()

    return corr, cos



def calc_metrics(pred, target, mode="per_feature"):
    """
    Calculate correlation and cosine similarity between pred and target.

    Args:
        pred (torch.Tensor): Predictions (n_samples, n_features)
        target (torch.Tensor): Ground truth (n_samples, n_features)
        mode (str): "global" or "per_feature"
            - "global": flatten everything and compute one correlation
            - "per_feature": compute correlation per feature and return the average

    Returns:
        corr (float): correlation value
        cos_sim (float): cosine similarity
    """
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    if mode == "global":
        # Flatten across samples and features
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        corr = np.corrcoef(pred_flat, target_flat)[0, 1]

    elif mode == "per_feature":
        # Compute correlation for each feature across samples
        n_features = pred.shape[1]
        corrs = []
        for j in range(n_features):
            if np.std(pred[:, j]) == 0 or np.std(target[:, j]) == 0:
                corrs.append(np.nan)  # avoid division by zero
            else:
                corrs.append(np.corrcoef(pred[:, j], target[:, j])[0, 1])
        corr = np.nanmean(corrs)  # average correlation across features

    else:
        raise ValueError("mode must be 'global' or 'per_feature'")

    # Cosine similarity (always global across all values)
    cos_sim = np.dot(pred.flatten(), target.flatten()) / (
        np.linalg.norm(pred.flatten()) * np.linalg.norm(target.flatten()) + 1e-8)
    
    return corr, cos_sim


