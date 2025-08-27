import numpy as np
import torch
import torch.nn.functional as F

def calc(pred, tgt, bsz, corr_mode="per_feature", cosine_mode="per_sample"):
    """
    Calculate correlation and cosine similarity between pred and tgt.

    Args:
        pred (torch.Tensor): Predictions (bsz, n_features)
        tgt (torch.Tensor): Ground truth (bsz, n_features)
        bsz (int): batch size
        corr_mode (str): "global" or "per_feature"
        cosine_mode (str): "global", "per_feature", or "per_sample"

    Returns:
        corr (float): correlation value
        cos (float): cosine similarity
    """
    if pred is None or tgt is None:
        return 0.0, 0.0

    p = pred.view(bsz, -1)
    t = tgt.view(bsz, -1)

    # ---- Correlation ----
    if corr_mode == "global":
        p_flat = p.reshape(-1).cpu().numpy()
        t_flat = t.reshape(-1).cpu().numpy()
        corr = np.corrcoef(p_flat, t_flat)[0, 1]

    elif corr_mode == "per_feature":
        p_np, t_np = p.cpu().numpy(), t.cpu().numpy()
        n_features = p_np.shape[1]
        corrs = []
        for j in range(n_features):
            if np.std(p_np[:, j]) == 0 or np.std(t_np[:, j]) == 0:
                corrs.append(np.nan)
            else:
                corrs.append(np.corrcoef(p_np[:, j], t_np[:, j])[0, 1])
        corr = np.nanmean(corrs)

    else:
        raise ValueError("corr_mode must be 'global' or 'per_feature'")

    # ---- Cosine Similarity ----
    if cosine_mode == "global":
        cos = F.cosine_similarity(p.flatten().unsqueeze(0), t.flatten().unsqueeze(0), dim=1).item()

    elif cosine_mode == "per_feature":
        cos_vals = F.cosine_similarity(p, t, dim=0)  # similarity per feature
        cos = cos_vals.mean().item()

    elif cosine_mode == "per_sample":
        cos_vals = F.cosine_similarity(p, t, dim=1)  # similarity per sample
        cos = cos_vals.mean().item()

    else:
        raise ValueError("cosine_mode must be 'global', 'per_feature', or 'per_sample'")

    return corr, cos


def calc_metrics(pred, target, corr_mode="per_feature", cosine_mode="per_sample"):
    """
    Calculate correlation and cosine similarity between pred and target.

    Args:
        pred (torch.Tensor): Predictions (n_samples, n_features)
        target (torch.Tensor): Ground truth (n_samples, n_features)
        corr_mode (str): "global" or "per_feature"
        cosine_mode (str): "global", "per_feature", or "per_sample"

    Returns:
        corr (float): correlation value
        cos_sim (float): cosine similarity
    """
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    # ---- Correlation ----
    if corr_mode == "global":
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        corr = np.corrcoef(pred_flat, target_flat)[0, 1]

    elif corr_mode == "per_feature":
        n_features = pred.shape[1]
        corrs = []
        for j in range(n_features):
            if np.std(pred[:, j]) == 0 or np.std(target[:, j]) == 0:
                corrs.append(np.nan)
            else:
                corrs.append(np.corrcoef(pred[:, j], target[:, j])[0, 1])
        corr = np.nanmean(corrs)

    else:
        raise ValueError("corr_mode must be 'global' or 'per_feature'")

    # ---- Cosine Similarity ----
    if cosine_mode == "global":
        cos_sim = np.dot(pred.flatten(), target.flatten()) / (
            np.linalg.norm(pred.flatten()) * np.linalg.norm(target.flatten()) + 1e-8)

    elif cosine_mode == "per_feature":
        cos_vals = []
        for j in range(pred.shape[1]):
            dot = np.dot(pred[:, j], target[:, j])
            denom = np.linalg.norm(pred[:, j]) * np.linalg.norm(target[:, j]) + 1e-8
            cos_vals.append(dot / denom)
        cos_sim = np.nanmean(cos_vals)

    elif cosine_mode == "per_sample":
        cos_vals = []
        for i in range(pred.shape[0]):
            dot = np.dot(pred[i, :], target[i, :])
            denom = np.linalg.norm(pred[i, :]) * np.linalg.norm(target[i, :]) + 1e-8
            cos_vals.append(dot / denom)
        cos_sim = np.nanmean(cos_vals)

    else:
        raise ValueError("cosine_mode must be 'global', 'per_feature', or 'per_sample'")

    return corr, cos_sim
