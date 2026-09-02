"""Fail-closed repair for localized collapsed H3 terminal video tokens.

MiniMax H3 video latents repeat a 5-token temporal phase (1 + 4 + 4 + 4 + 4).
On long video sequences, a rare failure can leave the lower spatial portion of the
final token with sharply reduced energy, resulting in a visible dark/blurred horizontal
band across roughly the final 5 decoded frames.

This guard detects if the terminal token lower region collapsed relative to its historical
5-token phase counterparts. If detected, it feathers the lower half conservatively toward
a same-phase linear estimate. Normal latents are passed through bit-identically.
Supports both PyTorch tensors and NumPy/MLX arrays.
"""

from __future__ import annotations
from typing import Any

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np


def _std_torch(val: "torch.Tensor") -> "torch.Tensor":
    return val.float().std(unbiased=False).clamp_min(1.0e-8)


def stabilize_terminal_video_latent_torch(
    video: "torch.Tensor",
    *,
    temporal_period: int = 5,
    split_fraction: float = 0.52,
    hard_ratio_limit: float = 0.85,
    robust_mad_multiplier: float = 8.0,
    stable_top_range: tuple[float, float] = (0.8, 1.2),
    collapsed_bottom_limit: float = 0.85,
    target_history_fraction: float = 0.95,
    target_ratio_cap: float = 0.97,
    max_repair_strength: float = 1.0,
) -> dict[str, Any]:
    """Stabilize terminal token collapse on PyTorch tensor [B, C, T, H, W] in-place."""
    profile: dict[str, Any] = {
        "schema_version": 1,
        "mode": "terminal_phase_guard",
        "triggered": False,
        "reason": "not_evaluated",
    }
    if video.ndim != 5 or video.shape[0] != 1:
        profile["reason"] = "unsupported_shape"
        return profile
    if not video.is_floating_point() or video.shape[2] < temporal_period * 3:
        profile["reason"] = "insufficient_history"
        return profile

    height = int(video.shape[-2])
    split = min(height - 1, max(1, round(height * split_fraction)))
    terminal = int(video.shape[2]) - 1
    phase = terminal % temporal_period
    history_indices = tuple(range(phase, terminal, temporal_period))
    if len(history_indices) < 2:
        profile["reason"] = "insufficient_phase_history"
        return profile

    ratios = []
    for index in history_indices:
        token = video[0, :, index]
        top = _std_torch(token[:, :split])
        bottom = _std_torch(token[:, split:])
        ratios.append(bottom / top)
    history = torch.stack(ratios)
    median = history.median()
    mad = (history - median).abs().median()

    current = video[:, :, terminal : terminal + 1]
    previous = video[:, :, terminal - temporal_period : terminal - temporal_period + 1]
    older = video[:, :, terminal - 2 * temporal_period : terminal - 2 * temporal_period + 1]

    current_top = _std_torch(current[..., :split, :])
    current_bottom = _std_torch(current[..., split:, :])
    previous_top = _std_torch(previous[..., :split, :])
    previous_bottom = _std_torch(previous[..., split:, :])

    current_ratio = current_bottom / current_top
    top_relative = current_top / previous_top
    bottom_relative = current_bottom / previous_bottom

    robust_limit = median - robust_mad_multiplier * mad.clamp_min(0.005)
    trigger_limit = torch.minimum(
        robust_limit,
        current_ratio.new_tensor(float(hard_ratio_limit)),
    )

    vals = torch.stack((
        current_ratio, median, mad, trigger_limit, top_relative, bottom_relative
    )).detach().cpu().tolist()

    profile.update({
        "terminal_phase": int(phase),
        "history_count": len(history_indices),
        "split_row": split,
        "bottom_top_ratio": float(vals[0]),
        "history_ratio_median": float(vals[1]),
        "history_ratio_mad": float(vals[2]),
        "trigger_limit": float(vals[3]),
        "top_relative_to_previous_phase": float(vals[4]),
        "bottom_relative_to_previous_phase": float(vals[5]),
    })

    top_is_stable = stable_top_range[0] <= vals[4] <= stable_top_range[1]
    bottom_collapsed = vals[5] < collapsed_bottom_limit
    ratio_is_outlier = vals[0] < vals[3]

    if not ratio_is_outlier:
        profile["reason"] = "ratio_not_outlier"
        return profile
    if not top_is_stable:
        profile["reason"] = "top_region_not_stable"
        return profile
    if not bottom_collapsed:
        profile["reason"] = "bottom_region_not_collapsed"
        return profile

    estimate = previous + (previous - older)
    y = torch.linspace(0.0, 1.0, height, device=video.device, dtype=torch.float32)
    base_mask = ((y - split_fraction) / (0.76 - split_fraction)).clamp(0.0, 1.0)
    base_mask = base_mask.view(1, 1, 1, height, 1)

    target_ratio = min(vals[1] * target_history_fraction, target_ratio_cap)
    low, high = 0.0, float(max_repair_strength)
    for _ in range(8):
        middle = (low + high) * 0.5
        candidate = current.lerp(estimate, base_mask.mul(middle))
        cand_top = _std_torch(candidate[..., :split, :])
        cand_bot = _std_torch(candidate[..., split:, :])
        if float((cand_bot / cand_top).item()) >= target_ratio:
            high = middle
        else:
            low = middle

    repair_strength = high
    mask = base_mask.mul(repair_strength)
    current.lerp_(estimate, mask.to(dtype=current.dtype))
    profile.update({
        "triggered": True,
        "reason": "localized_terminal_bottom_collapse",
        "repair": "adaptive_same_phase_linear_feather",
        "repair_strength": float(repair_strength),
        "repair_target_ratio": float(target_ratio),
    })
    return profile


def stabilize_terminal_video_latent_numpy(
    video: np.ndarray,
    *,
    temporal_period: int = 5,
    split_fraction: float = 0.52,
    hard_ratio_limit: float = 0.85,
    robust_mad_multiplier: float = 8.0,
    stable_top_range: tuple[float, float] = (0.8, 1.2),
    collapsed_bottom_limit: float = 0.85,
    target_history_fraction: float = 0.95,
    target_ratio_cap: float = 0.97,
    max_repair_strength: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """NumPy / MLX array variant for H3MLX standalone pipeline."""
    profile: dict[str, Any] = {
        "schema_version": 1,
        "mode": "terminal_phase_guard",
        "triggered": False,
        "reason": "not_evaluated",
    }
    if video.ndim != 5 or video.shape[0] != 1:
        profile["reason"] = "unsupported_shape"
        return video, profile
    if video.shape[2] < temporal_period * 3:
        profile["reason"] = "insufficient_history"
        return video, profile

    height = int(video.shape[-2])
    split = min(height - 1, max(1, round(height * split_fraction)))
    terminal = int(video.shape[2]) - 1
    phase = terminal % temporal_period
    history_indices = tuple(range(phase, terminal, temporal_period))
    if len(history_indices) < 2:
        profile["reason"] = "insufficient_phase_history"
        return video, profile

    ratios = []
    for idx in history_indices:
        token = video[0, :, idx]
        top = float(np.std(token[:, :split]) + 1e-8)
        bot = float(np.std(token[:, split:]) + 1e-8)
        ratios.append(bot / top)
    history = np.array(ratios)
    median = float(np.median(history))
    mad = float(np.median(np.abs(history - median)))

    current = video[:, :, terminal : terminal + 1].copy()
    previous = video[:, :, terminal - temporal_period : terminal - temporal_period + 1]
    older = video[:, :, terminal - 2 * temporal_period : terminal - 2 * temporal_period + 1]

    cur_top = float(np.std(current[..., :split, :]) + 1e-8)
    cur_bot = float(np.std(current[..., split:, :]) + 1e-8)
    prev_top = float(np.std(previous[..., :split, :]) + 1e-8)
    prev_bot = float(np.std(previous[..., split:, :]) + 1e-8)

    cur_ratio = cur_bot / cur_top
    top_rel = cur_top / prev_top
    bot_rel = cur_bot / prev_bot

    robust_limit = median - robust_mad_multiplier * max(mad, 0.005)
    trigger_limit = min(robust_limit, hard_ratio_limit)

    profile.update({
        "terminal_phase": int(phase),
        "history_count": len(history_indices),
        "split_row": split,
        "bottom_top_ratio": float(cur_ratio),
        "history_ratio_median": float(median),
        "history_ratio_mad": float(mad),
        "trigger_limit": float(trigger_limit),
        "top_relative_to_previous_phase": float(top_rel),
        "bottom_relative_to_previous_phase": float(bot_rel),
    })

    top_is_stable = stable_top_range[0] <= top_rel <= stable_top_range[1]
    bottom_collapsed = bot_rel < collapsed_bottom_limit
    ratio_is_outlier = cur_ratio < trigger_limit

    if not (ratio_is_outlier and top_is_stable and bottom_collapsed):
        profile["reason"] = "no_repair_needed"
        return video, profile

    estimate = previous + (previous - older)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    base_mask = np.clip((y - split_fraction) / (0.76 - split_fraction), 0.0, 1.0)
    base_mask = base_mask.reshape(1, 1, 1, height, 1)

    target_ratio = min(median * target_history_fraction, target_ratio_cap)
    low, high = 0.0, float(max_repair_strength)
    for _ in range(8):
        middle = (low + high) * 0.5
        cand = current * (1.0 - base_mask * middle) + estimate * (base_mask * middle)
        cand_top = float(np.std(cand[..., :split, :]) + 1e-8)
        cand_bot = float(np.std(cand[..., split:, :]) + 1e-8)
        if (cand_bot / cand_top) >= target_ratio:
            high = middle
        else:
            low = middle

    repair_strength = high
    mask = base_mask * repair_strength
    repaired_current = current * (1.0 - mask) + estimate * mask
    out_video = video.copy()
    out_video[:, :, terminal : terminal + 1] = repaired_current

    profile.update({
        "triggered": True,
        "reason": "localized_terminal_bottom_collapse",
        "repair": "adaptive_same_phase_linear_feather",
        "repair_strength": float(repair_strength),
        "repair_target_ratio": float(target_ratio),
    })
    return out_video, profile


def stabilize_terminal_video_latent_(video: Any, **kwargs) -> Any:
    """Unified entry point supporting PyTorch tensors and NumPy/MLX arrays."""
    if HAS_TORCH and isinstance(video, torch.Tensor):
        return stabilize_terminal_video_latent_torch(video, **kwargs)
    elif isinstance(video, np.ndarray):
        repaired, prof = stabilize_terminal_video_latent_numpy(video, **kwargs)
        np.copyto(video, repaired)
        return prof
    else:
        raise TypeError(f"Unsupported latent tensor type: {type(video)}")
