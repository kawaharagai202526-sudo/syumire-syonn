"""Shared constants and helpers for the dance-to-3D pipeline (MediaPipe 33-point pose)."""
import numpy as np

NUM_LM = 33

NOSE = 0
L_EYE, R_EYE = 2, 5
L_EAR, R_EAR = 7, 8
MOUTH_L, MOUTH_R = 9, 10
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_THUMB, R_THUMB = 21, 22
L_HIP, R_HIP = 23, 24
L_KN, R_KN = 25, 26
L_AN, R_AN = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FT, R_FT = 31, 32

# (left, right) index pairs, used to undo left/right label flips.
LR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
            (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]

FLIP_PERM = np.arange(NUM_LM)
for _l, _r in LR_PAIRS:
    FLIP_PERM[_l], FLIP_PERM[_r] = _r, _l

BODY_IDX = [L_SH, R_SH, L_HIP, R_HIP]


def body_center(kp):
    """Mean of shoulders and hips. kp: (..., 33, >=2)."""
    return kp[..., BODY_IDX, :2].mean(axis=-2)


def bbox_from_kp(kp, w, h, vis_thr=0.2):
    v = kp[:, 2] > vis_thr
    pts = kp[v, :2] if v.sum() >= 4 else kp[:, :2]
    x0, y0 = np.clip(pts.min(0), 0, [w - 1, h - 1])
    x1, y1 = np.clip(pts.max(0), 0, [w - 1, h - 1])
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def nan_interp(arr, max_gap=None):
    """Linearly interpolate NaNs along axis 0 (each trailing element independently).

    Gaps longer than ``max_gap`` frames are left as NaN. Leading/trailing NaNs are kept.
    """
    a = np.array(arr, dtype=np.float64)
    flat = a.reshape(a.shape[0], -1)
    n = flat.shape[0]
    t = np.arange(n)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        good = ~np.isnan(col)
        if good.sum() < 2:
            continue
        gi = np.flatnonzero(good)
        filled = np.interp(t, gi, col[gi])
        inside = (t > gi[0]) & (t < gi[-1])
        fill = (~good) & inside
        if max_gap is not None:
            # measure gap length for every missing run
            nxt = np.searchsorted(gi, t)
            prev_g = gi[np.clip(nxt - 1, 0, len(gi) - 1)]
            next_g = gi[np.clip(nxt, 0, len(gi) - 1)]
            fill &= (next_g - prev_g - 1) <= max_gap
        col[fill] = filled[fill]
        flat[:, j] = col
    return flat.reshape(a.shape)
