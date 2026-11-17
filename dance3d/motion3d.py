#!/usr/bin/env python3
"""Turn the per-frame MediaPipe result of the target dancer into clean 3D motion.

* fixes left/right label flips
* fills short gaps, smooths jitter, keeps bone lengths constant
* estimates where on the floor the dancer is (weak-perspective fit of the 3D pose to
  the 2D keypoints -> distance from camera) so the figure travels like in the video
* aligns the floor to y = 0 and pins the lowest foot to it

Stage coordinates of the output: x = right (as seen from the camera / audience),
y = up, z = toward the camera.  Units: metres.
"""
import argparse
import os
import sys

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (FLIP_PERM, L_AN, L_FT, L_HEEL, L_HIP, L_SH, R_AN, R_FT,  # noqa: E402
                    R_HEEL, R_HIP, R_SH, nan_interp)

# kinematic tree used to keep bone lengths constant (parent, child), root = hip centre
TREE = [
    (-1, L_HIP), (-1, R_HIP),
    (L_HIP, L_SH), (R_HIP, R_SH),
    (L_SH, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (R_SH, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (L_HIP, 25), (25, 27), (27, 29), (27, 31),
    (R_HIP, 26), (26, 28), (28, 30), (28, 32),
]
HEAD_PTS = list(range(0, 11))  # face points follow the shoulder centre rigidly
FEET = [L_AN, R_AN, L_HEEL, R_HEEL, L_FT, R_FT]


def smooth_runs(x, valid, win, order):
    out = x.copy()
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return out
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    for r in runs:
        w = min(win, len(r) if len(r) % 2 else len(r) - 1)
        if w > order + 1:
            out[r] = savgol_filter(x[r], w, order, axis=0)
    return out


def fix_lr_flips(world, valid):
    """MediaPipe sometimes swaps left/right labels (mostly when the dancer turns away)."""
    w = world.copy()
    prev = None
    flipped = np.zeros(len(w), bool)
    for t in range(len(w)):
        if not valid[t]:
            continue
        if prev is not None:
            cur = w[t]
            flp = cur[FLIP_PERM]
            d_cur = np.linalg.norm(cur - prev, axis=1).mean()
            d_flp = np.linalg.norm(flp - prev, axis=1).mean()
            if d_flp < 0.6 * d_cur:
                w[t] = flp
                flipped[t] = True
        prev = w[t]
    return w, flipped


def weak_perspective_depth(kp2d, world, f_px, valid):
    """Fit kp2d ~ s * world_xy + t per frame. Returns depth Z = f/s and hip centre pixel."""
    n = len(kp2d)
    Z = np.full(n, np.nan)
    for t in range(n):
        if not valid[t]:
            continue
        v = kp2d[t, :, 2] > 0.5
        idx = np.flatnonzero(v)
        idx = idx[idx >= 11]  # body only
        if len(idx) < 6:
            continue
        P = world[t, idx, :2]
        Q = kp2d[t, idx, :2]
        Pc, Qc = P - P.mean(0), Q - Q.mean(0)
        den = (Pc ** 2).sum()
        if den < 1e-6:
            continue
        s = (Pc * Qc).sum() / den
        if s > 1:
            Z[t] = f_px / s
    return Z


def sim_mat(p):
    """(log_s, theta, tx, ty) -> 3x3 similarity matrix."""
    sc = np.exp(p[0])
    c, si = sc * np.cos(p[1]), sc * np.sin(p[1])
    return np.array([[c, -si, p[2]], [si, c, p[3]], [0, 0, 1.0]])


def rotation_to_align(a, b):
    """Rotation matrix taking unit vector a to unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def enforce_bone_lengths(P, lengths):
    out = P.copy()
    root = (P[:, L_HIP] + P[:, R_HIP]) / 2
    for parent, child in TREE:
        p = root if parent < 0 else out[:, parent]
        d = P[:, child] - (root if parent < 0 else P[:, parent])
        nrm = np.linalg.norm(d, axis=1, keepdims=True)
        out[:, child] = p + d / np.maximum(nrm, 1e-6) * lengths[(parent, child)]
    # face follows the neck rigidly
    neck_old = (P[:, L_SH] + P[:, R_SH]) / 2
    neck_new = (out[:, L_SH] + out[:, R_SH]) / 2
    out[:, HEAD_PTS] = P[:, HEAD_PTS] + (neck_new - neck_old)[:, None]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("motion", help="output of track.py")
    ap.add_argument("-o", "--out", default="motion3d.npz")
    ap.add_argument("--fov", type=float, default=60.0, help="assumed horizontal field of view of the camera")
    ap.add_argument("--max-gap", type=float, default=1.5, help="seconds of missing pose to interpolate")
    ap.add_argument("--smooth", type=float, default=0.23, help="pose smoothing window (s)")
    ap.add_argument("--travel-smooth", type=float, default=0.8, help="floor-position smoothing window (s)")
    ap.add_argument("--depth-gain", type=float, default=0.6,
                    help="scale of front/back travel (depth estimate is noisy, so it is damped)")
    ap.add_argument("--camera", default="", help="camera.npz from camera.py (zoom/pan compensation)")
    args = ap.parse_args()

    z = np.load(args.motion)
    fps, W, H, n = float(z["fps"]), int(z["W"]), int(z["H"]), int(z["n"])
    kp2d, world, src = z["kp2d"].astype(np.float64), z["world"].astype(np.float64), z["src"]
    valid = src > 0
    print(f"frames={n} fps={fps:.2f} valid={valid.mean() * 100:.1f}%")

    # MediaPipe world: x right, y down, z away from camera  ->  stage: x right, y up, z toward camera
    world = world * np.array([1, -1, -1])
    world, flipped = fix_lr_flips(world, valid)
    kp2d[flipped] = kp2d[flipped][:, FLIP_PERM]  # 2D labels follow the same fix
    print(f"left/right flips fixed: {flipped.sum()}")

    # ---- camera zoom / pan compensation: express 2D keypoints in the reference view
    if args.camera:
        T = np.load(args.camera)["T"]
        if len(T) < n:
            T = np.vstack([T, np.repeat(T[-1:], n - len(T), axis=0)])
        for t in range(n):
            if valid[t]:
                M = sim_mat(T[t])
                kp2d[t, :, :2] = kp2d[t, :, :2] @ M[:2, :2].T + M[:2, 2]
        zf = np.exp(T[:n, 0])
        print(f"camera compensation applied (zoom factor {zf.min():.2f}..{zf.max():.2f})")

    # ---- global position on the floor
    f_px = (W / 2) / np.tan(np.radians(args.fov) / 2)
    Zc = weak_perspective_depth(kp2d, world * np.array([1, -1, 1]), f_px, valid)
    hip_px = (kp2d[:, L_HIP, :2] + kp2d[:, R_HIP, :2]) / 2
    good = ~np.isnan(Zc)
    if good.sum() > 10:
        Zc[good] = median_filter(Zc[good], size=int(fps * 0.5) | 1, mode="nearest")
    Zc = nan_interp(Zc, max_gap=int(args.max_gap * fps))
    hip_px = nan_interp(np.where(valid[:, None], hip_px, np.nan), max_gap=int(args.max_gap * fps))
    # camera coordinates (x right, y down, z forward) -> stage (x right, y up, z toward camera)
    cam = np.stack([(hip_px[:, 0] - W / 2) * Zc / f_px,
                    (hip_px[:, 1] - H / 2) * Zc / f_px,
                    Zc], axis=1)
    root = cam * np.array([1, -1, -1])

    # ---- fill gaps & smooth the body pose (relative to hip centre)
    win = max(5, int(args.smooth * fps) | 1)
    rel = np.where(valid[:, None, None], world, np.nan)
    # drop single-frame glitches: frames far from a robustly smoothed version of themselves
    ok = valid.copy()
    for _ in range(2):
        tmp = nan_interp(np.where(ok[:, None, None], rel, np.nan), max_gap=int(args.max_gap * fps))
        have0 = ~np.isnan(tmp[:, 0, 0])
        med = tmp.copy()
        med[have0] = median_filter(tmp[have0], size=(5, 1, 1), mode="nearest")
        sm = smooth_runs(np.nan_to_num(med), have0, win, 2)
        dev = np.linalg.norm(np.nan_to_num(rel) - sm, axis=2).mean(axis=1)
        thr = max(0.06, 3.0 * np.median(dev[ok])) if ok.any() else 0.06
        ok &= ~(dev > thr)
    print(f"glitch frames removed: {(valid & ~ok).sum()}")
    rel = np.where(ok[:, None, None], rel, np.nan)
    rel = nan_interp(rel, max_gap=int(args.max_gap * fps))
    have = ~np.isnan(rel[:, 0, 0]) & ~np.isnan(root[:, 0])
    rel = smooth_runs(np.nan_to_num(rel), have, win, 3)

    lengths = {}
    root_rel = (rel[have][:, L_HIP] + rel[have][:, R_HIP]) / 2
    for parent, child in TREE:
        p = root_rel if parent < 0 else rel[have][:, parent]
        lengths[(parent, child)] = float(np.median(np.linalg.norm(rel[have][:, child] - p, axis=1)))
    rel[have] = enforce_bone_lengths(rel[have], lengths)

    # ---- floor alignment: fit a plane through the lowest foot point of each frame
    twin = max(5, int(args.travel_smooth * fps) | 1)
    root_s = smooth_runs(np.nan_to_num(root), have, twin, 2)
    abs_pts = rel + root_s[:, None]
    feet = abs_pts[:, FEET]
    low = feet[np.arange(n), np.argmin(feet[:, :, 1], axis=1)]
    pts = low[have]
    R = np.eye(3)
    if len(pts) > 30:
        # robust plane fit (least squares on y = a x + b z + c, iterate dropping jump outliers)
        keep = np.ones(len(pts), bool)
        for _ in range(3):
            A = np.c_[pts[keep, 0], pts[keep, 2], np.ones(keep.sum())]
            coef, *_ = np.linalg.lstsq(A, pts[keep, 1], rcond=None)
            res = pts[:, 1] - np.c_[pts[:, 0], pts[:, 2], np.ones(len(pts))] @ coef
            keep = res < np.percentile(res, 80)
        normal = np.array([-coef[0], 1.0, -coef[1]])
        tilt = np.degrees(np.arccos(normal[1] / np.linalg.norm(normal)))
        if tilt < 25:
            R = rotation_to_align(normal, np.array([0.0, 1.0, 0.0]))
            print(f"floor tilt corrected: {tilt:.1f} deg")
    root_s = root_s @ R.T
    rel = rel @ R.T

    # centre the stage on the mean position, damp depth travel
    centre = np.nanmean(root_s[have], axis=0)
    root_s = root_s - centre
    root_s[:, 2] *= args.depth_gain
    root_s[:, 0] *= 1.0

    # vertical: pin the lowest foot to the floor (stable, no floating / sinking)
    P = rel + root_s[:, None] * np.array([1, 0, 1])
    lowest = P[:, FEET, 1].min(axis=1)
    lift = smooth_runs(-lowest, have, max(3, int(0.1 * fps) | 1), 1)
    P[:, :, 1] += lift[:, None]
    P[~have] = np.nan

    # hold the last pose through long gaps (flagged so the renderer can grey it out)
    held = ~have
    last = None
    first = np.flatnonzero(have)[0] if have.any() else 0
    for t in range(n):
        if have[t]:
            last = P[t]
        elif last is not None:
            P[t] = last
        else:
            P[t] = P[first]

    np.savez_compressed(args.out, fps=fps, joints=P.astype(np.float32), held=held,
                        bone_lengths=np.array(list(lengths.values()), np.float32))
    extent = np.nanmax(P[have][:, :, [0, 2]].reshape(-1, 2), 0) - np.nanmin(P[have][:, :, [0, 2]].reshape(-1, 2), 0)
    print(f"wrote {args.out}: {n} frames, held(gaps)={held.mean() * 100:.1f}%, "
          f"travel area {extent[0]:.1f} x {extent[1]:.1f} m")


if __name__ == "__main__":
    main()
