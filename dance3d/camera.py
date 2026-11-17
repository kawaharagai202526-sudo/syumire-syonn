#!/usr/bin/env python3
"""Estimate the camera's zoom / pan for every frame, so that positions measured in the
image can be expressed in one fixed "reference" view of the stage.

Output: per frame a similarity transform  T_t : frame-t pixel -> reference pixel
(scale, rotation, translation; full-resolution pixels) and a confidence flag.

Method
  * background only: dancers' boxes (and their reflections on the glossy floor) are
    masked out using the pass-1 detections
  * frame-to-frame increments with KLT optical flow (smooth, but drifts)
  * absolute anchors every few frames: SIFT matching against a reference frame of the
    same lighting (the video switches between bright and dark/orange lighting)
  * the references of different lightings are linked through the increments across
    each lighting change; between anchors the increments are bridged so that the
    result meets both neighbouring anchors (no drift)
"""
import argparse
import sys

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

SC = 0.5  # processing scale


def mat(p):
    """(log_s, theta, tx, ty) -> 3x3"""
    s = np.exp(p[0])
    c, si = s * np.cos(p[1]), s * np.sin(p[1])
    return np.array([[c, -si, p[2]], [si, c, p[3]], [0, 0, 1.0]])


def par(M):
    s = np.hypot(M[0, 0], M[1, 0])
    return np.array([np.log(s), np.arctan2(M[1, 0], M[0, 0]), M[0, 2], M[1, 2]])


def to3(A):
    return np.vstack([A, [0, 0, 1.0]])


def people_mask(shape, boxes):
    h, w = shape
    m = np.full((h, w), 255, np.uint8)
    for b in boxes:
        x0, y0, x1, y1 = b * SC
        bw, bh = x1 - x0, y1 - y0
        x0, x1 = x0 - 0.12 * bw, x1 + 0.12 * bw
        y0, y1 = y0 - 0.06 * bh, y1 + 0.45 * bh  # + reflection on the floor
        m[int(max(y0, 0)):int(min(y1, h)), int(max(x0, 0)):int(min(x1, w))] = 0
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("pass1", help="pass-1 cache from track.py (person boxes)")
    ap.add_argument("-o", "--out", default="camera.npz")
    ap.add_argument("--anchor-every", type=int, default=3)
    ap.add_argument("--min-inliers", type=int, default=30)
    args = ap.parse_args()

    z = np.load(args.pass1)
    n = int(z["n"])
    det_frame, det_box = z["det_frame"], z["det_box"]
    boxes_of = [[] for _ in range(n)]
    for f, b in zip(det_frame, det_box):
        boxes_of[f].append(b)

    cap = cv2.VideoCapture(args.video)
    clahe = cv2.createCLAHE(3.0, (8, 8))
    sift = cv2.SIFT_create(1500)

    bright = np.zeros(n)
    inc = np.tile(np.eye(3), (n, 1, 1))  # inc[t]: frame t-1 -> frame t
    inc_ok = np.zeros(n, bool)
    feats = {}
    prev = prev_m = None
    for t in range(n):
        ok, f = cap.read()
        if not ok:
            n = t
            break
        g0 = cv2.cvtColor(cv2.resize(f, None, fx=SC, fy=SC, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        m = people_mask(g0.shape, boxes_of[t])
        bright[t] = g0[m > 0].mean() if (m > 0).any() else g0.mean()
        g = clahe.apply(g0)
        if prev is not None:
            mm = cv2.bitwise_and(prev_m, m)
            p0 = cv2.goodFeaturesToTrack(prev, 500, 0.01, 6, mask=mm)
            if p0 is not None and len(p0) >= 12:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, winSize=(21, 21), maxLevel=3)
                good = st.ravel() == 1
                if good.sum() >= 10:
                    A, inl = cv2.estimateAffinePartial2D(p0[good], p1[good], ransacReprojThreshold=1.0)
                    if A is not None and inl.sum() >= 10:
                        inc[t] = to3(A)
                        inc_ok[t] = True
        if t % args.anchor_every == 0:
            k, d = sift.detectAndCompute(g, m)
            feats[t] = (np.float32([p.pt for p in k]) if k else np.zeros((0, 2), np.float32), d)
        prev, prev_m = g, m
        if (t + 1) % 600 == 0:
            print(f"[camera] read {t + 1}/{n}", flush=True)
    bright = bright[:n]

    # ---- lighting classes (bright / dark)
    lo, hi = np.percentile(bright, 10), np.percentile(bright, 90)
    two = hi - lo > 25
    cls = (median_filter(bright, size=31, mode="nearest") < (lo + hi) / 2).astype(int) if two else np.zeros(n, int)
    print(f"[camera] brightness {lo:.0f}..{hi:.0f}, lighting classes: {np.bincount(cls)}")

    # ---- reference per class: anchor frame with the most background keypoints
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    refs = {}
    for c in np.unique(cls):
        cands = [t for t in feats if cls[t] == c and feats[t][1] is not None]
        refs[c] = max(cands, key=lambda t: len(feats[t][0]))
    print(f"[camera] reference frames: {refs}")

    G = {}  # anchor frame -> 3x3 (frame -> class reference), half-res pixels
    for t, (pts, d) in feats.items():
        c = cls[t]
        r = refs[c]
        if t == r:
            G[t] = np.eye(3)
            continue
        rp, rd = feats[r]
        if d is None or rd is None or len(d) < 10:
            continue
        mm = flann.knnMatch(d, rd, k=2)
        good = [a for a, *b in mm if b and a.distance < 0.75 * b[0].distance]
        if len(good) < args.min_inliers:
            continue
        p = pts[[x.queryIdx for x in good]]
        q = rp[[x.trainIdx for x in good]]
        A, inl = cv2.estimateAffinePartial2D(p, q, ransacReprojThreshold=2.0)
        if A is None:
            continue
        ni = int(inl.sum())
        s = np.hypot(A[0, 0], A[1, 0])
        if ni >= args.min_inliers and ni >= 0.25 * len(good) and 0.4 < s < 2.5:
            G[t] = to3(A)
    print(f"[camera] anchors: {len(G)}/{len(feats)}")

    # cumulative increments: C[t] maps frame 0 -> frame t
    C = np.zeros((n, 3, 3))
    C[0] = np.eye(3)
    for t in range(1, n):
        C[t] = inc[t] @ C[t - 1]

    def chain(a, b):
        """frame a -> frame b via increments."""
        return C[b] @ np.linalg.inv(C[a])

    # reject anchors inconsistent with their neighbours (median of params over a window)
    at = np.array(sorted(G))
    P = np.array([par(G[t]) for t in at])
    keep = np.ones(len(at), bool)
    for c in np.unique(cls):
        idx = np.flatnonzero(cls[at] == c)
        if len(idx) > 5:
            med = median_filter(P[idx], size=(9, 1), mode="nearest")
            dev = np.abs(P[idx] - med)
            bad = (dev[:, 0] > 0.06) | (dev[:, 2] > 25) | (dev[:, 3] > 25) | (dev[:, 1] > 0.05)
            keep[idx[bad]] = False
    for t in at[~keep]:
        del G[t]
    print(f"[camera] anchors after outlier rejection: {len(G)}")

    # ---- link class references to the main reference (class of the most frames)
    main = int(np.bincount(cls).argmax())
    L = {main: np.eye(3)}
    for c in np.unique(cls):
        if c == main:
            continue
        ests = []
        changes = np.flatnonzero(np.diff(cls) != 0) + 1
        for t in changes:
            for a_side, b_side in ((main, c), (c, main)):
                if cls[t - 1] != a_side or cls[t] != b_side:
                    continue
                a = max([x for x in G if x < t and cls[x] == a_side and t - x < 300], default=None)
                b = min([x for x in G if x >= t and cls[x] == b_side and x - t < 300], default=None)
                if a is None or b is None:
                    continue
                # point in ref_c -> ref_main
                if a_side == main:   # a in main, b in c
                    est = G[a] @ np.linalg.inv(chain(a, b)) @ np.linalg.inv(G[b])
                else:                # a in c, b in main
                    est = G[b] @ chain(a, b) @ np.linalg.inv(G[a])
                ests.append(par(est))
        if ests:
            L[c] = mat(np.median(np.array(ests), axis=0))
            print(f"[camera] class {c} linked to main with {len(ests)} transitions: {np.round(par(L[c]), 3)}")
        else:
            L[c] = np.eye(3)
            print(f"[camera] WARNING: class {c} could not be linked; assuming same framing")

    # ---- absolute transform for every frame (frame -> main reference)
    A_abs = {t: L[cls[t]] @ G[t] for t in G}
    anchors = np.array(sorted(A_abs))
    T = np.zeros((n, 4))
    conf = np.zeros(n, np.float32)
    for t in range(n):
        if t in A_abs:
            T[t] = par(A_abs[t])
            conf[t] = 1
            continue
        i = np.searchsorted(anchors, t)
        a = anchors[i - 1] if i > 0 else None
        b = anchors[i] if i < len(anchors) else None
        est, wts = [], []
        if a is not None:
            est.append(par(A_abs[a] @ np.linalg.inv(chain(a, t))))
            wts.append(1.0 / (t - a))
        if b is not None:
            est.append(par(A_abs[b] @ chain(t, b)))
            wts.append(1.0 / (b - t))
        if est:
            w = np.array(wts) / sum(wts)
            T[t] = (np.array(est) * w[:, None]).sum(0)
            conf[t] = 0.5
        else:
            conf[t] = 0
            T[t] = np.nan
    # hold through long un-anchored stretches
    for k in range(4):
        col = T[:, k]
        good = ~np.isnan(col)
        if good.any():
            T[:, k] = np.interp(np.arange(n), np.flatnonzero(good), col[good])
    T = savgol_filter(T, 9, 2, axis=0)

    # half-res -> full-res pixels (translation scales, linear part unchanged)
    T_full = T.copy()
    T_full[:, 2:] /= SC
    np.savez_compressed(args.out, T=T_full, conf=conf, cls=cls, bright=bright, ref=refs[main])
    s = np.exp(T_full[:, 0])
    print(f"[camera] wrote {args.out}: zoom factor range {s.min():.2f}..{s.max():.2f} (frame->reference), "
          f"anchored {np.mean(conf == 1) * 100:.0f}%, bridged {np.mean(conf == 0.5) * 100:.0f}%")


if __name__ == "__main__":
    sys.exit(main())
