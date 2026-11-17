#!/usr/bin/env python3
"""Find one dancer (by hair colour, default: blonde) in a group dance video and extract
that dancer's 2D + 3D pose for every frame.

Pass 1  every dancer: YOLOX person boxes on every frame; every N frames also a
        MediaPipe pose on each person's crop, hair segmentation of the head (hair
        colour) and a clothing colour histogram of the torso.
Select  boxes are linked into tracklets; the tracklets whose hair is the most
        "blonde" (light + yellow) are the seed, then neighbouring tracklets are
        stitched on by position, hair colour and clothes.
Pass 2  the target is re-estimated on a crop centred on them every frame (heavy
        model, video mode), which is accurate and fills frames pass 1 missed.

Output: .npz with per-frame 2D keypoints (full-frame pixels), MediaPipe world landmarks
(metres, hip centred) and box of the target, plus a debug overlay video if asked.
"""
import argparse
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (L_EAR, L_EYE, L_HIP, L_SH, NOSE, NUM_LM, R_EAR, R_EYE, R_HIP,  # noqa: E402
                    R_SH, bbox_from_kp, body_center, nan_interp)

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- models

def pose_landmarker(model_path, num_poses, video=True, conf=0.3):
    opts = vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_poses=num_poses,
        min_pose_detection_confidence=conf,
        min_pose_presence_confidence=conf,
        min_tracking_confidence=conf,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def hair_segmenter(model_path):
    opts = vision.ImageSegmenterOptions(
        base_options=mpt.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    return vision.ImageSegmenter.create_from_options(opts)


class PersonDetector:
    """YOLOX (ONNX, COCO) person detector."""

    def __init__(self, path, size=640, threads=0):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.size = size
        g, s = [], []
        for st in (8, 16, 32):
            h = w = size // st
            xv, yv = np.meshgrid(np.arange(w), np.arange(h))
            g.append(np.stack((xv, yv), 2).reshape(-1, 2))
            s.append(np.full((h * w, 1), st))
        self.grid = np.concatenate(g).astype(np.float32)
        self.stride = np.concatenate(s).astype(np.float32)

    def __call__(self, img, thr=0.4, nms=0.5):
        size = self.size
        r = min(size / img.shape[0], size / img.shape[1])
        pad = np.full((size, size, 3), 114, np.uint8)
        rs = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR)
        pad[:rs.shape[0], :rs.shape[1]] = rs
        x = np.ascontiguousarray(pad.transpose(2, 0, 1)[None], np.float32)
        out = self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0][0]
        score = out[:, 4] * out[:, 5]  # objectness * P(person)
        m = score > thr
        if not m.any():
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        xy = (out[m, :2] + self.grid[m]) * self.stride[m]
        wh = np.exp(out[m, 2:4]) * self.stride[m]
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / r
        score = score[m]
        keep = cv2.dnn.NMSBoxes([[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
                                score.tolist(), thr, nms)
        keep = np.array(keep, int).ravel()
        return boxes[keep].astype(np.float32), score[keep].astype(np.float32)


def to_mp(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def unpack(res, sx, sy, ox=0.0, oy=0.0):
    """MediaPipe result -> list of (kp (33,3) pixel+visibility, world (33,3))."""
    out = []
    for pl, wl in zip(res.pose_landmarks, res.pose_world_landmarks):
        kp = np.array([[p.x * sx + ox, p.y * sy + oy, p.visibility if p.visibility is not None else 1.0]
                       for p in pl], dtype=np.float32)
        wp = np.array([[p.x, p.y, p.z] for p in wl], dtype=np.float32)
        out.append((kp, wp))
    return out


def box_iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# --------------------------------------------------------------------------- appearance

def head_crop_box(kp):
    """Square box (x0, y0, side) around the head, including all of the hair."""
    face = kp[[NOSE, L_EYE, R_EYE, L_EAR, R_EAR], :2]
    sh_w = np.linalg.norm(kp[L_SH, :2] - kp[R_SH, :2])
    w = max(np.linalg.norm(kp[L_EAR, :2] - kp[R_EAR, :2]),
            2.2 * np.linalg.norm(kp[L_EYE, :2] - kp[R_EYE, :2]),
            0.45 * sh_w)
    c = face.mean(0)
    sh_mid = (kp[L_SH, :2] + kp[R_SH, :2]) / 2
    up = c - sh_mid
    n = np.linalg.norm(up)
    up = up / n if n > 1e-3 else np.array([0.0, -1.0])
    c = c + up * 0.3 * w
    side = 2.6 * w
    return c[0] - side / 2, c[1] - side / 2, side


def crop_square(frame, x0, y0, side, out=None):
    """Crop a square (may extend outside the frame, padded with black)."""
    h, w = frame.shape[:2]
    side = max(int(round(side)), 2)
    x0, y0 = int(round(x0)), int(round(y0))
    crop = np.zeros((side, side, 3), np.uint8)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, w), min(y0 + side, h)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]
    if out is not None:
        crop = cv2.resize(crop, (out, out), interpolation=cv2.INTER_CUBIC if side < out else cv2.INTER_AREA)
    return crop, x0, y0, side


def hair_color(frame, kp, seg):
    """Median Lab colour of the hair pixels of one person. Returns (L, a, b, n_pixels) or None."""
    x0, y0, side = head_crop_box(kp)
    if side < 10:
        return None
    crop, _, _, _ = crop_square(frame, x0, y0, side, out=256)
    res = seg.segment(to_mp(crop))
    masks = res.confidence_masks
    if not masks:
        return None
    hair = masks[-1].numpy_view()
    if hair.ndim == 3:
        hair = hair[..., 0]
    m = hair > 0.5
    if m.sum() < 80:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[m].astype(np.float32)
    L, a, b = np.median(lab, axis=0)
    return np.array([L, a, b, m.sum()], np.float32)


def blonde_metric(lab):
    """Higher = lighter and more yellow hair. Black hair ~40-70, blonde ~150-220."""
    L, b = lab[..., 0], lab[..., 2]
    return L + 1.2 * np.maximum(b - 128.0, 0.0)


def clothes_hist(frame, kp):
    """Normalised HSV histogram of the torso (shoulders-hips quad)."""
    poly = kp[[L_SH, R_SH, R_HIP, L_HIP], :2].astype(np.int32)
    x, y, w, h = cv2.boundingRect(poly)
    if w < 4 or h < 4:
        return None
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    roi = frame[y0:y1, x0:x1]
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly - [x0, y0]], 255)
    if mask.sum() < 255 * 30:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hs = cv2.calcHist([hsv], [0, 1], mask, [12, 4], [0, 180, 0, 256]).ravel()
    v = cv2.calcHist([hsv], [2], mask, [6], [0, 256]).ravel()
    return np.concatenate([hs / max(hs.sum(), 1), v / max(v.sum(), 1)]).astype(np.float32)


def hist_sim(a, b):
    return float(np.minimum(a, b).sum() / 2.0)  # two normalised parts -> [0, 1]


def pose_in_box(lm, frame, box, crop_px=256):
    """MediaPipe pose (image mode) of the person inside `box`."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * 1.2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    crop, ox, oy, side_i = crop_square(frame, cx - side / 2, cy - side / 2, side, out=crop_px)
    res = lm.detect(to_mp(crop))
    s = side_i / crop_px
    best, best_q = None, -1e9
    for kp, wp in unpack(res, crop_px * s, crop_px * s, ox, oy):
        b2 = bbox_from_kp(kp, W, H)
        q = box_iou(b2, box) - np.linalg.norm(body_center(kp) - [cx, cy]) / max(y1 - y0, 1)
        if q > best_q:
            best, best_q = (kp, wp), q
    if best is None or best_q < 0.05:
        return None
    return best


# --------------------------------------------------------------------------- pass 1

def pass1(video, args):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_max = n_total if args.max_frames <= 0 else min(n_total, args.max_frames)
    det = PersonDetector(args.detector_model)
    lm = pose_landmarker(args.model_pass1, 2, video=False, conf=0.3)
    seg = hair_segmenter(args.hair_model)

    rows = dict(frame=[], box=[], score=[], kp=[], world=[], hair=[], hist=[])
    nan_kp = np.full((NUM_LM, 3), np.nan, np.float32)
    t0 = time.time()
    i = 0
    while i < n_max:
        ok, frame = cap.read()
        if not ok:
            break
        boxes, scores = det(frame, thr=args.det_thr)
        for b, s in zip(boxes, scores):
            if (b[3] - b[1]) < args.min_height * H:
                continue
            kp = wp = None
            hair = hist = None
            if i % args.hair_every == 0:
                r = pose_in_box(lm, frame, b)
                if r is not None:
                    kp, wp = r
                    hair = hair_color(frame, kp, seg)
                    hist = clothes_hist(frame, kp)
            rows["frame"].append(i)
            rows["box"].append(b)
            rows["score"].append(s)
            rows["kp"].append(kp if kp is not None else nan_kp)
            rows["world"].append(wp if wp is not None else nan_kp)
            rows["hair"].append(hair if hair is not None else np.full(4, np.nan, np.float32))
            rows["hist"].append(hist if hist is not None else np.full(54, np.nan, np.float32))
        i += 1
        if i % 200 == 0:
            el = time.time() - t0
            print(f"[pass1] {i}/{n_max} frames  {i / el:.1f} fps  dets={len(rows['frame'])}", flush=True)
    cap.release()
    lm.close()
    seg.close()
    return dict(fps=fps, W=W, H=H, n=i,
                det_frame=np.array(rows["frame"], np.int32),
                det_box=np.array(rows["box"], np.float32).reshape(-1, 4),
                det_score=np.array(rows["score"], np.float32),
                det_kp=np.array(rows["kp"], np.float32).reshape(-1, NUM_LM, 3),
                det_world=np.array(rows["world"], np.float32).reshape(-1, NUM_LM, 3),
                det_hair=np.array(rows["hair"], np.float32).reshape(-1, 4),
                det_hist=np.array(rows["hist"], np.float32).reshape(-1, 54))


# --------------------------------------------------------------------------- tracking

def box_geom(p1):
    b = p1["det_box"]
    centers = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
    heights = np.maximum(b[:, 3] - b[:, 1], 20)
    return centers, heights


def build_tracklets(p1, max_gap=8):
    n = p1["n"]
    fr = p1["det_frame"]
    boxes = p1["det_box"]
    centers, heights = box_geom(p1)
    hist = p1["det_hist"]
    by_frame = [[] for _ in range(n)]
    for d, f in enumerate(fr):
        by_frame[f].append(d)

    tracks = []
    tid_of = np.full(len(fr), -1, np.int32)
    for t in range(n):
        cands = by_frame[t]
        live = [k for k, tr in enumerate(tracks) if t - tr["last"] <= max_gap]
        if live and cands:
            cost = np.full((len(live), len(cands)), 1e6)
            for a, k in enumerate(live):
                tr = tracks[k]
                gap = t - tr["last"]
                pred = tr["c"] + tr["v"] * gap
                pbox = tr["box"] + np.r_[tr["v"], tr["v"]] * gap
                for b, d in enumerate(cands):
                    dist = np.linalg.norm(centers[d] - pred) / max(tr["h"], heights[d])
                    if dist > 0.3 + 0.04 * gap:
                        continue
                    iou = box_iou(pbox, boxes[d])
                    hs = 0.0
                    if not np.isnan(hist[d, 0]) and tr["hist"] is not None:
                        hs = 1.0 - hist_sim(hist[d], tr["hist"])
                    dh = abs(np.log(heights[d] / tr["h"]))
                    cost[a, b] = dist + 0.5 * (1 - iou) + 0.4 * hs + 0.3 * dh
            ra, cb = linear_sum_assignment(cost)
            used = set()
            for a, b in zip(ra, cb):
                if cost[a, b] >= 1e5:
                    continue
                k, d = live[a], cands[b]
                tr = tracks[k]
                gap = t - tr["last"]
                v_new = (centers[d] - tr["c"]) / gap
                tr["v"] = 0.6 * tr["v"] + 0.4 * v_new
                tr["c"], tr["box"], tr["last"] = centers[d], boxes[d].copy(), t
                tr["h"] = 0.7 * tr["h"] + 0.3 * heights[d]
                if not np.isnan(hist[d, 0]):
                    tr["hist"] = hist[d] if tr["hist"] is None else 0.85 * tr["hist"] + 0.15 * hist[d]
                tr["dets"].append(d)
                tid_of[d] = k
                used.add(d)
            cands = [d for d in cands if d not in used]
        for d in cands:
            tracks.append(dict(dets=[d], last=t, c=centers[d], v=np.zeros(2), h=heights[d], box=boxes[d].copy(),
                               hist=None if np.isnan(hist[d, 0]) else hist[d].copy()))
            tid_of[d] = len(tracks) - 1
    return tracks, tid_of, centers, heights


def tracklet_stats(p1, tracks):
    hair = p1["det_hair"]
    hist = p1["det_hist"]
    stats = []
    for k, tr in enumerate(tracks):
        d = np.array(tr["dets"])
        hv = hair[d]
        ok = ~np.isnan(hv[:, 0])
        bm = blonde_metric(hv[ok]) if ok.any() else np.array([])
        hh = hist[d]
        hok = ~np.isnan(hh[:, 0])
        stats.append(dict(
            id=k, n=len(d), f0=int(p1["det_frame"][d[0]]), f1=int(p1["det_frame"][d[-1]]),
            n_hair=int(ok.sum()),
            blonde=float(np.median(bm)) if len(bm) else np.nan,
            lab=np.median(hv[ok, :3], axis=0) if ok.any() else np.full(3, np.nan),
            hist=hh[hok].mean(0) if hok.any() else None,
        ))
    return stats


def select_target(p1, tracks, stats, centers, heights, args):
    fr = p1["det_frame"]
    n = p1["n"]
    fps = p1["fps"]
    min_len = max(5, int(0.3 * fps))
    cand = [s for s in stats if s["n"] >= min_len and s["n_hair"] >= 2 and not np.isnan(s["blonde"])]
    if not cand:
        raise SystemExit("no usable tracklets found")
    others = float(np.median([s["blonde"] for s in cand]))

    # ---- seed tracklet
    if args.hint:
        ht, hx, hy = [float(v) for v in args.hint.split(",")]
        hf = int(round(ht * fps))
        pt = np.array([hx * p1["W"], hy * p1["H"]])
        best, bd = None, 1e9
        for s in stats:
            for d in tracks[s["id"]]["dets"]:
                if abs(int(fr[d]) - hf) <= 2:
                    dist = np.linalg.norm(centers[d] - pt)
                    if dist < bd:
                        best, bd = s, dist
        if best is None:
            raise SystemExit("--hint did not match any detected dancer")
        seed = best
        print(f"[select] seed from hint: tracklet {seed['id']}")
    else:
        # evidence = how much lighter than the others x how many hair samples back it up
        seed = max(cand, key=lambda s: (s["blonde"] - others) * min(s["n_hair"], 60))
        print(f"[select] seed tracklet {seed['id']} blonde={seed['blonde']:.1f} (median of all {others:.1f}) "
              f"len={seed['n']} frames {seed['f0']}-{seed['f1']}")

    seed_b = seed["blonde"]
    base_b = others
    span = max(seed_b - base_b, 15.0)
    # when the target's hair is not much lighter than the others, rely more on clothing colour
    w_hair = args.w_hair * float(np.clip((seed_b - base_b) / 40.0, 0.15, 1.0))

    def score(s):
        hs = 0.5
        if s["hist"] is not None and seed["hist"] is not None:
            hs = hist_sim(s["hist"], seed["hist"])
        bz = 0.3 if np.isnan(s["blonde"]) else np.clip((s["blonde"] - base_b) / span, -1, 1.5)
        return w_hair * bz + args.w_clothes * (hs - 0.5) * 2

    scored = sorted(((score(s), s) for s in stats if s["n"] >= 3), key=lambda x: -x[0])
    for sc, s in scored[:12]:
        print(f"   tracklet {s['id']:4d}  frames {s['f0']:5d}-{s['f1']:5d} n={s['n']:5d} "
              f"blonde={s['blonde']:6.1f}  score={sc:5.2f}")

    seed_score = score(seed)
    app = {s["id"]: score(s) for s in stats}
    pool = [s for s in stats if s["n"] >= 3]
    max_gap = int(args.stitch_gap * fps)

    def ends(s, last):
        d = tracks[s["id"]]["dets"][-1 if last else 0]
        return int(fr[d]), centers[d], heights[d]

    def stitch(cur, forward):
        """Next tracklet after (or before) `cur`: near in space and time, similar look."""
        f_end, c_end, h_end = ends(cur, forward)
        best, best_cost = None, 1e9
        for s in pool:
            if s["id"] in used:
                continue
            f_s, c_s, h_s = ends(s, not forward)
            gap = (f_s - f_end) if forward else (f_end - f_s)
            if gap < -3 or gap > max_gap:
                continue
            if forward and s["f1"] <= cur["f1"]:
                continue
            if not forward and s["f0"] >= cur["f0"]:
                continue
            g = max(gap, 1)
            dist = np.linalg.norm(c_s - c_end) / max(h_end, h_s)
            if dist > 0.45 + 2.0 * g / fps:
                continue
            if app[s["id"]] < args.stitch_thr * seed_score:
                continue
            cost = dist + 0.6 * (seed_score - app[s["id"]]) / max(seed_score, 1e-3) + 0.3 * g / fps
            if cost < best_cost:
                best, best_cost = s, cost
        if best is not None:
            return best
        # re-identification jump: the dancer was lost; take the next clearly matching tracklet
        later = [s for s in pool if s["id"] not in used and s["n"] >= min_len
                 and app[s["id"]] >= args.select_thr * seed_score
                 and ((s["f0"] > cur["f1"] - 3) if forward else (s["f1"] < cur["f0"] + 3))]
        if not later:
            return None
        return min(later, key=lambda s: (s["f0"] if forward else -s["f1"]))

    used = {seed["id"]}
    chain = [seed]
    for forward in (True, False):
        cur = seed
        while True:
            nxt = stitch(cur, forward)
            if nxt is None:
                break
            used.add(nxt["id"])
            chain.append(nxt)
            cur = nxt

    owner = np.full(n, -1, np.int32)  # detection id per frame
    for s in sorted(chain, key=lambda s: -app[s["id"]]):
        for d in tracks[s["id"]]["dets"]:
            if owner[fr[d]] < 0:
                owner[fr[d]] = d

    # safety net: drop stretches where the chosen person's hair is clearly not the target's
    hair = p1["det_hair"]
    sel = np.flatnonzero(owner >= 0)
    hz = np.full(n, np.nan)
    hv = hair[owner[sel]]
    ok = ~np.isnan(hv[:, 0])
    hz[sel[ok]] = (blonde_metric(hv[ok]) - base_b) / span
    samp = np.flatnonzero(~np.isnan(hz))
    dropped = 0
    if seed_b - base_b < 40:
        print("[select] hair colours are not very distinct -> hair safety check skipped")
    elif len(samp) > 9:
        zs = median_filter(hz[samp], size=9, mode="nearest")
        for f in samp[zs < args.hair_drop]:
            lo, hi = max(f - args.hair_every, 0), min(f + args.hair_every + 1, n)
            dropped += int((owner[lo:hi] >= 0).sum())
            owner[lo:hi] = -1
    cov = (owner >= 0).mean()
    print(f"[select] chain of {len(chain)} tracklets, dropped {dropped} frames by hair check, "
          f"coverage {cov * 100:.1f}% of frames")
    return owner, seed


# --------------------------------------------------------------------------- pass 2

def plan_crops(p1, owner, max_gap_s=2.0):
    """Smoothed square crop (cx, cy, side) per frame around the target, NaN where unknown."""
    n, fps = p1["n"], p1["fps"]
    boxes = p1["det_box"]
    raw = np.full((n, 3), np.nan)
    for f in range(n):
        d = owner[f]
        if d < 0:
            continue
        b = boxes[d]
        raw[f] = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], b[3] - b[1]) * 1.3)
    filled = nan_interp(raw, max_gap=int(max_gap_s * fps))
    good = ~np.isnan(filled[:, 0])
    out = filled.copy()
    win = int(fps * 0.3) | 1
    idx = np.flatnonzero(good)
    if len(idx):
        for r in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(r) > win:
                out[r, 0] = savgol_filter(filled[r, 0], win, 2)
                out[r, 1] = savgol_filter(filled[r, 1], win, 2)
                out[r, 2] = np.maximum(savgol_filter(filled[r, 2], win, 1), filled[r, 2])
    return out


def pass2(video, p1, owner, crops, args):
    """Heavy-model pose on a crop around the target, every frame.

    Image mode (detector runs every frame) with up to 3 people per crop: the pose whose
    box agrees with the target's YOLOX box is kept, so a dancer in front of the target
    is not picked up by mistake. Other dancers' boxes are greyed out in the crop.
    """
    n = p1["n"]
    lm = pose_landmarker(args.model_pass2, 3, video=False, conf=0.3)
    kp_out = np.full((n, NUM_LM, 3), np.nan, np.float32)
    w_out = np.full((n, NUM_LM, 3), np.nan, np.float32)
    src = np.zeros(n, np.int8)
    by_frame = {}
    for d, f in enumerate(p1["det_frame"]):
        by_frame.setdefault(int(f), []).append(d)
    cap = cv2.VideoCapture(video)
    W, H = p1["W"], p1["H"]
    CROP = args.crop_size
    t0 = time.time()
    for f in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        d = owner[f]
        box = p1["det_box"][d] if d >= 0 else None
        if not np.isnan(crops[f, 0]):
            cx, cy, side = crops[f]
            work = frame
            if box is not None:
                others = [p1["det_box"][o] for o in by_frame.get(f, []) if o != d]
                if others:
                    work = frame.copy()
                    keep = work[int(max(box[1], 0)):int(min(box[3], H)), int(max(box[0], 0)):int(min(box[2], W))].copy()
                    for ob in others:
                        x0o, y0o = int(max(ob[0], 0)), int(max(ob[1], 0))
                        x1o, y1o = int(min(ob[2], W)), int(min(ob[3], H))
                        work[y0o:y1o, x0o:x1o] = 128
                    work[int(max(box[1], 0)):int(min(box[3], H)), int(max(box[0], 0)):int(min(box[2], W))] = keep
            crop, x0, y0, side_i = crop_square(work, cx - side / 2, cy - side / 2, side, out=CROP)
            res = lm.detect(to_mp(crop))
            s = side_i / CROP
            best, best_cost = None, 1e9
            for kp, wp in unpack(res, CROP * s, CROP * s, x0, y0):
                c = body_center(kp)
                if box is not None:
                    bh = max(box[3] - box[1], 1)
                    kb = bbox_from_kp(kp, W, H)
                    cost = np.linalg.norm(c - [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]) / bh \
                        + 0.5 * (1 - box_iou(kb, box))
                    ok_match = cost < 0.45
                else:
                    cost = np.linalg.norm(c - [cx, cy]) / side
                    ok_match = cost < 0.12
                if ok_match and cost < best_cost:
                    best, best_cost = (kp, wp), cost
            if best is not None:
                kp_out[f], w_out[f], src[f] = best[0], best[1], 2
        if src[f] == 0 and d >= 0 and not np.isnan(p1["det_kp"][d, 0, 0]):
            kp_out[f], w_out[f], src[f] = p1["det_kp"][d], p1["det_world"][d], 1
        if (f + 1) % 300 == 0:
            print(f"[pass2] {f + 1}/{n}  {(f + 1) / (time.time() - t0):.1f} fps", flush=True)
    cap.release()
    lm.close()
    return kp_out, w_out, src


# --------------------------------------------------------------------------- debug video

def debug_video(video, out_path, p1, kp, src, owner):
    conns = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
    cap = cv2.VideoCapture(video)
    W, H, fps = p1["W"], p1["H"], p1["fps"]
    scale = min(1.0, 960 / W)
    size = (int(W * scale) // 2 * 2, int(H * scale) // 2 * 2)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    by_frame = {}
    for d, f in enumerate(p1["det_frame"]):
        by_frame.setdefault(int(f), []).append(d)
    for f in range(p1["n"]):
        ok, frame = cap.read()
        if not ok:
            break
        img = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        for d in by_frame.get(f, []):
            b = p1["det_box"][d] * scale
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (120, 120, 120), 1)
        if src[f] > 0:
            k = kp[f, :, :2] * scale
            col = (0, 255, 255) if src[f] == 2 else (0, 160, 255)
            for a, b in conns:
                cv2.line(img, tuple(np.int32(k[a])), tuple(np.int32(k[b])), col, 2, cv2.LINE_AA)
        if owner[f] >= 0:
            b = p1["det_box"][owner[f]] * scale
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 2)
            cv2.putText(img, "TARGET", (int(b[0]), int(b[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(img, f"{f / fps:6.2f}s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        vw.write(img)
    vw.release()
    cap.release()


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("-o", "--out", default="motion.npz")
    ap.add_argument("--models", default=os.path.join(HERE, "models"))
    ap.add_argument("--det-thr", type=float, default=0.4, help="person detection score threshold")
    ap.add_argument("--min-height", type=float, default=0.12, help="ignore people smaller than this x frame height")
    ap.add_argument("--hair-every", type=int, default=3, help="pose/hair/clothes sampling interval (frames)")
    ap.add_argument("--crop-size", type=int, default=384)
    ap.add_argument("--hint", default="", help="t_sec,x,y (x,y in 0..1) to point at the target manually")
    ap.add_argument("--w-hair", type=float, default=1.0)
    ap.add_argument("--w-clothes", type=float, default=0.6)
    ap.add_argument("--select-thr", type=float, default=0.6,
                    help="re-identification: fraction of the seed score a far-away tracklet needs")
    ap.add_argument("--stitch-thr", type=float, default=0.2,
                    help="stitching: fraction of the seed score a nearby tracklet needs")
    ap.add_argument("--stitch-gap", type=float, default=1.0, help="max seconds between stitched tracklets")
    ap.add_argument("--hair-drop", type=float, default=0.3,
                    help="drop frames whose (smoothed) hair score falls below this (0=others, 1=seed)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-pass2", action="store_true")
    ap.add_argument("--debug-video", default="")
    ap.add_argument("--cache", default="", help="pass-1 cache .npz (reused if it exists)")
    args = ap.parse_args()
    args.detector_model = os.path.join(args.models, "yolox_s.onnx")
    args.model_pass1 = os.path.join(args.models, "pose_landmarker_full.task")
    args.model_pass2 = os.path.join(args.models, "pose_landmarker_heavy.task")
    args.hair_model = os.path.join(args.models, "hair_segmenter.tflite")

    if args.cache and os.path.exists(args.cache):
        z = np.load(args.cache)
        p1 = {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
        print(f"[pass1] loaded cache {args.cache}")
    else:
        p1 = pass1(args.video, args)
        if args.cache:
            np.savez_compressed(args.cache, **p1)

    tracks, tid_of, centers, heights = build_tracklets(p1)
    stats = tracklet_stats(p1, tracks)
    owner, seed = select_target(p1, tracks, stats, centers, heights, args)

    if args.no_pass2:
        n = p1["n"]
        kp = np.full((n, NUM_LM, 3), np.nan, np.float32)
        wl = np.full((n, NUM_LM, 3), np.nan, np.float32)
        src = np.zeros(n, np.int8)
        m = owner >= 0
        m &= ~np.isnan(p1["det_kp"][np.maximum(owner, 0), 0, 0])
        kp[m], wl[m], src[m] = p1["det_kp"][owner[m]], p1["det_world"][owner[m]], 1
    else:
        crops = plan_crops(p1, owner)
        kp, wl, src = pass2(args.video, p1, owner, crops, args)

    box = np.full((p1["n"], 4), np.nan, np.float32)
    box[owner >= 0] = p1["det_box"][owner[owner >= 0]]
    print(f"[done] target box in {(owner >= 0).mean() * 100:.1f}% of frames, pose in {(src > 0).mean() * 100:.1f}% "
          f"(refined {(src == 2).mean() * 100:.1f}%)")
    np.savez_compressed(args.out, fps=p1["fps"], W=p1["W"], H=p1["H"], n=p1["n"],
                        kp2d=kp, world=wl, src=src, box=box, seed_lab=seed["lab"])
    if args.debug_video:
        debug_video(args.video, args.debug_video, p1, kp, src, owner)
        print(f"[debug] wrote {args.debug_video}")


if __name__ == "__main__":
    main()
