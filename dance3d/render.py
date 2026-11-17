#!/usr/bin/env python3
"""Render the 3D motion of one dancer as an easy-to-follow video.

Layout (1920x1080):
  +-------------------------------+---------------------+
  | title / time                                        |
  +-------------------------------+---------------------+
  |                               |  front view         |
  |   3D view (diagonal, follows) |  (same as video)    |
  |   + floor position map        +---------------------+
  |   + hand / foot trails        |  back view          |
  |                               |  (same as you: for  |
  |                               |   practising)       |
  +-------------------------------+---------------------+

Dancer's LEFT arm/leg = blue, RIGHT arm/leg = orange. Hair = blonde, face = skin,
so the facing direction of the head is always visible. The arrow on the floor shows
which way the body faces.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import savgol_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (L_AN, L_EAR, L_EL, L_FT, L_HEEL, L_HIP, L_INDEX, L_KN,  # noqa: E402
                    L_PINKY, L_SH, L_WR, NOSE, R_AN, R_EAR, R_EL, R_FT, R_HEEL,
                    R_HIP, R_INDEX, R_KN, R_PINKY, R_SH, R_WR)

FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_R = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# colours are BGR for OpenCV
BG_TOP = np.array([34, 26, 22], np.float32)
BG_BOT = np.array([52, 40, 33], np.float32)
GRID = np.array([150, 130, 115], np.float32)
COL_L = (245, 169, 63)      # dancer's left: blue
COL_R = (61, 138, 255)      # dancer's right: orange
COL_TORSO_F = (240, 236, 233)
COL_TORSO_B = (170, 158, 150)
COL_HAIR = (102, 199, 232)  # blonde
COL_SKIN = (168, 200, 240)
COL_OUT = (18, 14, 12)
COL_ACCENT = (102, 199, 232)

W_OUT, H_OUT = 1920, 1080
HEADER = 72
MAIN = (0, HEADER, 1200, H_OUT - HEADER)          # x, y, w, h
FRONT = (1200, HEADER, 720, (H_OUT - HEADER) // 2)
BACK = (1200, HEADER + (H_OUT - HEADER) // 2, 720, (H_OUT - HEADER) - (H_OUT - HEADER) // 2)
SS = 2  # supersampling


def font(size, bold=True):
    return ImageFont.truetype(FONT_B if bold else FONT_R, size, index=0)


# --------------------------------------------------------------------------- camera

class Camera:
    def __init__(self, eye, target, w, h, fov_y=34.0):
        self.eye = np.asarray(eye, np.float64)
        fwd = np.asarray(target, np.float64) - self.eye
        self.fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(self.fwd, [0.0, 1.0, 0.0])
        self.right = right / np.linalg.norm(right)
        self.up = np.cross(self.right, self.fwd)
        self.w, self.h = w, h
        self.f = (h / 2) / np.tan(np.radians(fov_y) / 2)

    def cam(self, p):
        d = np.asarray(p, np.float64) - self.eye
        return np.stack([d @ self.right, d @ self.up, d @ self.fwd], axis=-1)

    def project(self, p):
        c = self.cam(p)
        z = np.maximum(c[..., 2], 1e-3)
        u = self.w / 2 + self.f * c[..., 0] / z
        v = self.h / 2 - self.f * c[..., 1] / z
        return np.stack([u, v], -1), c[..., 2]


# --------------------------------------------------------------------------- body model

LIMBS = [  # (a, b, radius m, side)
    (L_SH, L_EL, 0.048, "L"), (L_EL, L_WR, 0.040, "L"), (L_WR, "LH", 0.032, "L"),
    (R_SH, R_EL, 0.048, "R"), (R_EL, R_WR, 0.040, "R"), (R_WR, "RH", 0.032, "R"),
    (L_HIP, L_KN, 0.068, "L"), (L_KN, L_AN, 0.052, "L"), (L_AN, L_HEEL, 0.038, "L"), (L_HEEL, L_FT, 0.036, "L"),
    (L_AN, L_FT, 0.036, "L"),
    (R_HIP, R_KN, 0.068, "R"), (R_KN, R_AN, 0.052, "R"), (R_AN, R_HEEL, 0.038, "R"), (R_HEEL, R_FT, 0.036, "R"),
    (R_AN, R_FT, 0.036, "R"),
    ("NECK", "HEADC", 0.042, "C"),
]
JOINTS = [(L_EL, "L", 0.046), (R_EL, "R", 0.046), (L_KN, "L", 0.062), (R_KN, "R", 0.062),
          (L_WR, "L", 0.038), (R_WR, "R", 0.038), (L_AN, "L", 0.046), (R_AN, "R", 0.046),
          (L_SH, "L", 0.055), (R_SH, "R", 0.055), (L_HIP, "L", 0.07), (R_HIP, "R", 0.07)]
SIDE_COL = {"L": COL_L, "R": COL_R, "C": (215, 210, 205)}


def norm(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def named_points(J):
    """Extra derived points for one frame J (33, 3)."""
    p = {i: J[i] for i in range(33)}
    p["LH"] = (J[L_INDEX] + J[L_PINKY]) / 2
    p["RH"] = (J[R_INDEX] + J[R_PINKY]) / 2
    neck = (J[L_SH] + J[R_SH]) / 2
    ear_mid = (J[L_EAR] + J[R_EAR]) / 2
    hip_mid = (J[L_HIP] + J[R_HIP]) / 2
    up_b = norm(neck - hip_mid)
    fwd_h = J[NOSE] - ear_mid
    fwd_h = norm(fwd_h - up_b * (fwd_h @ up_b) * 0.3)
    head_c = ear_mid + up_b * 0.03
    p["NECK"] = neck + up_b * 0.02
    p["HEADC"] = head_c
    p["_fwd_h"] = fwd_h
    p["_up"] = up_b
    across = J[L_SH] - J[R_SH] + J[L_HIP] - J[R_HIP]
    front = norm(np.cross(across, up_b))
    p["_front"] = front
    return p


def body_front_xz(J):
    p = named_points(J)
    f = p["_front"].copy()
    f[1] = 0
    return norm(f)


# --------------------------------------------------------------------------- drawing

def bg_image(w, h):
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = BG_TOP * (1 - t) + BG_BOT * t
    return np.repeat(img, w, axis=1)


def draw_floor(img, cam, centre, extent=7.0, step=0.5):
    """Perspective grid on y=0 around `centre`, fading with distance."""
    cx = np.round(centre[0] / step) * step
    cz = np.round(centre[2] / step) * step
    vals = np.arange(-extent, extent + 1e-6, step)
    segs = []
    for v in vals:
        major = abs((cx + v) % 1.0) < 1e-6 or abs((cx + v) % 1.0 - 1.0) < 1e-6
        segs.append(([cx + v, 0, cz - extent], [cx + v, 0, cz + extent], major))
        major = abs((cz + v) % 1.0) < 1e-6 or abs((cz + v) % 1.0 - 1.0) < 1e-6
        segs.append(([cx - extent, 0, cz + v], [cx + extent, 0, cz + v], major))
    t = np.linspace(0, 1, 29)[:, None]
    for a, b, major in segs:
        a, b = np.array(a), np.array(b)
        pts = a + (b - a) * t
        uv, z = cam.project(pts)
        d = np.linalg.norm((pts - centre)[:, [0, 2]], axis=1)
        for k in range(len(pts) - 1):
            if z[k] < 0.3 or z[k + 1] < 0.3:
                continue
            alpha = np.clip(1.0 - (d[k] + d[k + 1]) / 2 / extent, 0, 1) ** 1.3 * (0.55 if major else 0.25)
            if alpha < 0.02:
                continue
            p0, p1 = uv[k], uv[k + 1]
            base = img[int(np.clip(p0[1], 0, img.shape[0] - 1)), int(np.clip(p0[0], 0, img.shape[1] - 1))]
            col = base * (1 - alpha) + GRID * alpha
            cv2.line(img, tuple(np.int32(p0 * 16)), tuple(np.int32(p1 * 16)), col.tolist(),
                     max(1, SS), cv2.LINE_AA, shift=4)


def draw_poly_floor(img, cam, pts3, color, alpha):
    uv, z = cam.project(pts3)
    if (z < 0.3).any():
        return
    over = img.copy()
    cv2.fillPoly(over, [np.int32(uv * 16)], color, cv2.LINE_AA, shift=4)
    cv2.addWeighted(over, alpha, img, 1 - alpha, 0, dst=img)


def draw_facing_arrow(img, cam, J):
    hip = (J[L_HIP] + J[R_HIP]) / 2
    f = body_front_xz(J)
    s = np.array([f[2], 0, -f[0]])
    base = np.array([hip[0], 0.003, hip[2]])
    tip = base + f * 0.62
    pts = np.array([base + f * 0.30 + s * 0.13, tip, base + f * 0.30 - s * 0.13, base + f * 0.38])
    draw_poly_floor(img, cam, pts, COL_ACCENT, 0.55)


def draw_shadow(img, cam, J, strength=0.55):
    h, w = img.shape[:2]
    sc = 4
    sh = np.zeros((h // sc, w // sc), np.float32)
    flat = J.copy()
    flat[:, 1] = 0.0
    p = named_points(flat)
    for a, b, r, _ in LIMBS:
        uv, z = cam.project(np.array([p[a], p[b]]))
        if (z < 0.3).any():
            continue
        th = max(1, int(cam.f * r * 2 / z.mean() / sc))
        cv2.line(sh, tuple(np.int32(uv[0] / sc)), tuple(np.int32(uv[1] / sc)), 1.0, th, cv2.LINE_AA)
    quad = np.array([flat[L_SH], flat[R_SH], flat[R_HIP], flat[L_HIP]])
    uv, z = cam.project(quad)
    if (z > 0.3).all():
        cv2.fillPoly(sh, [np.int32(uv / sc)], 1.0, cv2.LINE_AA)
    uv, z = cam.project(p["HEADC"][None])
    if z[0] > 0.3:
        cv2.circle(sh, tuple(np.int32(uv[0] / sc)), max(1, int(cam.f * 0.1 / z[0] / sc)), 1.0, -1, cv2.LINE_AA)
    # height-dependent fade: lifted body parts cast lighter shadows (cheap approximation)
    k = max(3, int(cam.f * 0.06 / sc / max(np.linalg.norm(cam.eye - J.mean(0)), 1)) | 1)
    sh = cv2.GaussianBlur(sh, (0, 0), k)
    sh = cv2.resize(sh, (w, h), interpolation=cv2.INTER_LINEAR)
    img *= (1 - strength * np.clip(sh, 0, 1))[:, :, None]


def shade(col, k):
    return tuple(float(np.clip(c * k, 0, 255)) for c in col)


def capsule(img, a, b, rad_px, col, light=(-0.45, -0.7)):
    a16, b16 = np.int32(a * 16), np.int32(b * 16)
    t = max(1, int(round(rad_px * 2)))
    cv2.line(img, tuple(a16), tuple(b16), COL_OUT, t + 2 * SS, cv2.LINE_AA, shift=4)
    cv2.line(img, tuple(a16), tuple(b16), shade(col, 0.78), t, cv2.LINE_AA, shift=4)
    off = np.array(light) * rad_px * 0.32
    cv2.line(img, tuple(np.int32((a + off) * 16)), tuple(np.int32((b + off) * 16)), col,
             max(1, int(t * 0.55)), cv2.LINE_AA, shift=4)
    off2 = np.array(light) * rad_px * 0.55
    cv2.line(img, tuple(np.int32((a + off2) * 16)), tuple(np.int32((b + off2) * 16)), shade(col, 1.18),
             max(1, int(t * 0.18)), cv2.LINE_AA, shift=4)


def sphere(img, c, rad_px, col, light=(-0.45, -0.7)):
    c16 = np.int32(c * 16)
    r16 = int(rad_px * 16)
    cv2.circle(img, tuple(c16), r16 + 16 * SS, COL_OUT, -1, cv2.LINE_AA, shift=4)
    cv2.circle(img, tuple(c16), r16, shade(col, 0.78), -1, cv2.LINE_AA, shift=4)
    off = np.array(light) * rad_px * 0.22
    cv2.circle(img, tuple(np.int32((c + off) * 16)), int(r16 * 0.72), col, -1, cv2.LINE_AA, shift=4)
    off2 = np.array(light) * rad_px * 0.42
    cv2.circle(img, tuple(np.int32((c + off2) * 16)), int(r16 * 0.32), shade(col, 1.08), -1, cv2.LINE_AA, shift=4)


def draw_body(img, cam, J, grey=False):
    p = named_points(J)
    prims = []  # (depth, kind, data)

    def depth(pt):
        return cam.cam(pt)[2]

    for a, b, r, side in LIMBS:
        pa, pb = p[a], p[b]
        if a == "NECK":
            # the neck always sits behind the head spheres (never painted over the head)
            pb = p["HEADC"] - p["_up"] * 0.05
            prims.append((depth(p["HEADC"]) + 0.08, "cap", (pa, pb, r, SIDE_COL[side])))
            continue
        prims.append(((depth(pa) + depth(pb)) / 2, "cap", (pa, pb, r, SIDE_COL[side])))
    for j, side, r in JOINTS:
        prims.append((depth(p[j]) + 0.01, "sph", (p[j], r, shade(SIDE_COL[side], 0.95))))

    # torso: front lighter than back so the viewer sees which way the chest faces
    quad = np.array([J[L_SH], J[R_SH], J[R_HIP], J[L_HIP]])
    ctr = quad.mean(0)
    vis_front = (p["_front"] @ (cam.eye - ctr)) > 0
    prims.append((depth(ctr) + 0.02, "torso", (quad, vis_front)))

    # head: hair sphere + face sphere, ordered by depth -> head direction is readable
    hc, fwd, up = p["HEADC"], p["_fwd_h"], p["_up"]
    hair_c = hc - fwd * 0.018 + up * 0.014
    face_c = hc + fwd * 0.034 - up * 0.012
    prims.append((depth(hair_c), "sph", (hair_c, 0.112, COL_HAIR)))
    prims.append((depth(face_c), "face", (face_c, 0.088, fwd, up)))

    prims.sort(key=lambda x: -x[0])
    for _, kind, d in prims:
        if kind == "cap":
            pa, pb, r, col = d
            uv, z = cam.project(np.array([pa, pb]))
            if (z < 0.2).any():
                continue
            col = grey_col(col) if grey else col
            capsule(img, uv[0], uv[1], cam.f * r / z.mean(), col)
        elif kind == "sph":
            c, r, col = d
            uv, z = cam.project(c[None])
            if z[0] < 0.2:
                continue
            col = grey_col(col) if grey else col
            sphere(img, uv[0], cam.f * r / z[0], col)
        elif kind == "torso":
            quad, front = d
            # inflate the quad a little so it reads as a chunky torso
            c = quad.mean(0)
            q = c + (quad - c) * 1.12
            uv, z = cam.project(q)
            if (z < 0.2).any():
                continue
            col = COL_TORSO_F if front else COL_TORSO_B
            col = grey_col(col) if grey else col
            pts = np.int32(uv * 16)
            cv2.fillPoly(img, [pts], COL_OUT, cv2.LINE_AA, shift=4)
            cv2.polylines(img, [pts], True, COL_OUT, 4 * SS, cv2.LINE_AA, shift=4)
            inner = c + (quad - c) * 1.04
            uv2, _ = cam.project(inner)
            cv2.fillPoly(img, [np.int32(uv2 * 16)], shade(col, 0.9), cv2.LINE_AA, shift=4)
            if front:  # a stripe down the chest marks the front
                top = (quad[0] + quad[1]) / 2
                bot = (quad[2] + quad[3]) / 2
                s_uv, s_z = cam.project(np.array([top * 0.85 + bot * 0.15, top * 0.25 + bot * 0.75]))
                cv2.line(img, tuple(np.int32(s_uv[0] * 16)), tuple(np.int32(s_uv[1] * 16)),
                         grey_col(COL_ACCENT) if grey else COL_ACCENT,
                         max(1, int(cam.f * 0.03 / s_z.mean())), cv2.LINE_AA, shift=4)
        elif kind == "face":
            c, r, fwd, up = d
            uv, z = cam.project(c[None])
            if z[0] < 0.2:
                continue
            col = grey_col(COL_SKIN) if grey else COL_SKIN
            sphere(img, uv[0], cam.f * r / z[0], col)
            # eyes: only when the face looks toward the camera
            if fwd @ (cam.eye - c) > 0:
                side = norm(np.cross(up, fwd))
                for s in (-1, 1):
                    e = c + fwd * 0.075 + up * 0.02 + side * s * 0.032
                    euv, ez = cam.project(e[None])
                    cv2.circle(img, tuple(np.int32(euv[0] * 16)), int(cam.f * 0.013 / ez[0] * 16),
                               (40, 30, 30), -1, cv2.LINE_AA, shift=4)


def grey_col(col):
    g = sum(col) / 3
    return (g * 0.7, g * 0.7, g * 0.7)


def draw_trails(img, cam, hist_pts, cols):
    """hist_pts: (T, k, 3) recent positions of k joints, oldest first."""
    T = len(hist_pts)
    if T < 2:
        return
    uv, z = cam.project(hist_pts)
    for j in range(hist_pts.shape[1]):
        for t in range(T - 1):
            if z[t, j] < 0.3 or z[t + 1, j] < 0.3:
                continue
            a = (t + 1) / T
            col = tuple(c * (0.35 + 0.65 * a) for c in cols[j])
            th = max(1, int(SS * (1 + 4 * a)))
            cv2.line(img, tuple(np.int32(uv[t, j] * 16)), tuple(np.int32(uv[t + 1, j] * 16)), col, th,
                     cv2.LINE_AA, shift=4)


# --------------------------------------------------------------------------- panels

def render_view(J, cam, w, h, centre, trails=None, grey=False):
    img = bg_image(w, h)
    draw_floor(img, cam, centre)
    draw_shadow(img, cam, J)
    draw_facing_arrow(img, cam, J)
    if trails is not None:
        draw_trails(img, cam, *trails)
    draw_body(img, cam, J, grey=grey)
    out = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.resize(out, (w // SS, h // SS), interpolation=cv2.INTER_AREA)


def minimap(root_hist, front, bounds, size=250, now_grey=False):
    img = np.full((size, size, 3), (40, 31, 26), np.uint8)
    (x0, z0), (x1, z1) = bounds
    pad = 22
    span = max(x1 - x0, z1 - z0)
    sc = (size - 2 * pad) / span
    cxm, czm = (x0 + x1) / 2, (z0 + z1) / 2

    def to_px(x, z):  # top of the map = back of stage, bottom = audience / camera
        return (size / 2 + (x - cxm) * sc, size / 2 + (z - czm) * sc)

    for g in np.arange(np.floor(x0 - span), np.ceil(x1 + span) + 1):
        u, _ = to_px(g, 0)
        if pad <= u <= size - pad:
            cv2.line(img, (int(u), pad), (int(u), size - pad), (70, 58, 50), 1, cv2.LINE_AA)
    for g in np.arange(np.floor(z0 - span), np.ceil(z1 + span) + 1):
        _, v = to_px(0, g)
        if pad <= v <= size - pad:
            cv2.line(img, (pad, int(v)), (size - pad, int(v)), (70, 58, 50), 1, cv2.LINE_AA)
    u, v = to_px(0, 0)
    cv2.drawMarker(img, (int(u), int(v)), (110, 95, 85), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
    T = len(root_hist)
    for t in range(T - 1):
        a = (t + 1) / T
        p0 = to_px(root_hist[t, 0], root_hist[t, 2])
        p1 = to_px(root_hist[t + 1, 0], root_hist[t + 1, 2])
        cv2.line(img, (int(p0[0] * 16), int(p0[1] * 16)), (int(p1[0] * 16), int(p1[1] * 16)),
                 tuple(c * (0.25 + 0.75 * a) for c in COL_ACCENT), 2, cv2.LINE_AA, shift=4)
    cur = root_hist[-1]
    c = np.array(to_px(cur[0], cur[2]))
    f = np.array([front[0], front[2]])
    s = np.array([-f[1], f[0]])
    tri = np.array([c + f * 16, c - f * 6 + s * 8, c - f * 6 - s * 8])
    col = (150, 150, 150) if now_grey else COL_ACCENT
    cv2.fillPoly(img, [np.int32(tri * 16)], col, cv2.LINE_AA, shift=4)
    cv2.circle(img, (int(c[0] * 16), int(c[1] * 16)), 7 * 16, col, -1, cv2.LINE_AA, shift=4)
    cv2.circle(img, (int(c[0] * 16), int(c[1] * 16)), 7 * 16, COL_OUT, 2, cv2.LINE_AA, shift=4)
    return img


# --------------------------------------------------------------------------- static overlay (text)

def static_overlay(title, subtitle):
    ov = Image.new("RGBA", (W_OUT, H_OUT), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.rectangle([0, 0, W_OUT, HEADER], fill=(14, 17, 22, 255))
    d.text((28, 12), title, font=font(34), fill=(255, 255, 255, 255))
    d.text((28 + d.textlength(title, font=font(34)) + 18, 26), subtitle, font=font(20, False),
           fill=(170, 180, 195, 255))
    # panel borders
    for (x, y, w, h) in (FRONT, BACK):
        d.rectangle([x, y, x + w - 1, y + h - 1], outline=(70, 78, 92, 255), width=2)
    d.line([MAIN[2], HEADER, MAIN[2], H_OUT], fill=(70, 78, 92, 255), width=2)

    def label(x, y, text, sub=None):
        f1 = font(24)
        tw = d.textlength(text, font=f1)
        sw = d.textlength(sub, font=font(17, False)) if sub else 0
        bw = max(tw, sw) + 26
        bh = 40 + (24 if sub else 0)
        d.rounded_rectangle([x, y, x + bw, y + bh], 10, fill=(10, 12, 16, 190))
        d.text((x + 13, y + 5), text, font=f1, fill=(255, 255, 255, 255))
        if sub:
            d.text((x + 13, y + 38), sub, font=font(17, False), fill=(185, 195, 210, 255))

    label(MAIN[0] + 18, MAIN[1] + 16, "3Dビュー（斜め前から）")
    label(FRONT[0] + 14, FRONT[1] + 12, "正面（動画と同じ向き）")
    label(BACK[0] + 14, BACK[1] + 12, "背面（自分と同じ向き）", "左右がそのまま自分の手足と一致＝真似しやすい")

    # legend (top right of main panel)
    lx, ly = MAIN[0] + MAIN[2] - 300, MAIN[1] + 16
    d.rounded_rectangle([lx, ly, lx + 282, ly + 150], 12, fill=(10, 12, 16, 190))
    rows = [((63, 169, 245), "左腕・左脚（本人の左）"), ((255, 138, 61), "右腕・右脚（本人の右）"),
            ((233, 236, 240), "胴体（明るい面＝胸側）"), ((232, 199, 102), "床の矢印＝体の向き")]
    for i, (c, t) in enumerate(rows):
        yy = ly + 16 + i * 32
        d.rounded_rectangle([lx + 16, yy + 6, lx + 52, yy + 18], 6, fill=c + (255,))
        d.text((lx + 64, yy - 2), t, font=font(18, False), fill=(235, 238, 245, 255))

    # minimap caption
    mx, my = MAIN[0] + 18, MAIN[1] + MAIN[3] - 250 - 18 - 30
    d.rounded_rectangle([mx, my - 34, mx + 250, my], 8, fill=(10, 12, 16, 190))
    d.text((mx + 10, my - 31), "立ち位置（真上から）", font=font(19), fill=(255, 255, 255, 255))
    d.text((mx + 70, my + 250 + 2), "▼ 客席・カメラ側", font=font(16, False), fill=(185, 195, 210, 255))
    return np.array(ov)


def blend_rgba(dst_bgr, rgba, x=0, y=0):
    h, w = rgba.shape[:2]
    roi = dst_bgr[y:y + h, x:x + w].astype(np.float32)
    a = rgba[:, :, 3:4].astype(np.float32) / 255
    src = rgba[:, :, [2, 1, 0]].astype(np.float32)
    dst_bgr[y:y + h, x:x + w] = (roi * (1 - a) + src * a).astype(np.uint8)


def text_patch(lines, size=(560, 50), fsize=30, align="right"):
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = font(fsize)
    x = size[0] - 8 - d.textlength(lines, font=f) if align == "right" else 8
    d.text((x, (size[1] - fsize) / 2 - 6), lines, font=f, fill=(255, 255, 255, 255))
    return np.array(im)


# --------------------------------------------------------------------------- frame assembly

class Scene:
    def __init__(self, data, args):
        self.J = data["joints"].astype(np.float64)
        self.held = data["held"].astype(bool)
        self.fps = float(data["fps"])
        self.n = len(self.J)
        self.args = args
        root = (self.J[:, L_HIP] + self.J[:, R_HIP]) / 2
        root[:, 1] = 0
        self.root = root
        w = min(self.n - (1 - self.n % 2), int(self.fps * 2.5) | 1)
        self.follow = savgol_filter(root, w, 1, axis=0) if w > 3 else root.copy()
        w2 = min(self.n - (1 - self.n % 2), int(self.fps * 0.6) | 1)
        self.follow_fast = savgol_filter(root, w2, 1, axis=0) if w2 > 3 else root.copy()
        xs, zs = root[:, 0], root[:, 2]
        span = max(np.ptp(xs), np.ptp(zs), 3.0) + 1.0
        cx, cz = (xs.max() + xs.min()) / 2, (zs.max() + zs.min()) / 2
        self.bounds = ((cx - span / 2, cz - span / 2), (cx + span / 2, cz + span / 2))
        self.overlay = static_overlay(args.title, args.subtitle)
        self.duration = self.n / self.fps

    def sample(self, arr, t_frames):
        i0 = int(np.floor(t_frames))
        i0 = min(max(i0, 0), self.n - 1)
        i1 = min(i0 + 1, self.n - 1)
        a = t_frames - np.floor(t_frames)
        return arr[i0] * (1 - a) + arr[i1] * a

    def frame(self, t_frames):
        J = self.sample(self.J, t_frames)
        held = self.held[min(int(round(t_frames)), self.n - 1)]
        out = np.zeros((H_OUT, W_OUT, 3), np.uint8)

        # main 3D view, camera gently follows the dancer
        fc = self.sample(self.follow, t_frames)
        tgt = np.array([fc[0], 0.85, fc[2]])
        az, el, dist = np.radians(self.args.main_azimuth), np.radians(17), 5.4
        eye = tgt + dist * np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
        x, y, w, h = MAIN
        cam = Camera(eye, tgt, w * SS, h * SS, fov_y=36)
        k = int(self.fps * 0.7)
        i_now = int(round(t_frames))
        idx = np.clip(np.arange(i_now - k, i_now + 1), 0, self.n - 1)
        trails = (self.J[idx][:, [L_WR, R_WR, L_AN, R_AN]],
                  [COL_L, COL_R, shade(COL_L, 0.75), shade(COL_R, 0.75)])
        out[y:y + h, x:x + w] = render_view(J, cam, w * SS, h * SS, tgt, trails=trails, grey=held)

        # front & back views centred on the dancer
        fc2 = self.sample(self.follow_fast, t_frames)
        for (px, py, pw, ph), sgn in ((FRONT, 1.0), (BACK, -1.0)):
            tgt2 = np.array([fc2[0], 0.88, fc2[2]])
            eye2 = tgt2 + np.array([0.0, 0.55, 4.6 * sgn])
            cam2 = Camera(eye2, tgt2, pw * SS, ph * SS, fov_y=34)
            out[py:py + ph, px:px + pw] = render_view(J, cam2, pw * SS, ph * SS, tgt2, grey=held)

        # floor position map
        k2 = int(self.fps * 4)
        idx2 = np.clip(np.arange(i_now - k2, i_now + 1), 0, self.n - 1)
        mm = minimap(self.root[idx2], body_front_xz(J), self.bounds, now_grey=held)
        mx, my = MAIN[0] + 18, MAIN[1] + MAIN[3] - 250 - 18 - 30
        out[my:my + 250, mx:mx + 250] = cv2.addWeighted(out[my:my + 250, mx:mx + 250], 0.15, mm, 0.85, 0)

        blend_rgba(out, self.overlay)

        # time + speed
        t_sec = t_frames / self.fps
        txt = f"{int(t_sec // 60):02d}:{t_sec % 60:04.1f} / {int(self.duration // 60):02d}:{self.duration % 60:04.1f}"
        if self.args.speed != 1.0:
            txt = f"{self.args.speed:g}倍速   " + txt
        blend_rgba(out, text_patch(txt), W_OUT - 580, 11)
        if held:
            blend_rgba(out, text_patch("この瞬間は人物が隠れていて推定できません（直前の姿勢を表示）",
                                       size=(1160, 46), fsize=24, align="left"), MAIN[0] + 20, MAIN[1] + 80)
        # progress bar
        by = H_OUT - 10
        cv2.rectangle(out, (0, by), (MAIN[2], H_OUT), (40, 34, 30), -1)
        cv2.rectangle(out, (0, by), (int(MAIN[2] * min(t_sec / self.duration, 1)), H_OUT), COL_ACCENT, -1)
        return out


_SCENE = None


def _init(path, args):
    global _SCENE
    _SCENE = Scene(np.load(path), args)


def _render_chunk(job):
    k, times, out_path, fps_out = job
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W_OUT}x{H_OUT}",
           "-r", f"{fps_out}", "-i", "-", "-c:v", "libx264", "-preset", _SCENE.args.preset,
           "-crf", str(_SCENE.args.crf), "-tune", "animation",
           "-pix_fmt", "yuv420p", out_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i, t in enumerate(times):
        p.stdin.write(_SCENE.frame(t).tobytes())
        if k == 0 and i % 60 == 0:
            print(f"[render] worker0 {i}/{len(times)}", flush=True)
    p.stdin.close()
    p.wait()
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("motion3d")
    ap.add_argument("-o", "--out", default="dance3d.mp4")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (0.5 = slow motion)")
    ap.add_argument("--fps", type=float, default=0, help="output fps (default: source fps)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)
    ap.add_argument("--title", default="金髪メンバーの動き ― 3D再現")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--main-azimuth", type=float, default=32.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--audio", default="", help="video/audio file whose sound is added (speed 1.0 only)")
    ap.add_argument("--still", type=float, default=-1, help="render a single PNG at this time (s) and exit")
    ap.add_argument("--crf", type=int, default=23, help="x264 quality (lower = better, bigger)")
    ap.add_argument("--preset", default="medium", help="x264 preset")
    args = ap.parse_args()

    data = np.load(args.motion3d)
    fps_src = float(data["fps"])
    n = len(data["joints"])
    fps_out = args.fps or fps_src

    if args.still >= 0:
        sc = Scene(data, args)
        cv2.imwrite(args.out, sc.frame(min(args.still * fps_src, n - 1)))
        print(f"wrote {args.out}")
        return

    t0 = args.start * fps_src
    t1 = (args.end * fps_src) if args.end > 0 else n - 1
    step = fps_src / fps_out * args.speed
    times = np.arange(t0, t1, step)
    tmp = tempfile.mkdtemp(prefix="render_")
    k = max(1, args.workers)
    chunks = np.array_split(times, k)
    jobs = [(i, c, os.path.join(tmp, f"part{i:02d}.mp4"), fps_out) for i, c in enumerate(chunks) if len(c)]
    print(f"[render] {len(times)} frames, {len(jobs)} workers")
    with Pool(len(jobs), initializer=_init, initargs=(args.motion3d, args)) as pool:
        parts = pool.map(_render_chunk, jobs)
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{p}'\n" for p in parts)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst]
    if args.audio and args.speed == 1.0:
        cmd += ["-ss", f"{args.start}", "-i", args.audio, "-map", "0:v", "-map", "1:a", "-c:a", "aac",
                "-b:a", "160k", "-shortest"]
    cmd += ["-c:v", "copy", "-movflags", "+faststart", args.out]
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[render] wrote {args.out}")


if __name__ == "__main__":
    main()
