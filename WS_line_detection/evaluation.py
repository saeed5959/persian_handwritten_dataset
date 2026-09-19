#!/usr/bin/env python3
"""
Segmentation eval: baselines, line polygons, region polygons.
  python eval_seg.py checkpoints/best_0.4655.safetensors dataset_test/img_xml
"""
import argparse, glob, os, subprocess, sys, tempfile, shutil
import numpy as np, cv2
from lxml import etree


# ------------------------------------------------------------------ parsing
def local(el):
    return etree.QName(el).localname


def parse_pts(s):
    return np.array([[float(v) for v in p.split(',')] for p in s.split()], float)


def read_page(path):
    """-> dict(baselines=[polyline], lines=[poly], regions=[poly])"""
    root = etree.parse(path).getroot()
    out = {'baselines': [], 'lines': [], 'regions': []}
    for el in root.iter():
        n = local(el)
        if n == 'Baseline' and el.get('points'):
            out['baselines'].append(parse_pts(el.get('points')))
        elif n == 'TextLine':
            for c in el:
                if local(c) == 'Coords' and c.get('points'):
                    out['lines'].append(parse_pts(c.get('points')))
        elif n == 'TextRegion':
            for c in el:
                if local(c) == 'Coords' and c.get('points'):
                    out['regions'].append(parse_pts(c.get('points')))
    # ALTO fallback: BASELINE attribute on TextLine
    if not out['baselines']:
        for el in root.iter():
            b = el.get('BASELINE') if local(el) == 'TextLine' else None
            if not b:
                continue
            v = b.replace(',', ' ').split()
            try:
                f = [float(x) for x in v]
            except ValueError:
                continue
            if len(f) >= 4:
                out['baselines'].append(np.array(f, float).reshape(-1, 2))
    return out


# ------------------------------------------------------------------ geometry
def poly_iou(a, b):
    ax0, ay0 = a.min(0); ax1, ay1 = a.max(0)
    bx0, by0 = b.min(0); bx1, by1 = b.max(0)
    if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
        return 0.0
    x0, y0 = int(min(ax0, bx0)), int(min(ay0, by0))
    w = int(max(ax1, bx1)) - x0 + 2
    h = int(max(ay1, by1)) - y0 + 2
    if w <= 0 or h <= 0 or w * h > 40_000_000:
        return 0.0
    off = np.array([x0, y0])
    ma = np.zeros((h, w), np.uint8); mb = np.zeros((h, w), np.uint8)
    cv2.fillPoly(ma, [np.round(a - off).astype(np.int32)], 1)
    cv2.fillPoly(mb, [np.round(b - off).astype(np.int32)], 1)
    inter = int((ma & mb).sum()); union = int((ma | mb).sum())
    return inter / union if union else 0.0


def resample(bl, n=20):
    bl = np.asarray(bl, float)
    if len(bl) < 2:
        return np.repeat(bl[:1], n, 0)
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(bl, 0 + 0), axis=1))]) \
        if False else np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(bl, axis=0), axis=1))])
    if d[-1] <= 0:
        return np.repeat(bl[:1], n, 0)
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, bl[:, 0]), np.interp(t, d, bl[:, 1])], 1)


def bl_dist(a, b):
    """Symmetric mean point-to-polyline distance."""
    def one(P, L):
        p0, p1 = L[:-1], L[1:]
        ab = p1 - p0
        L2 = np.clip((ab ** 2).sum(1), 1e-9, None)
        o = np.empty(len(P))
        for i, p in enumerate(P):
            t = np.clip(((p - p0) * ab).sum(1) / L2, 0, 1)
            o[i] = np.linalg.norm((p0 + t[:, None] * ab) - p, axis=1).min()
        return o.mean()
    A, B = resample(a), resample(b)
    if len(a) < 2 or len(b) < 2:
        return 1e9
    return float((one(A, np.asarray(b, float)) + one(B, np.asarray(a, float))) / 2)


def greedy(cost, thr, higher_better):
    """-> (n_matched, [scores])"""
    if cost.size == 0:
        return 0, []
    order = np.argsort(-cost, axis=None) if higher_better else np.argsort(cost, axis=None)
    up, ug, sc = set(), set(), []
    for k in order:
        i, j = np.unravel_index(k, cost.shape)
        v = cost[i, j]
        if (higher_better and v < thr) or (not higher_better and v > thr):
            break
        if i in up or j in ug:
            continue
        up.add(int(i)); ug.add(int(j)); sc.append(float(v))
    return len(up), sc


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


# ------------------------------------------------------------------ main
ap = argparse.ArgumentParser()
ap.add_argument('--model')
ap.add_argument('--data_dir')
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--iou-thrs', default='0.3')
ap.add_argument('--bl-tols', default='5,10,20')
ap.add_argument('--keep')
a = ap.parse_args()

IOUS = [float(x) for x in a.iou_thrs.split(',')]
TOLS = [float(x) for x in a.bl_tols.split(',')]
tmp = a.keep or tempfile.mkdtemp(prefix='seg_')
os.makedirs(tmp, exist_ok=True)

files = sorted(glob.glob(os.path.join(a.data_dir, '*.xml')))
if not files:
    sys.exit('no xml found')

agg = {('lines', t): [0, 0, 0, []] for t in IOUS}
agg.update({('regions', t): [0, 0, 0, []] for t in IOUS})
agg.update({('baselines', t): [0, 0, 0, []] for t in TOLS})
npages = 0

for f in files:
    stem = os.path.splitext(os.path.basename(f))[0]
    img = None
    for e in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
        p = os.path.join(a.data_dir, stem + e)
        if os.path.exists(p):
            img = p; break
    if img is None:                       # fall back to imageFilename
        fn = etree.parse(f).getroot().find('.//{*}Page')
        if fn is not None:
            p = os.path.join(a.data_dir, fn.get('imageFilename') or '')
            img = p if os.path.exists(p) else None
    if img is None:
        print('skip (no image):', stem); continue

    out = os.path.join(tmp, stem + '.xml')
    r = subprocess.run(['kraken', '-x', '-d', a.device, '-i', img, out,
                        'segment', '-bl', '-i', a.model],
                       capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        print('skip (kraken failed):', stem, r.stderr.strip()[-200:]); continue

    gt, pr = read_page(f), read_page(out)
    npages += 1

    for key, thrs, hb in (('lines', IOUS, True), ('regions', IOUS, True),
                          ('baselines', TOLS, False)):
        G, P = gt[key], pr[key]
        if hb:
            C = np.array([[poly_iou(p, g) for g in G] for p in P]) if P and G \
                else np.zeros((len(P), len(G)))
        else:
            C = np.array([[bl_dist(p, g) for g in G] for p in P]) if P and G \
                else np.full((len(P), len(G)), 1e9)
        for t in thrs:
            m, sc = greedy(C, t, hb)
            agg[(key, t)][0] += m
            agg[(key, t)][1] += len(P) - m
            agg[(key, t)][2] += len(G) - m
            agg[(key, t)][3] += sc

    print(f'{stem:42s} bl {len(pr["baselines"]):3d}/{len(gt["baselines"]):3d}  '
          f'ln {len(pr["lines"]):3d}/{len(gt["lines"]):3d}  '
          f'rg {len(pr["regions"]):2d}/{len(gt["regions"]):2d}')

print(f'\n{npages}/{len(files)} pages\n')
for key, thrs, unit in (('baselines', TOLS, 'px'), ('lines', IOUS, 'IoU'),
                        ('regions', IOUS, 'IoU')):
    print(key)
    for t in thrs:
        tp, fp, fn, sc = agg[(key, t)]
        p, r, f1 = prf(tp, fp, fn)
        extra = f'  mean {np.mean(sc):.3f}' if sc else ''
        print(f'  @{t:<5g}{unit:4s} P {p:.3f}  R {r:.3f}  F1 {f1:.3f}   '
              f'(TP {tp} FP {fp} FN {fn}){extra}')
    print()

if not a.keep:
    shutil.rmtree(tmp, ignore_errors=True)
else:
    print('predicted XML in', tmp)
