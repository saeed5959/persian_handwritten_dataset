"""
OCR baseline for word spotting: decode to text, then string match.
Same metrics as eval_spotting.py, for comparison.
python eval_ocr.py
"""
import os, glob, random, unicodedata
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from kraken.lib import models
from kraken.lib.xml import XMLPage
from kraken.lib.segmentation import extract_polygons
from kraken.lib.dataset import ImageInputTransforms

SRC       = './dataset_test/img_xml'
MODEL     = "./checkpoints/arabic_ma.mlmodel"#'/home/saeed/.local/share/htrmopo/230a3928-733e-5524-baa5-f89ba9b9eb70/all_arabic_scripts.mlmodel'
DEVICE    = 'cuda'
N_QUERIES = 10
MIN_LEN   = 3
SEED      = 0


def normalize(t):
    t = unicodedata.normalize('NFC', t)
    for a, b in [('ي','ی'), ('ك','ک'), ('أ','ا'), ('إ','ا'), ('آ','آ'), ('ة','ه'), ('\u200c',' '), (',', ''), ('.', ''), (':', ''), ('!', ''),('?',''),
                 ('ٔ', ""), ('،', ""), ('؟', ""), ('ـ', ""), ('ؤ', ""), ('ٰ', ""), ('—', ""), ('؛', ""), ('ئ', "ی"), ('ء', ""), ('ۀ', "")]:
        t = t.replace(a, b)
    t = ''.join(c for c in t if not ('\u064B' <= c <= '\u0652'))
    return ' '.join(t.split())


# ------------------------------------------------------------------ model
model = models.load_any(MODEL, device=DEVICE)
model.nn.nn.eval()
codec = model.codec
_, CH, H, W = model.nn.input
tf = ImageInputTransforms(batch=1, height=H, width=W, channels=CH,
                          pad=0, valid_norm=False)

l2c = {}
for ch, labels in codec.c2l.items():
    for l in labels:
        l2c[l] = ch


@torch.no_grad()
def decode(line_img):
    """Greedy CTC decode -> text string."""
    t = tf(line_img).unsqueeze(0)
    t = F.pad(t, (16, 16), value=float(t.max()))
    o, _ = model.nn.nn(t.to(DEVICE))
    best = o[0].squeeze(1).argmax(0).cpu().numpy()      # (T,)
    seq, prev = [], -1
    for c in best:
        if c != prev and c != 0:
            seq.append(l2c.get(int(c), ''))
        prev = c
    return ''.join(seq)


# ------------------------------------------------------------------ index
print('decoding...')
PAGES, GTS, OCRS = [], [], []
for xml in sorted(glob.glob(os.path.join(SRC, '*.xml'))):
    doc = XMLPage(xml)
    seg = doc.to_container()
    im = Image.open(doc.imagename).convert('L')
    page = os.path.basename(xml)
    for line_img, rec in extract_polygons(im, seg):
        gt = normalize(rec.text or '')
        if not gt:
            continue
        try:
            txt = decode(line_img)
        except Exception as e:
            print(' skip line:', e); continue
        PAGES.append(page); GTS.append(gt); OCRS.append(txt)
    print(' ', page, len(GTS))

N = len(GTS)
print(f'{N} lines')


# ------------------------------------------- direction calibration
def overlap(rev):
    hit = 0
    for gt, tx in list(zip(GTS, OCRS))[:60]:
        t = normalize(tx[::-1] if rev else tx)
        hit += sum(1 for w in gt.split() if w in t)
    return hit

REVERSE = overlap(True) > overlap(False)
print(f'logical={overlap(False)}  reversed={overlap(True)}  -> REVERSE={REVERSE}')
OCRS = [normalize(t[::-1] if REVERSE else t) for t in OCRS]

# CER for reference
try:
    import editdistance
    cer = sum(editdistance.eval(g, o) for g, o in zip(GTS, OCRS)) / \
          max(1, sum(len(g) for g in GTS))
    print(f'CER = {cer:.3f}  (acc {1-cer:.3f})')
except ImportError:
    editdistance = None
    print('pip install editdistance for CER + fuzzy matching')


# ------------------------------------------------------------------ queries
random.seed(SEED)
by_page = {}
for p, g in zip(PAGES, GTS):
    by_page.setdefault(p, []).extend(w for w in g.split() if len(w) >= MIN_LEN)

queries = set()
for p, ws in by_page.items():
    u = sorted(set(ws))
    if u:
        queries.update(random.sample(u, min(N_QUERIES, len(u))))
queries = sorted(queries)
print(f'{len(queries)} queries')


# ------------------------------------------------------------------ scorers
def score_exact(q, txt):
    """1.0 if the query is a whole token in the OCR text, else 0."""
    return 1.0 if q in txt.split() else 0.0


def score_substr(q, txt):
    """1.0 if the query appears anywhere in the OCR text."""
    return 1.0 if q in txt else 0.0


def score_fuzzy(q, txt):
    """1 - min normalised edit distance to any token (continuous)."""
    if editdistance is None:
        return score_exact(q, txt)
    ws = txt.split()
    if not ws:
        return 0.0
    return 1.0 - min(editdistance.eval(q, w) / max(len(q), len(w)) for w in ws)


# ------------------------------------------------------------------ report
def report(name, scores_list, rels_list):
    sc = np.concatenate(scores_list)
    rl = np.concatenate(rels_list)
    print(f'\n--- {name}: {rl.sum()} relevant / {len(rl)} pairs')

    best = (0, 0, 0, 0)
    thrs = np.unique(sc) if len(np.unique(sc)) < 500 else \
           np.quantile(sc, np.linspace(0.5, 0.9999, 400))
    for thr in thrs:
        pred = sc >= thr
        tp = int((pred & rl).sum()); fp = int((pred & ~rl).sum())
        fn = int((~pred & rl).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        if f > best[2]:
            best = (p, r, f, thr)
    print(f'best F1 @ thr={best[3]:.3f}  P={best[0]:.3f} '
          f'R={best[1]:.3f} F1={best[2]:.3f}')

    aps = []
    for s, r in zip(scores_list, rels_list):
        o = np.argsort(-s, kind='stable'); r = r[o]
        if not r.any():
            continue
        hits = np.cumsum(r)
        aps.append(float((hits[r] / (np.where(r)[0] + 1)).mean()))
    print(f'mAP = {np.mean(aps):.3f} over {len(aps)} queries')


upages = sorted(set(PAGES))
pidx = {p: i for i, p in enumerate(upages)}
pa = np.array([pidx[p] for p in PAGES])

for label, fn in (('exact token', score_exact),
                  ('substring', score_substr),
                  ('fuzzy (1-NED)', score_fuzzy)):
    SC, RL = [], []
    for q in queries:
        SC.append(np.array([fn(q, t) for t in OCRS]))
        RL.append(np.array([q in g.split() for g in GTS]))

    print(f'\n================ OCR baseline: {label} ================')
    report('line level', SC, RL)

    PSC, PRL = [], []
    for s, r in zip(SC, RL):
        ps = np.full(len(upages), -np.inf); pr = np.zeros(len(upages), bool)
        np.maximum.at(ps, pa, s)
        np.logical_or.at(pr, pa, r)
        PSC.append(ps); PRL.append(pr)
    report('page level', PSC, PRL)
