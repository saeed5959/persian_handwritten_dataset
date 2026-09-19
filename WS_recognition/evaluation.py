"""
Word-spotting evaluation via CTC posteriors (local alignment).
python eval_spotting.py
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
NEG       = -1e30


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


@torch.no_grad()
def posteriors(line_img):
    t = tf(line_img).unsqueeze(0)
    t = F.pad(t, (16, 16), value=float(t.max()))
    o, _ = model.nn.nn(t.to(DEVICE))
    return F.log_softmax(o[0].squeeze(1), dim=0).T      # (T, C)


def encode(q):
    ids = []
    for c in q:
        if c not in codec.c2l:
            return None
        ids.extend(codec.c2l[c])
    return ids


# ------------------------------------------------------------------ index
print('indexing...')
lines = []                                   # (page, lp(T,C), gt)
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
            lines.append((page, posteriors(line_img).half(), gt))
        except Exception as e:
            print(' skip line:', e)
    print(' ', page, len(lines))

N = len(lines)
Tmax = max(l[1].shape[0] for l in lines)
C = lines[0][1].shape[1]
LP = torch.full((N, Tmax, C), NEG, device=DEVICE, dtype=torch.float32)
LENS = torch.zeros(N, dtype=torch.long, device=DEVICE)
for i, (_, lp, _) in enumerate(lines):
    LP[i, :lp.shape[0]] = lp.float()
    LENS[i] = lp.shape[0]
PAGES = [l[0] for l in lines]
GTS   = [l[2] for l in lines]
print(f'{N} lines, Tmax={Tmax}, C={C}')


# ------------------------------------------------------- local-alignment CTC
def spot(ids):
    """Best per-char log P(query) anywhere in each line -> (N,) numpy."""
    S = len(ids)
    L = 2 * S + 1
    ext = [0] * L
    for i, c in enumerate(ids):
        ext[2 * i + 1] = c
    ext_t = torch.tensor(ext, device=DEVICE)

    skip = torch.zeros(L, dtype=torch.bool, device=DEVICE)
    for j in range(2, L):
        if ext[j] != 0 and ext[j] != ext[j - 2]:
            skip[j] = True

    a = torch.full((N, L), NEG, device=DEVICE)
    a[:, 0] = 0.0
    best = torch.full((N,), NEG, device=DEVICE)
    idx = ext_t.unsqueeze(0).expand(N, L)

    for t in range(Tmax):
        emit = LP[:, t, :].gather(1, idx)
        s = torch.logaddexp(a, F.pad(a, (1, -1), value=NEG))
        s = torch.logaddexp(s, F.pad(a, (2, -2), value=NEG)
                               .masked_fill(~skip, NEG))
        a = s + emit
        a[:, 0] = 0.0                                  # free start
        end = torch.logaddexp(a[:, L - 1], a[:, L - 2])
        best = torch.where(t < LENS, torch.maximum(best, end), best)

    return (best / S).cpu().numpy()


# ------------------------------------------------- direction calibration
probe = [(i, w) for i in range(min(40, N)) for w in GTS[i].split()
         if len(w) >= MIN_LEN][:20]
fwd = rev = 0.0
for i, w in probe:
    a_, b_ = encode(w), encode(w[::-1])
    if a_ is None or b_ is None:
        continue
    fwd += spot(a_)[i]
    rev += spot(b_)[i]
REVERSE = rev > fwd
print(f'logical={fwd/len(probe):.3f}  reversed={rev/len(probe):.3f}  '
      f'-> REVERSE={REVERSE}')


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


# ------------------------------------------------------------------ run
Q, SC, RL = [], [], []
for qi, q in enumerate(queries):
    ids = encode(q[::-1] if REVERSE else q)
    if ids is None:
        continue
    sc = spot(ids)
    rel = np.array([q in g.split() for g in GTS])
    Q.append(q); SC.append(sc); RL.append(rel)
    if qi % 25 == 0:
        print(f' {qi}/{len(queries)}')


def report(name, scores_list, rels_list):
    sc = np.concatenate(scores_list)
    rl = np.concatenate(rels_list)
    print(f'\n--- {name}: {rl.sum()} relevant / {len(rl)} pairs')

    best = (0, 0, 0, 0)
    for thr in np.quantile(sc, np.linspace(0.5, 0.9999, 400)):
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
        o = np.argsort(-s); r = r[o]
        if not r.any():
            continue
        hits = np.cumsum(r)
        aps.append(float((hits[r] / (np.where(r)[0] + 1)).mean()))
    print(f'mAP = {np.mean(aps):.3f} over {len(aps)} queries')


report('line level', SC, RL)

# page level: max score per page, relevant if word appears anywhere on page
upages = sorted(set(PAGES))
pidx = {p: i for i, p in enumerate(upages)}
pa = np.array([pidx[p] for p in PAGES])
PSC, PRL = [], []
for s, r in zip(SC, RL):
    ps = np.full(len(upages), -np.inf); pr = np.zeros(len(upages), bool)
    np.maximum.at(ps, pa, s)
    np.logical_or.at(pr, pa, r)
    PSC.append(ps); PRL.append(pr)
report('page level', PSC, PRL)
