"""
Word search over pages via CTC posteriors.
Build index once, then query interactively.

  python search.py                    # interactive
  python search.py کتاب               # single query
"""
import os, sys, glob, pickle, unicodedata
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from kraken.lib import models
from kraken.lib.xml import XMLPage
from kraken.lib.segmentation import extract_polygons
from kraken.lib.dataset import ImageInputTransforms

SRC    = './dataset_test/img_xml'
MODEL  = "./checkpoints/arabic_ma.mlmodel"#'/home/saeed/.local/share/htrmopo/230a3928-733e-5524-baa5-f89ba9b9eb70/all_arabic_scripts.mlmodel'
CACHE_USE = False
CACHE  = './index.pkl'
DEVICE = 'cuda'
TOPK   = 15
NEG    = -1e30


def normalize(t):
    t = unicodedata.normalize('NFC', t)
    for a, b in  [('ي','ی'), ('ك','ک'), ('أ','ا'), ('إ','ا'), ('آ','آ'), ('ة','ه'), ('\u200c',' '), (',', ''), ('.', ''), (':', ''), ('!', ''),('?',''),
                 ('ٔ', ""), ('،', ""), ('؟', ""), ('ـ', ""), ('ؤ', ""), ('ٰ', ""), ('—', ""), ('؛', ""), ('ئ', "ی"), ('ء', ""), ('ۀ', "")]:
        t = t.replace(a, b)
    t = ''.join(c for c in t if not ('\u064B' <= c <= '\u0652'))
    return ' '.join(t.split())


model = models.load_any(MODEL, device=DEVICE)
model.nn.nn.eval()
codec = model.codec
_, CH, H, W = model.nn.input
tf = ImageInputTransforms(batch=1, height=H, width=W, channels=CH,
                          pad=0, valid_norm=False)
PAD = 16
DOWNSAMPLE = 8            # three Mp2,2 layers


@torch.no_grad()
def posteriors(line_img):
    t = tf(line_img).unsqueeze(0)
    w_resized = t.shape[3]
    t = F.pad(t, (PAD, PAD), value=float(t.max()))
    o, _ = model.nn.nn(t.to(DEVICE))
    return F.log_softmax(o[0].squeeze(1), dim=0).T, w_resized


# ------------------------------------------------------------------ index
def build():
    recs = []
    for xml in sorted(glob.glob(os.path.join(SRC, '*.xml'))):
        doc = XMLPage(xml)
        seg = doc.to_container()
        im = Image.open(doc.imagename).convert('L')
        for line_img, r in extract_polygons(im, seg):
            try:
                lp, wres = posteriors(line_img)
            except Exception as e:
                print(' skip:', e); continue
            xs = [p[0] for p in r.boundary]; ys = [p[1] for p in r.boundary]
            recs.append(dict(
                page=os.path.basename(xml),
                image=doc.imagename,
                line_id=r.id,
                box=(min(xs), min(ys), max(xs), max(ys)),
                gt=normalize(r.text or ''),
                lp=lp.cpu().numpy().astype(np.float16),
                wres=wres,
            ))
        print(' indexed', os.path.basename(xml), len(recs))
    with open(CACHE, 'wb') as f:
        pickle.dump(recs, f)
    return recs


if CACHE_USE and os.path.exists(CACHE):
    recs = pickle.load(open(CACHE, 'rb'))
    print(f'loaded index: {len(recs)} lines')
else:
    print('building index...')
    recs = build()

N = len(recs)
Tmax = max(r['lp'].shape[0] for r in recs)
C = recs[0]['lp'].shape[1]
LP = torch.full((N, Tmax, C), NEG, device=DEVICE)
LENS = torch.zeros(N, dtype=torch.long, device=DEVICE)
for i, r in enumerate(recs):
    t = r['lp'].shape[0]
    LP[i, :t] = torch.from_numpy(r['lp'].astype(np.float32)).to(DEVICE)
    LENS[i] = t


def encode(q):
    ids = []
    for c in q:
        if c not in codec.c2l:
            return None, c
        ids.extend(codec.c2l[c])
    return ids, None


# ------------------------------------------- local-alignment CTC + backtrace
def spot(ids):
    """-> (scores (N,), end_frame (N,))"""
    S = len(ids); L = 2 * S + 1
    ext = [0] * L
    for i, c in enumerate(ids):
        ext[2 * i + 1] = c
    idx = torch.tensor(ext, device=DEVICE).unsqueeze(0).expand(N, L)
    skip = torch.zeros(L, dtype=torch.bool, device=DEVICE)
    for j in range(2, L):
        if ext[j] != 0 and ext[j] != ext[j - 2]:
            skip[j] = True

    a = torch.full((N, L), NEG, device=DEVICE); a[:, 0] = 0.0
    best = torch.full((N,), NEG, device=DEVICE)
    bend = torch.zeros(N, dtype=torch.long, device=DEVICE)

    for t in range(Tmax):
        emit = LP[:, t, :].gather(1, idx)
        s = torch.logaddexp(a, F.pad(a, (1, -1), value=NEG))
        s = torch.logaddexp(s, F.pad(a, (2, -2), value=NEG).masked_fill(~skip, NEG))
        a = s + emit
        a[:, 0] = 0.0
        end = torch.logaddexp(a[:, L - 1], a[:, L - 2])
        upd = (t < LENS) & (end > best)
        bend = torch.where(upd, t, bend)
        best = torch.where(upd, end, best)

    return (best / S).cpu().numpy(), bend.cpu().numpy()


# ------------------------------------------------------------------ search
REVERSE = True          # Arabic: frames run visual order

def search(word, topk=TOPK):
    q = normalize(word)
    if not q:
        print('empty query'); return
    ids, bad = encode(q[::-1] if REVERSE else q)
    if ids is None:
        print(f'char not in model charset: {bad!r}'); return

    scores, ends = spot(ids)
    order = np.argsort(-scores)[:topk]

    # page ranking
    pg = {}
    for i, s in enumerate(scores):
        p = recs[i]['page']
        if s > pg.get(p, (-1e9,))[0]:
            pg[p] = (s, i)
    top_pages = sorted(pg.items(), key=lambda kv: -kv[1][0])[:8]

    print(f'\n=== "{q}" ===')
    print('\ntop pages:')
    for p, (s, i) in top_pages:
        mark = '*' if q in recs[i]['gt'].split() else ' '
        print(f' {mark} {s:+.3f}  {p}')

    print('\ntop lines:')
    for i in order:
        r = recs[i]
        # frame -> x in original image
        x0, y0, x1, y1 = r['box']
        frac = max(0.0, (ends[i] * DOWNSAMPLE - PAD)) / max(1, r['wres'])
        xh = x0 + frac * (x1 - x0)
        mark = '*' if q in r['gt'].split() else ' '
        print(f' {mark} {scores[i]:+.3f}  {r["page"]}  {r["line_id"]}  '
              f'x≈{xh:.0f} y≈{y0}-{y1}')
        print(f'       {r["gt"][:70]}')


if len(sys.argv) > 1:
    search(' '.join(sys.argv[1:]))
else:
    while True:
        try:
            w = input('\nquery> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not w:
            break
        search(w)
