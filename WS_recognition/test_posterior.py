import numpy as np, torch
from PIL import Image
from kraken.lib import models
from kraken.lib.dataset import ImageInputTransforms

model = models.load_any(
    '/home/saeed/.local/share/htrmopo/230a3928-733e-5524-baa5-f89ba9b9eb70/all_arabic_scripts.mlmodel',
    device='cuda')

_, ch, h, w = model.nn.input          # (1, 1, 120, 0)
tf = ImageInputTransforms(batch=1, height=h, width=w, channels=ch,
                          pad=0, valid_norm=False)

img = Image.open('./dataset/lines/0001_NLAI-5-11187-0000_000.png')
t = tf(img).unsqueeze(0)
assert t.shape[2] == h, t.shape      # must be 120

t = torch.nn.functional.pad(t, (16, 16), value=t.max().item())   # width only
t = t.to('cuda')

model.nn.nn.eval()
with torch.no_grad():
    o, olens = model.nn.nn(t)

logits = np.squeeze(o.cpu().numpy()[0])          # (197, T)
p = np.exp(logits - logits.max(0, keepdims=True))
p /= p.sum(0, keepdims=True)
post = p.T                                        # (T, 197)
print(post.shape)

best = post.argmax(1)
seq, prev = [], -1
for c in best:
    if c != prev and c != 0:
        seq.append(int(c))
    prev = c
print(model.codec.decode([(c, 0, 0, 0) for c in seq]))



import numpy as np

# class index -> character
l2c = {}
for m in list(model.codec.c2l.items()):
    l2c[m[0]] = m[1]
l2c = {}
for char, labels in model.codec.c2l.items():
    for l in labels:
        l2c[l] = char
l2c[0] = '␀'   # blank

# top-k per frame
for t in range(post.shape[0]):
    top = np.argsort(post[t])[::-1][:4]
    print(f"{t:3d}  " + "  ".join(f"{l2c.get(int(c),'?')}={post[t,c]:.3f}" for c in top))