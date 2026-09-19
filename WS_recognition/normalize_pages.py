import os, glob, shutil, unicodedata
from xml.etree import ElementTree as ET

NS = 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'
ET.register_namespace('', NS)
SRC, OUT = './dataset_test/img_xml', './dataset_test/pages'
os.makedirs(OUT, exist_ok=True)

def normalize(t):
    t = unicodedata.normalize('NFC', t)
    for a, b in [('ي','ی'), ('ك','ک'), ('أ','ا'), ('إ','ا'), ('آ','آ'), ('ة','ه'),
                 ('\u200c',' '), (',',''), ('.',''), (':',''), ('!',''), ('?',''),
                 ('ٔ',''), ('،',''), ('؟',''), ('ـ',''), ('ؤ',''), ('ٰ',''),
                 ('—',''), ('؛',''), ('ئ','ی'), ('ء',''), ('ۀ','')]:
        t = t.replace(a, b)
    t = ''.join(c for c in t if not ('\u064B' <= c <= '\u0652'))
    return ' '.join(t.split())

n_pages = n_lines = 0
for xml in sorted(glob.glob(os.path.join(SRC, '*.xml'))):
    tree = ET.parse(xml)
    root = tree.getroot()
    page = root.find(f'{{{NS}}}Page')

    src_img = os.path.join(SRC, page.get('imageFilename'))
    if not os.path.exists(src_img):
        print('missing image:', src_img); continue

    # image must sit next to the xml for -f page
    dst_img = os.path.join(OUT, os.path.basename(src_img))
    if not os.path.exists(dst_img):
        shutil.copy2(src_img, dst_img)

    for line in root.iter(f'{{{NS}}}TextLine'):
        uni = line.find(f'./{{{NS}}}TextEquiv/{{{NS}}}Unicode')
        if uni is None or not uni.text:
            continue
        uni.text = normalize(uni.text)
        n_lines += 1

    # region-level TextEquiv: rebuild from its lines so it stays consistent
    for region in root.iter(f'{{{NS}}}TextRegion'):
        te = region.find(f'./{{{NS}}}TextEquiv/{{{NS}}}Unicode')
        if te is None:
            continue
        parts = [u.text for u in
                 (l.find(f'./{{{NS}}}TextEquiv/{{{NS}}}Unicode')
                  for l in region.iter(f'{{{NS}}}TextLine'))
                 if u is not None and u.text]
        te.text = '\n'.join(parts)

    tree.write(os.path.join(OUT, os.path.basename(xml)),
               encoding='UTF-8', xml_declaration=True)
    n_pages += 1

print(n_pages, 'pages |', n_lines, 'lines')