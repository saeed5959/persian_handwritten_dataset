#!/usr/bin/env python3
"""Render estimated baselines onto the page images so you can eyeball them
before wasting GPU time.  Run this on every page; fix the bad ones in eScriptorium.

  python qc_overlay.py --xml-dir page_bl/ --img-dir images/ --out-dir qc/
"""
import argparse, glob, os
import numpy as np, cv2
from lxml import etree

PC = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
NS = {"pc": PC}


def pts(s):
    return np.array([[int(float(v)) for v in p.split(",")] for p in s.split()], np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml-dir", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    for f in sorted(glob.glob(os.path.join(a.xml_dir, "*.xml"))):
        page = etree.parse(f).getroot().find("pc:Page", NS)
        img = cv2.imread(os.path.join(a.img_dir, page.get("imageFilename")))
        if img is None:
            print(f"skip {f}: no image"); continue

        for line in page.findall(".//pc:TextLine", NS):
            c = line.find("pc:Coords", NS)
            if c is not None:
                cv2.polylines(img, [pts(c.get("points"))], True, (0, 200, 0), 1)
            b = line.find("pc:Baseline", NS)
            if b is not None:
                p = pts(b.get("points"))
                cv2.polylines(img, [p], False, (0, 0, 255), 2)
                cv2.circle(img, tuple(p[0]), 4, (255, 0, 0), -1)   # start point

        out = os.path.join(a.out_dir, os.path.basename(page.get("imageFilename")))
        cv2.imwrite(out, img)
        print(out)


if __name__ == "__main__":
    main()