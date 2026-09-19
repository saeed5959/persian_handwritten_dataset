import glob, os
from lxml import etree
import argparse


PC = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

def convert(xml_dir, out_dir, ratio=0.35, rtl=False):
    os.makedirs(out_dir, exist_ok=True)

    for f in sorted(glob.glob(os.path.join(xml_dir, "*.xml"))):
        tree = etree.parse(f)
        n = 0
        for line in tree.getroot().iterfind(f".//{{{PC}}}TextLine"):
            coords = line.find(f"{{{PC}}}Coords")
            if coords is None:
                continue
            pts = [[float(v) for v in p.split(",")] for p in coords.get("points").split()]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x0, x1 = min(xs), max(xs)
            y = min(ys) + ratio * (max(ys) - min(ys))

            bl = [(x1, y), (x0, y)] if rtl else [(x0, y), (x1, y)]
            el = etree.Element(f"{{{PC}}}Baseline")
            el.set("points", " ".join(f"{round(a)},{round(b)}" for a, b in bl))
            coords.addnext(el)          # PAGE order: Coords, Baseline
            n += 1

        tree.write(os.path.join(out_dir, os.path.basename(f)),
                encoding="UTF-8", xml_declaration=True, pretty_print=True)
        print(f"{os.path.basename(f)}: {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml-dir", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ratio", type=float, default=0.62)
    ap.add_argument("--rtl", action="store_true")
    a = ap.parse_args()

    convert(a.xml_dir, a.out_dir, ratio=a.ratio, rtl=a.rtl)
    print("Done")

if __name__ == "__main__":
    main()