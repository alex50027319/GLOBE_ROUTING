from pathlib import Path
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent / "rendered"
for group in ("main", "supplementary", "cover"):
    files = sorted((root / group).glob("page-*.png"))
    if not files:
        continue
    thumbs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        im.thumbnail((260, 370))
        canvas = Image.new("RGB", (280, 405), "white")
        canvas.paste(im, ((280-im.width)//2, 20))
        ImageDraw.Draw(canvas).text((10, 385), f.stem, fill="black")
        thumbs.append(canvas)
    cols = 4
    rows = (len(thumbs)+cols-1)//cols
    sheet = Image.new("RGB", (cols*280, rows*405), "#D8DEE8")
    for i, im in enumerate(thumbs):
        sheet.paste(im, ((i%cols)*280, (i//cols)*405))
    sheet.save(root / f"{group}_contact_sheet.png")
