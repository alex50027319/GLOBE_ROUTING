from pathlib import Path

from PIL import Image, ImageDraw


root = Path(__file__).resolve().parent / "rendered"
for group in ("main", "supplementary", "cover"):
    files = sorted((root / group).glob("page-*.png"))
    if not files:
        continue

    thumbnails = []
    for source in files:
        image = Image.open(source).convert("RGB")
        image.thumbnail((260, 370))
        card = Image.new("RGB", (280, 405), "white")
        card.paste(image, ((280 - image.width) // 2, 20))
        ImageDraw.Draw(card).text((10, 385), source.stem, fill="black")
        thumbnails.append(card)

    columns = 4
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 280, rows * 405), "#D8DEE8")
    for index, thumbnail in enumerate(thumbnails):
        sheet.paste(thumbnail, ((index % columns) * 280, (index // columns) * 405))
    sheet.save(root / f"{group}_contact_sheet.png")
