from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
PDF_SPECS = {
    "main": (ROOT / "build/main/main.pdf", 16, "DiSwitch"),
    "supplementary": (
        ROOT / "build/supplementary/supplementary.pdf",
        15,
        "Complete Figure Portfolio",
    ),
    "cover": (ROOT / "build/cover/cover_letter.pdf", 1, "Dear Editors"),
}
FORBIDDEN_TEXT = (
    "??",
    "TODO",
    "FIXME",
    "/Users/alex",
    "Undefined control sequence",
    "Fused" + " Exact",
    "Anonymous Author",
    "Affiliation withheld",
)


def dereference(value):
    return value.get_object() if hasattr(value, "get_object") else value


def font_record(font_reference):
    font = dereference(font_reference)
    base_font = str(font.get("/BaseFont", "unnamed"))
    subtype = str(font.get("/Subtype", "unknown"))
    candidates = [font]
    if subtype == "/Type0":
        candidates.extend(
            dereference(item) for item in dereference(font.get("/DescendantFonts", []))
        )

    embedded = subtype == "/Type3"
    for candidate in candidates:
        descriptor = candidate.get("/FontDescriptor")
        if descriptor:
            descriptor = dereference(descriptor)
            embedded = embedded or any(
                key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")
            )
    return base_font, subtype, embedded


def collect_fonts(resources, seen_resources, records):
    resources = dereference(resources)
    if not resources:
        return
    marker = id(resources)
    if marker in seen_resources:
        return
    seen_resources.add(marker)

    fonts = dereference(resources.get("/Font", {}))
    for reference in fonts.values():
        records.add(font_record(reference))

    xobjects = dereference(resources.get("/XObject", {}))
    for reference in xobjects.values():
        xobject = dereference(reference)
        nested = xobject.get("/Resources")
        if nested:
            collect_fonts(nested, seen_resources, records)


for name, (path, expected_pages, required_text) in PDF_SPECS.items():
    reader = PdfReader(path)
    if len(reader.pages) != expected_pages:
        raise AssertionError(f"{name}: expected {expected_pages} pages, found {len(reader.pages)}")

    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if required_text.lower() not in text.lower():
        raise AssertionError(f"{name}: required text not found: {required_text}")
    found_forbidden = [token for token in FORBIDDEN_TEXT if token in text]
    if found_forbidden:
        raise AssertionError(f"{name}: forbidden text found: {found_forbidden}")

    fonts = set()
    seen_resources = set()
    page_sizes = set()
    for page in reader.pages:
        collect_fonts(page.get("/Resources"), seen_resources, fonts)
        page_sizes.add((round(float(page.mediabox.width), 3), round(float(page.mediabox.height), 3)))

    unembedded = sorted(record for record in fonts if not record[2])
    if unembedded:
        raise AssertionError(f"{name}: unembedded fonts: {unembedded}")

    print(
        f"{name}: pages={len(reader.pages)}, page_sizes={sorted(page_sizes)}, "
        f"fonts={len(fonts)}, all_fonts_embedded=yes, extracted_chars={len(text)}"
    )
