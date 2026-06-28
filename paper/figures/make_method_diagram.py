from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


BASE = Path(__file__).resolve().parent
PDF_OUT = BASE / "method_overview_pub.pdf"
PNG_OUT = BASE / "method_overview_pub_preview.png"

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
BOLD_FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

W_IN, H_IN = 7.15, 3.65
PDF_W, PDF_H = W_IN * inch, H_IN * inch
SCALE = 3
PNG_W, PNG_H = int(PDF_W * SCALE), int(PDF_H * SCALE)

BLACK = colors.black
WHITE = colors.white
GREY_1 = colors.Color(0.965, 0.965, 0.965)
GREY_2 = colors.Color(0.90, 0.90, 0.90)
GREY_3 = colors.Color(0.82, 0.82, 0.82)


def register_fonts():
    pdfmetrics.registerFont(TTFont("AU", FONT_PATH))
    pdfmetrics.registerFont(TTFont("AUB", BOLD_FONT_PATH))


def pdf_text(c, x, y, text, size=7.4, bold=False, align="center"):
    c.setFillColor(BLACK)
    c.setFont("AUB" if bold else "AU", size)
    if align == "center":
        c.drawCentredString(x, y, text)
    elif align == "right":
        c.drawRightString(x, y, text)
    else:
        c.drawString(x, y, text)


def pdf_box(c, x, y, w, h, title, lines, fill=WHITE, title_size=7.5, line_size=7.0):
    c.setStrokeColor(BLACK)
    c.setFillColor(fill)
    c.setLineWidth(0.85)
    c.roundRect(x, y, w, h, 5, stroke=1, fill=1)
    pdf_text(c, x + w / 2, y + h - 11, title, title_size, bold=True)
    yy = y + h - 24
    for line in lines:
        pdf_text(c, x + w / 2, yy, line, line_size)
        yy -= line_size + 2.5


def pdf_panel(c, x, y, w, h, title):
    c.setStrokeColor(BLACK)
    c.setFillColor(GREY_1)
    c.setLineWidth(0.85)
    c.roundRect(x, y, w, h, 5, stroke=1, fill=1)
    c.setFillColor(GREY_2)
    c.roundRect(x, y + h - 18, w, 18, 5, stroke=1, fill=1)
    pdf_text(c, x + w / 2, y + h - 12.2, title, 7.6, bold=True)


def pdf_arrow(c, x1, y1, x2, y2, label=None):
    c.setStrokeColor(BLACK)
    c.setLineWidth(0.9)
    c.line(x1, y1, x2, y2)
    import math

    angle = math.atan2(y2 - y1, x2 - x1)
    size = 4.0
    for delta in (2.55, -2.55):
        a = angle + delta
        c.line(x2, y2, x2 + size * math.cos(a), y2 + size * math.sin(a))
    if label:
        pdf_text(c, (x1 + x2) / 2, (y1 + y2) / 2 + 5, label, 6.2)


def make_pdf():
    register_fonts()
    c = canvas.Canvas(str(PDF_OUT), pagesize=(PDF_W, PDF_H))

    # Border.
    c.setStrokeColor(colors.Color(0.5, 0.5, 0.5))
    c.setLineWidth(0.75)
    c.rect(12, 12, PDF_W - 24, PDF_H - 24, stroke=1, fill=0)

    # Main flow panels.
    left_x, left_w = 24, 290
    right_x, right_w = 328, 164
    row_h = 56
    y1, y2, y3 = 186, 112, 38

    pdf_panel(c, left_x, y1, left_w, row_h, "1) Offline training: global teacher → local targets")
    pdf_panel(c, left_x, y2, left_w, row_h, "2) Deployment: local observation → nominal action")
    pdf_panel(c, left_x, y3, left_w, row_h, "3) Safeguard: predictive risk switch")
    pdf_panel(c, right_x, y3, right_w, row_h * 3 + 18, "Equation view")

    # Row 1.
    pdf_box(
        c,
        34,
        y1 + 7,
        76,
        35,
        "Global state",
        ["s_t = (G_t, p_t, v_t, q_t, d)"],
        fill=WHITE,
        line_size=6.5,
    )
    pdf_box(c, 126, y1 + 7, 76, 35, "Teacher", ["pi_T(a | s_t)"], fill=WHITE, line_size=8.0)
    pdf_box(
        c,
        218,
        y1 + 7,
        84,
        35,
        "KD target",
        ["z_T,  a_T = argmax pi_T"],
        fill=WHITE,
        line_size=6.4,
    )
    pdf_arrow(c, 110, y1 + 24, 126, y1 + 24)
    pdf_arrow(c, 202, y1 + 24, 218, y1 + 24)

    # Row 2.
    pdf_box(
        c,
        34,
        y2 + 7,
        76,
        35,
        "Local state",
        ["o_u(t) = {x_u, x_i, x_ui, d}"],
        fill=WHITE,
        line_size=6.1,
    )
    pdf_box(c, 126, y2 + 7, 76, 35, "Student", ["pi_S(a | o_u)"], fill=WHITE, line_size=8.0)
    pdf_box(c, 218, y2 + 7, 84, 35, "Nominal action", ["a_N = argmax pi_N"], fill=WHITE, line_size=7.2)
    pdf_arrow(c, 110, y2 + 24, 126, y2 + 24)
    pdf_arrow(c, 202, y2 + 24, 218, y2 + 24)

    # Row 3.
    pdf_box(
        c,
        34,
        y3 + 7,
        76,
        35,
        "Risk features",
        ["x_i = [m_i, ell_i, o_i,", "rho_i, p_keep, eta_i]"],
        fill=WHITE,
        line_size=5.8,
    )
    pdf_box(c, 126, y3 + 7, 76, 35, "Switch rule", ["S ∈ {0,1}"], fill=WHITE, line_size=8.0)
    pdf_box(c, 218, y3 + 7, 84, 35, "Final action", ["a* = a_P if S=1", "else a_N"], fill=GREY_2, line_size=6.8)
    pdf_arrow(c, 110, y3 + 24, 126, y3 + 24, "a_P")
    pdf_arrow(c, 202, y3 + 24, 218, y3 + 24, "a*")

    # Equation panel.
    x = right_x + 12
    y = y3 + row_h * 3 - 8
    eqs = [
        ("Distillation", True),
        ("L_KD = T^2 D_KL(pi_T || pi_S)", False),
        ("       + CE(a_T, pi_S) + L_oracle", False),
        ("", False),
        ("Danger score", True),
        ("D_i = [g_m - m_i]_+ + [g_l - ell_i]_+", False),
        ("     + [g_o - o_i]_+ + [g_p - p_keep]_+", False),
        ("", False),
        ("Safety gain", True),
        ("G(a_P,a_N) = Q(a_P) − Q(a_N)", False),
        ("Q_i = m_i + ell_i + o_i + rho_i + p_keep", False),
        ("", False),
        ("Switch decision", True),
        ("S = 1[a_N=DROP or D(a_N)>tau", False),
        ("      or G(a_P,a_N)>delta]", False),
        ("", False),
        ("Execution constraint", True),
        ("pi_T is offline only; a* uses local features.", False),
    ]
    for text, bold in eqs:
        if text == "":
            y -= 5
            continue
        pdf_text(c, x, y, text, 7.0 if not bold else 7.5, bold=bold, align="left")
        y -= 9.2 if not bold else 10.0

    c.showPage()
    c.save()


def load_font(size, bold=False):
    candidates = [
        BOLD_FONT_PATH if bold else FONT_PATH,
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def pxy(x, y):
    return int(x * SCALE), int((PDF_H - y) * SCALE)


def make_png_preview():
    img = Image.new("RGB", (PNG_W, PNG_H), "white")
    d = ImageDraw.Draw(img)
    font = load_font(20)
    font_b = load_font(21, bold=True)
    small = load_font(18)
    tiny = load_font(16)

    def rect(x, y, w, h, fill="#f7f7f7", outline="black", width=3):
        x1, y1 = pxy(x, y + h)
        x2, y2 = pxy(x + w, y)
        d.rounded_rectangle((x1, y1, x2, y2), radius=13, fill=fill, outline=outline, width=width)

    def text_center(x, y, w, h, lines, fnt=small, title=False):
        line_h = fnt.size + 5
        total = len(lines) * line_h
        yy = pxy(0, y + h / 2)[1] - total // 2
        for i, line in enumerate(lines):
            use = font_b if title and i == 0 else fnt
            bbox = d.textbbox((0, 0), line, font=use)
            xx = int((x + w / 2) * SCALE - (bbox[2] - bbox[0]) / 2)
            d.text((xx, yy), line, fill="black", font=use)
            yy += line_h

    def arrow(x1, y, x2):
        x1p, yp = pxy(x1, y)
        x2p, _ = pxy(x2, y)
        d.line((x1p, yp, x2p, yp), fill="black", width=4)
        d.polygon([(x2p, yp), (x2p - 14, yp - 8), (x2p - 14, yp + 8)], fill="black")

    for y, label in [
        (186, "1) Offline training: global teacher → local targets"),
        (112, "2) Deployment: local observation → nominal action"),
        (38, "3) Safeguard: predictive risk switch"),
    ]:
        rect(24, y, 290, 56, "#f5f5f5", width=2)
        x1, y1 = pxy(30, y + 50)
        d.text((x1, y1), label, fill="black", font=font_b)

    rows = [
        (193, [("Global state", "s_t=(G_t,p_t,v_t,q_t,d)"), ("Teacher", "pi_T(a|s_t)"), ("KD target", "z_T, a_T")]),
        (119, [("Local state", "o_u={x_u,x_i,x_ui,d}"), ("Student", "pi_S(a|o_u)"), ("Nominal", "a_N=argmax pi_N")]),
        (45, [("Risk feat.", "x_i=[m_i,ell_i,o_i,rho_i,...]"), ("Switch", "S in {0,1}"), ("Final", "a*=a_P if S=1")]),
    ]
    for y, boxes in rows:
        for x, (title, eq) in zip([34, 126, 218], boxes):
            w = 76 if x < 218 else 84
            rect(x, y, w, 35, "white")
            text_center(x, y, w, 35, [title, eq], tiny, title=True)
        arrow(110, y + 17, 126)
        arrow(202, y + 17, 218)

    rect(328, 38, 164, 206, "#f5f5f5", width=2)
    x0, y0 = pxy(340, 226)
    lines = [
        ("Equation view", font_b),
        ("L_KD = T^2 D_KL(pi_T||pi_S) + CE + L_oracle", small),
        ("D_i = [g_m-m_i]_+ + [g_l-ell_i]_+ + ...", small),
        ("G(a_P,a_N)=Q(a_P)-Q(a_N)", small),
        ("Q_i=m_i+ell_i+o_i+rho_i+p_keep", small),
        ("S=1[a_N=DROP or D(a_N)>tau", small),
        ("     or G(a_P,a_N)>delta]", small),
        ("pi_T offline only; a* uses local features.", small),
    ]
    yy = y0
    for text, fnt in lines:
        d.text((x0, yy), text, fill="black", font=fnt)
        yy += fnt.size + 12

    img.save(PNG_OUT)


def main():
    make_pdf()
    make_png_preview()


if __name__ == "__main__":
    main()
