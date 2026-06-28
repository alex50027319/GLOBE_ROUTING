from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.pdf"
FIGURE = ROOT / "figures" / "method_overview_pub.pdf"
OUT = ROOT / "main_final.pdf"

FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("AU", FONT))
    pdfmetrics.registerFont(TTFont("AUB", BOLD))


def draw_text(c: canvas.Canvas, x: float, y: float, text: str, size: float = 8.0, bold: bool = False) -> None:
    c.setFillColor(colors.black)
    c.setFont("AUB" if bold else "AU", size)
    c.drawString(x, y, text)


def draw_center(c: canvas.Canvas, x: float, y: float, text: str, size: float = 8.0, bold: bool = False) -> None:
    c.setFillColor(colors.black)
    c.setFont("AUB" if bold else "AU", size)
    c.drawCentredString(x, y, text)


def draw_box(c: canvas.Canvas, x: float, y: float, w: float, h: float, title: str, lines: list[str]) -> None:
    c.setStrokeColor(colors.black)
    c.setFillColor(colors.Color(0.965, 0.965, 0.965))
    c.roundRect(x, y, w, h, 5, stroke=1, fill=1)
    draw_text(c, x + 8, y + h - 14, title, 8.4, True)
    yy = y + h - 28
    for line in lines:
        draw_text(c, x + 10, yy, line, 7.3)
        yy -= 11


def draw_equation(c: canvas.Canvas, x: float, y: float, lines: list[str], size: float = 8.6) -> float:
    c.setFillColor(colors.black)
    c.setFont("AU", size)
    yy = y
    for line in lines:
        c.drawString(x, yy, line)
        yy -= size + 4
    return yy


def make_teacher_rl_page() -> PdfReader:
    register_fonts()
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    page_w, page_h = letter

    margin = 48
    col_gap = 24
    col_w = (page_w - 2 * margin - col_gap) / 2
    left_x = margin
    right_x = margin + col_w + col_gap
    y = page_h - 54

    draw_center(c, page_w / 2, y, "IV-B. Reinforcement-Learning-Based Global Teacher Training", 12.0, True)
    y -= 24

    intro = [
        "Reinforcement learning is used before deployment to train the privileged global teacher.",
        "The deployed UAVs do not run PPO; they only execute the distilled local student and risk switch.",
    ]
    for line in intro:
        draw_text(c, left_x, y, line, 8.0)
        y -= 12

    y -= 6
    draw_box(
        c,
        left_x,
        y - 92,
        col_w,
        92,
        "1) Global Markov state and teacher policy",
        [
            "s_t = (G_t, p_t, v_t, q_t, d)",
            "a_t ~ pi_T(a_t | s_t; theta_T)",
            "G_t: global FANET graph",
            "p_t, v_t, q_t: position, velocity, queue state",
            "d: packet destination",
        ],
    )

    draw_box(
        c,
        right_x,
        y - 92,
        col_w,
        92,
        "2) Routing reward",
        [
            "r_t = alpha_s R_succ - alpha_D D_t - alpha_E E_t",
            "      - alpha_H H_t - alpha_F F_t",
            "R_succ: delivery reward",
            "D_t, E_t, H_t: delay, energy, hop penalties",
            "F_t: drop, loop, or route-failure penalty",
        ],
    )

    y -= 118
    draw_text(c, left_x, y, "The teacher maximizes expected discounted routing return:", 8.0, True)
    y -= 16
    y = draw_equation(
        c,
        left_x + 12,
        y,
        [
            "J(theta_T) = E_{pi_T} [ sum_{t=0}^{T_ep} gamma^t r_t ]",
        ],
        8.8,
    )

    y -= 8
    draw_text(c, left_x, y, "PPO updates the teacher by comparing the new and old policies:", 8.0, True)
    y -= 16
    y = draw_equation(
        c,
        left_x + 12,
        y,
        [
            "rho_t(theta_T) = pi_T(a_t | s_t; theta_T)",
            "                 / pi_T(a_t | s_t; theta_T_old)",
            "A_t = R_hat_t - V_T(s_t; phi_T)",
        ],
        8.2,
    )

    y2 = page_h - 54 - 24 - 24 - 118
    draw_text(c, right_x, y2, "The clipped PPO objective limits unstable policy updates:", 8.0, True)
    y2 -= 16
    y2 = draw_equation(
        c,
        right_x + 4,
        y2,
        [
            "L_PPO = -E_t [ min(",
            "    rho_t(theta_T) A_t,",
            "    clip(rho_t(theta_T), 1-epsilon, 1+epsilon) A_t",
            ") ]",
        ],
        8.0,
    )
    y2 -= 8
    draw_text(c, right_x, y2, "The value head and entropy regularizer are trained jointly:", 8.0, True)
    y2 -= 16
    y2 = draw_equation(
        c,
        right_x + 4,
        y2,
        [
            "L_V = E_t [(V_T(s_t; phi_T) - R_hat_t)^2]",
            "L_T = L_PPO + c_V L_V - c_H H(pi_T)",
        ],
        8.0,
    )

    mid_y = 284
    draw_box(
        c,
        left_x,
        mid_y,
        col_w,
        96,
        "3) What the global teacher learns",
        [
            "The teacher is rewarded for long-term routing outcome,",
            "not only for one-hop geographic progress.",
            "It can learn to prefer:",
            "- a slightly longer but more stable link",
            "- a neighbor with onward connectivity",
            "- a route avoiding drops, loops, and dead ends",
        ],
    )
    draw_box(
        c,
        right_x,
        mid_y,
        col_w,
        96,
        "4) Why the teacher is not deployed",
        [
            "The teacher uses s_t, which includes global topology.",
            "That information is expensive or unavailable online.",
            "Therefore only pi_S(. | o_u; theta_S*) is deployed.",
            "Each UAV executes local inference plus risk switching.",
            "No PPO update is performed during packet forwarding.",
        ],
    )

    # Bottom pipeline.
    bottom_y = 142
    draw_text(c, left_x, bottom_y + 82, "After teacher convergence, the teacher is frozen and becomes a distillation target:", 8.2, True)
    c.setStrokeColor(colors.black)
    c.setFillColor(colors.Color(0.98, 0.98, 0.98))
    c.roundRect(left_x, bottom_y, page_w - 2 * margin, 66, 5, stroke=1, fill=1)

    stages = [
        ("PPO teacher", "theta_T*"),
        ("Teacher logits", "pi_T(. | s_t; theta_T*)"),
        ("KD loss", "D_KL(pi_T || pi_S)"),
        ("Deploy student", "pi_S(. | o_u; theta_S*)"),
    ]
    box_w = 114
    gap = 14
    x = left_x + 12
    for idx, (title, body) in enumerate(stages):
        c.setFillColor(colors.white)
        c.roundRect(x, bottom_y + 16, box_w, 34, 4, stroke=1, fill=1)
        draw_center(c, x + box_w / 2, bottom_y + 38, title, 7.4, True)
        draw_center(c, x + box_w / 2, bottom_y + 25, body, 6.7)
        if idx < len(stages) - 1:
            c.line(x + box_w + 2, bottom_y + 33, x + box_w + gap - 2, bottom_y + 33)
            c.line(x + box_w + gap - 2, bottom_y + 33, x + box_w + gap - 7, bottom_y + 36)
            c.line(x + box_w + gap - 2, bottom_y + 33, x + box_w + gap - 7, bottom_y + 30)
        x += box_w + gap

    draw_text(c, left_x, 96, "Key deployment implication:", 8.4, True)
    draw_text(c, left_x, 82, "PPO and the global teacher are offline-only. During UAV operation, each node uses local observation o_u,", 7.6)
    draw_text(c, left_x, 70, "the distilled student weights theta_S*, and the risk-switch rule to select the next hop.", 7.6)

    c.showPage()
    c.save()
    buf.seek(0)
    return PdfReader(buf)


def refresh_method_figure(page) -> None:
    if not FIGURE.exists():
        return
    fig_page = PdfReader(str(FIGURE)).pages[0]
    fig_w = float(fig_page.mediabox.width)
    page_w = float(page.mediabox.width)
    page_h = float(page.mediabox.height)

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_w, page_h))
    c.setFillColor(colors.white)
    c.rect(40, 492, 532, 286, stroke=0, fill=1)
    c.save()
    buf.seek(0)
    page.merge_page(PdfReader(buf).pages[0])
    page.merge_transformed_page(fig_page, Transformation().scale(516 / fig_w, 516 / fig_w).translate(48, 500))


def main() -> None:
    teacher_page = make_teacher_rl_page().pages[0]
    reader = PdfReader(str(MAIN))
    writer = PdfWriter()
    inserted = False

    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if (
            "Reinforcement Learning for Global Teacher Training" in text
            or "Reinforcement-Learning-Based Global Teacher Training" in text
        ):
            continue
        if (not inserted) and ("System architecture of the proposed" in text or "The candidate safety utility" in text):
            writer.add_page(teacher_page)
            inserted = True
        if "System architecture of the proposed" in text or "The candidate safety utility" in text:
            refresh_method_figure(page)
        writer.add_page(page)

    if not inserted:
        writer.add_page(teacher_page)

    with OUT.open("wb") as handle:
        writer.write(handle)
    MAIN.write_bytes(OUT.read_bytes())
    print(MAIN)


if __name__ == "__main__":
    main()
