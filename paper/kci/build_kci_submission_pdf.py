from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper" / "kci"
OUTPUT_PDF = OUT_DIR / "kci_submission_lite_globe_p.pdf"
SOURCE_MD = OUT_DIR / "kci_submission_lite_globe_p_source.md"

FONT_GOTHIC = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
FONT_MYUNGJO = "/System/Library/Fonts/Supplemental/AppleMyungjo.ttf"
FONT_MATH = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("KoreanGothic", FONT_GOTHIC))
    pdfmetrics.registerFont(TTFont("KoreanMyungjo", FONT_MYUNGJO))
    pdfmetrics.registerFont(TTFont("MathRoman", FONT_MATH))


def p(text: str, style: ParagraphStyle) -> Paragraph:
    safe = text.replace("<br/>", "\n").replace("<br />", "\n")
    return Paragraph(escape(safe).replace("\n", "<br/>"), style)


def section(title: str, styles: dict[str, ParagraphStyle]) -> list:
    return [Spacer(1, 5 * mm), p(title, styles["section"]), Spacer(1, 2 * mm)]


def subsection(title: str, styles: dict[str, ParagraphStyle]) -> list:
    return [Spacer(1, 3 * mm), p(title, styles["subsection"]), Spacer(1, 1.2 * mm)]


def formula(lines: list[str], styles: dict[str, ParagraphStyle]) -> Table:
    body = [p(line, styles["formula"]) for line in lines]
    table = Table([[item] for item in body], colWidths=[166 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7F7F7")),
                ("BOX", (0, 0), (-1, -1), 0.35, colors.HexColor("#BDBDBD")),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def caption(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return p(text, styles["caption"])


def make_table(data: list[list[str]], widths: list[float], font_size: float = 7.7) -> Table:
    table = Table(data, colWidths=[w * mm for w in widths], repeatRows=1, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "KoreanGothic"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 2.0),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EDF3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1F2933")),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9CED6")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#6B7280")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFAFA")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def bullet(items: list[str], styles: dict[str, ParagraphStyle]) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item, styles["body"]), leftIndent=4) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=12,
        bulletFontName="KoreanMyungjo",
        bulletFontSize=8.5,
    )


def on_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFont("KoreanGothic", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(22 * mm, height - 13 * mm, "KCI 투고용 원고")
    canvas.drawRightString(width - 22 * mm, height - 13 * mm, "Lite-GLOBE-P UAV Routing")
    canvas.setStrokeColor(colors.HexColor("#D0D0D0"))
    canvas.setLineWidth(0.3)
    canvas.line(22 * mm, height - 16 * mm, width - 22 * mm, height - 16 * mm)
    canvas.drawCentredString(width / 2, 12 * mm, f"- {doc.page} -")
    canvas.restoreState()


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="KoreanGothic",
            fontSize=17.5,
            leading=23,
            alignment=TA_CENTER,
            spaceAfter=7,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName="KoreanGothic",
            fontSize=9.5,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4A5568"),
            spaceAfter=7,
        ),
        "abstract_title": ParagraphStyle(
            "abstract_title",
            parent=base["Heading2"],
            fontName="KoreanGothic",
            fontSize=10.5,
            leading=14,
            alignment=TA_LEFT,
            spaceBefore=5,
            spaceAfter=4,
        ),
        "abstract": ParagraphStyle(
            "abstract",
            parent=base["BodyText"],
            fontName="KoreanMyungjo",
            fontSize=8.8,
            leading=13,
            alignment=TA_JUSTIFY,
            firstLineIndent=0,
        ),
        "keywords": ParagraphStyle(
            "keywords",
            parent=base["BodyText"],
            fontName="KoreanGothic",
            fontSize=8.5,
            leading=12,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#374151"),
        ),
        "section": ParagraphStyle(
            "section",
            parent=base["Heading1"],
            fontName="KoreanGothic",
            fontSize=12.2,
            leading=16,
            spaceBefore=8,
            spaceAfter=3,
            textColor=colors.HexColor("#111827"),
        ),
        "subsection": ParagraphStyle(
            "subsection",
            parent=base["Heading2"],
            fontName="KoreanGothic",
            fontSize=10.4,
            leading=14,
            spaceBefore=5,
            spaceAfter=2,
            textColor=colors.HexColor("#1F2937"),
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="KoreanMyungjo",
            fontSize=9.25,
            leading=14.4,
            alignment=TA_JUSTIFY,
            firstLineIndent=8,
            spaceAfter=4,
        ),
        "body_noindent": ParagraphStyle(
            "body_noindent",
            parent=base["BodyText"],
            fontName="KoreanMyungjo",
            fontSize=9.25,
            leading=14.4,
            alignment=TA_JUSTIFY,
            firstLineIndent=0,
            spaceAfter=4,
        ),
        "formula": ParagraphStyle(
            "formula",
            parent=base["Code"],
            fontName="MathRoman",
            fontSize=9.1,
            leading=11.2,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#111827"),
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["BodyText"],
            fontName="KoreanGothic",
            fontSize=8.0,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#374151"),
            spaceBefore=2,
            spaceAfter=5,
        ),
        "reference": ParagraphStyle(
            "reference",
            parent=base["BodyText"],
            fontName="KoreanMyungjo",
            fontSize=8.0,
            leading=11.4,
            alignment=TA_LEFT,
            firstLineIndent=-10,
            leftIndent=10,
            spaceAfter=3,
        ),
    }


def build_source_markdown() -> str:
    return """# KCI 제출용 원고: Lite-GLOBE-P

이 파일은 `kci_submission_lite_globe_p.pdf` 생성을 위한 한국어 원고 소스입니다. 특정 학회/저널 템플릿이 제공되지 않았기 때문에 KCI 일반 투고 원고에 맞춘 보수적 1단 조판으로 작성했습니다.

## 핵심 주장

- 제안기법 Risk-Switch Lite-GLOBE-P는 offline global teacher, global-to-local distillation, online local student, predictive risk-switch를 결합한다.
- Phase 12 전체 14개 시나리오, 5개 seed, 84,000 episode 평가에서 PDR과 deadline delivery가 비교군 중 최고였다.
- Predictive Geographic, Evo-QGeo, DRAMA 대비 낮은 p95 지연과 낮은 입력 정보량을 보였다.
- GPSR 대비 신뢰성은 크게 개선되지만, 에너지 및 입력 정보량은 증가하므로 해당 부분은 한계로 명시한다.

## 수식 구성

1. Global teacher 상태: s_t = (G_t, p_t, v_t, q_t, d)
2. PPO 목적함수와 clipped surrogate loss
3. Teacher-student KD loss
4. Danger score, safety utility, risk-switch decision
5. 최종 next-hop 선택식
"""


def build_pdf() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    register_fonts()
    styles = build_styles()

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=21 * mm,
        rightMargin=21 * mm,
        topMargin=23 * mm,
        bottomMargin=20 * mm,
        title="KCI Submission - Lite-GLOBE-P",
        author="GLOBE Routing Project",
    )

    story: list = []

    story.append(p("UAV 군집 네트워크를 위한 지식 증류 기반<br/>예측 위험 전환 라우팅 기법", styles["title"]))
    story.append(p("Risk-Switch Lite-GLOBE-P: Knowledge-Distilled Predictive Risk-Switch Routing for UAV Swarm Networks", styles["subtitle"]))
    story.append(p("익명 심사용 원고 / KCI 일반 투고 양식 기반", styles["subtitle"]))

    abstract_ko = (
        "무인비행체(UAV) 군집 네트워크는 빠른 이동성, 빈번한 링크 단절, 제한된 노드별 관측 범위로 인해 "
        "기존 위치 기반 라우팅만으로 안정적인 패킷 전달을 보장하기 어렵다. 본 논문은 학습 시에는 전체 네트워크 "
        "상태를 활용하고, 실제 운용 시에는 각 UAV가 1-hop 이웃 정보만으로 다음 홉을 선택하는 Risk-Switch "
        "Lite-GLOBE-P 라우팅 기법을 제안한다. 제안기법은 PPO 기반 global teacher를 오프라인에서 학습한 뒤, "
        "teacher의 전역 경로 판단을 local student 정책으로 증류한다. 또한 곧 끊길 가능성이 높은 링크, onward "
        "redundancy 부족, 큐 혼잡, 에너지 비효율 경로를 감지하는 predictive risk-switch를 추가하여 nominal "
        "student가 위험한 다음 홉을 선택할 때에만 안전 분기로 전환한다. 14개 시나리오, 5개 seed, 총 84,000개 "
        "episode 평가에서 제안기법은 GPSR, Predictive Geographic, Evo-QGeo, IQMR, DRAMA 및 내부 ablation "
        "기법과 비교하여 최고 평균 PDR 0.905와 deadline delivery 0.838을 달성하였다. 특히 DRAMA 대비 PDR은 "
        "1.6% 향상되고 입력 정보량은 22.7% 감소하였다. 결과적으로 제안기법은 전역 학습의 장점과 분산 실행의 "
        "현실성을 동시에 만족하는 UAV 라우팅 구조임을 보인다."
    )
    abstract_en = (
        "This paper proposes Risk-Switch Lite-GLOBE-P, a knowledge-distilled predictive routing framework for UAV swarm networks. "
        "A reinforcement-learning-based global teacher is trained offline with privileged network-wide graph information, and its "
        "routing behavior is distilled into a lightweight local student that uses only one-hop observations during deployment. "
        "The deployed router further incorporates a predictive risk-switch that overrides the nominal student decision only when "
        "the selected next hop is likely to suffer link breakage, insufficient onward connectivity, congestion, or inefficient energy use. "
        "In a 14-scenario evaluation with five random seeds and 84,000 episodes, the proposed method achieves the best overall packet "
        "delivery ratio and deadline delivery among the implemented baselines while reducing the input footprint against predictive and "
        "message-passing routing baselines."
    )

    story.append(p("국문초록", styles["abstract_title"]))
    story.append(p(abstract_ko, styles["abstract"]))
    story.append(p("주요어: UAV 네트워크, FANET, 라우팅, 강화학습, 지식 증류, Predictive Risk-Switch", styles["keywords"]))
    story.append(Spacer(1, 3 * mm))
    story.append(p("Abstract", styles["abstract_title"]))
    story.append(p(abstract_en, styles["abstract"]))
    story.append(p("Keywords: UAV Networks, FANET, Routing, Reinforcement Learning, Knowledge Distillation, Predictive Risk-Switch", styles["keywords"]))

    story += section("1. 서론", styles)
    story.append(
        p(
            "UAV 군집 네트워크는 재난 감시, 지능형 교통, 군집 탐사, 임시 통신망 구축과 같이 고정 인프라가 부족하거나 빠르게 변하는 환경에서 중요한 역할을 수행한다. 그러나 UAV 노드는 지상 센서망과 달리 높은 이동성을 가지며, 상대 위치와 속도 변화에 따라 링크 품질이 급격히 변한다. 이 때문에 특정 시점에서 목적지에 가장 가까워 보이는 이웃을 선택하는 단순 greedy 라우팅은 local dead-end, routing hole, 곧 끊길 링크 선택과 같은 문제에 취약하다.",
            styles["body"],
        )
    )
    story.append(
        p(
            "최근 연구들은 Q-learning, multi-agent reinforcement learning, predictive geographic routing을 활용하여 UAV 라우팅의 신뢰성을 높이고자 하였다. 하지만 강화학습 기반 기법은 온라인 메시지 교환이나 전역 상태 추정이 요구될 수 있고, 예측형 기법은 링크 안정성을 반영하지만 학습된 전역 경로 선호를 충분히 활용하지 못하는 경우가 있다. 본 연구의 핵심 질문은 다음과 같다. 전역 정보를 이용해 좋은 라우팅 판단을 학습하되, 실제 UAV에서는 낮은 계산량과 낮은 관측 범위로 실행할 수 있는가?",
            styles["body"],
        )
    )
    story.append(
        p(
            "이를 위해 본 논문은 Risk-Switch Lite-GLOBE-P를 제안한다. 제안기법은 offline 단계에서 global teacher를 강화학습으로 학습하고, 그 정책을 local student로 증류한 뒤, online 단계에서는 predictive risk-switch를 통해 위험한 다음 홉만 선택적으로 대체한다. 즉, 항상 복잡한 예측 경로 탐색을 수행하지 않고, nominal student가 충분히 안전하면 그대로 사용하며, 위험 신호가 명확할 때만 보수적인 안전 분기로 전환한다.",
            styles["body"],
        )
    )
    story.append(p("본 논문의 주요 기여는 다음과 같다.", styles["body_noindent"]))
    story.append(
        bullet(
            [
                "전역 네트워크 정보를 활용하는 PPO 기반 global teacher와 1-hop 관측만 사용하는 local student를 결합한 global-to-local 라우팅 구조를 제안한다.",
                "링크 수명, 링크 마진, onward redundancy, survival probability, 에너지 proxy를 이용한 predictive risk-switch를 설계하여 곧 끊길 링크 선택 문제를 완화한다.",
                "14개 시나리오와 5개 seed 기반 평가를 통해 GPSR 및 최신 RL/예측형 baseline 대비 PDR, deadline delivery, p95 지연, 입력 정보량 관점의 성능을 검증한다.",
            ],
            styles,
        )
    )

    story += section("2. 관련 연구", styles)
    story.append(
        p(
            "GPSR은 위치 정보를 이용해 목적지에 가까운 이웃으로 패킷을 전달하는 대표적인 geographic routing 기법이다. 구조가 단순하고 추가 메시지 비용이 낮지만, UAV처럼 topology가 빠르게 바뀌는 환경에서는 가까운 이웃이 곧 단절되거나 이후 경로가 막힐 수 있다. Evo-QGeo와 IQMR 계열 연구는 Q-learning 기반으로 링크 상태 변화와 multi-hop routing을 고려하며, DRAMA는 multi-agent reinforcement learning과 emergent communication을 통해 동적 라우팅을 수행한다. 이러한 방법들은 성능을 개선하지만, 온라인 학습/통신 비용, 입력 feature 크기, 훈련 안정성 문제가 함께 증가할 수 있다.",
            styles["body"],
        )
    )
    story.append(
        p(
            "본 연구는 기존 RL 라우팅과 달리 강화학습을 실제 UAV 실행 시점에 직접 수행하지 않는다. 강화학습은 오프라인 global teacher 학습에 집중하고, 배포 시에는 local student와 risk-switch가 deterministic 또는 lightweight inference로 동작한다. 따라서 제안기법은 중앙집중형 학습과 분산 실행의 절충 구조로 볼 수 있으며, 특히 teacher의 전역 경로 지식을 local policy로 압축한다는 점에서 차별성을 가진다.",
            styles["body"],
        )
    )

    story += section("3. 시스템 모델 및 문제 정의", styles)
    story.append(
        p(
            "시간 t에서 UAV 군집은 동적 그래프 G_t=(V_t,E_t)로 표현된다. V_t는 UAV 노드 집합이고, E_t는 통신 가능 링크 집합이다. 노드 u의 위치와 속도는 각각 p_u(t), v_u(t)이며, 큐 상태는 q_u(t)로 둔다. 링크 (u,i)는 통신 반경 R_c, 수신 신호 여유도, 상대 이동 방향, 채널 안정성에 따라 생성 또는 소멸한다. 각 패킷은 출발지에서 목적지 d까지 제한 시간 안에 전달되어야 하며, 각 홉에서 노드 u는 이웃 집합 N_u(t) 중 하나를 다음 홉으로 선택하거나 drop action을 선택한다.",
            styles["body"],
        )
    )
    story.append(
        KeepTogether(
            [
                formula(
                    [
                        "G_t = (V_t, E_t)",
                        "N_u(t) = { i | (u,i) in E_t }",
                        "a_u(t) in N_u(t) union {DROP}",
                    ],
                    styles,
                ),
                caption("식 1. 동적 UAV 네트워크와 노드 u의 다음 홉 선택 공간", styles),
            ]
        )
    )
    story.append(
        p(
            "본 연구의 목적은 packet delivery ratio(PDR)와 deadline delivery를 높이면서 p95 지연, energy proxy, 입력 정보량을 과도하게 증가시키지 않는 next-hop 정책을 학습하는 것이다. 여기서 입력 정보량은 각 라우팅 판단에 필요한 feature footprint를 나타내며, 실제 통신 오버헤드의 완전한 대체 지표는 아니지만 복잡한 메시지 기반 baseline과의 실행 부담 비교에 유용한 proxy로 사용한다.",
            styles["body"],
        )
    )

    story += section("4. 제안 기법", styles)
    story += subsection("4.1 전체 구조", styles)
    overview = ROOT / "paper" / "figures" / "method_overview_pub_preview.png"
    if overview.exists():
        img = Image(str(overview), width=156 * mm, height=82 * mm)
        story.append(KeepTogether([img, caption("그림 1. Risk-Switch Lite-GLOBE-P의 전체 구조: offline teacher 학습, student 증류, online risk-switch 실행", styles)]))
    story.append(
        p(
            "제안기법은 세 단계로 구성된다. 첫째, global teacher는 전체 topology와 큐/속도/위치 정보를 관측하면서 PPO로 학습된다. 둘째, teacher가 수렴한 뒤 local student는 teacher의 soft action distribution을 모방하도록 지식 증류된다. 셋째, 실제 UAV 배포 시에는 student가 기본 다음 홉을 제안하고, predictive risk-switch가 해당 선택이 위험하다고 판단할 때만 안전 후보로 전환한다.",
            styles["body"],
        )
    )

    story += subsection("4.2 강화학습 기반 전역 Teacher 학습", styles)
    story.append(
        p(
            "Teacher는 훈련 단계에서만 사용되는 정책 pi_T이다. teacher의 상태는 전체 그래프, 모든 노드 위치, 속도, 큐 상태, 목적지를 포함한다. 이는 실제 UAV 한 대가 운용 중 직접 관측하기 어려운 privileged information이지만, offline simulation에서는 사용할 수 있으므로 최적에 가까운 경로 판단을 학습하는 지도자 역할을 한다.",
            styles["body"],
        )
    )
    story.append(
        KeepTogether(
            [
                formula(
                    [
                        "s_t = (G_t, p_t, v_t, q_t, d)",
                        "a_t ~ π_T(a_t | s_t; θ_T)",
                        "J(θ_T) = E_{π_T} [ Σ_t γ^t r_t ]",
                    ],
                    styles,
                ),
                caption("식 2. Global teacher의 상태, 행동 선택, 강화학습 목적함수", styles),
            ]
        )
    )
    story.append(
        p(
            "보상 r_t는 성공 전달 보상에서 지연, 에너지, 불필요한 홉, 실패 penalty를 차감하는 방식으로 설계한다. 따라서 teacher는 단순히 목적지에 가까운 노드를 고르는 것이 아니라, 전달 성공 가능성과 경로 안정성, deadline 만족 가능성을 함께 고려한다.",
            styles["body"],
        )
    )
    story.append(
        KeepTogether(
            [
                formula(
                    [
                        "r_t = α_succ R_succ - α_D D_t - α_E E_t - α_H H_t - α_F F_t",
                        "ρ_t(θ_T) = π_T(a_t|s_t;θ_T) / π_T(a_t|s_t;θ_T_old)",
                        "L_PPO = - E_t [ min( ρ_t A_t, clip(ρ_t, 1-ε, 1+ε) A_t ) ]",
                    ],
                    styles,
                ),
                caption("식 3. Teacher 보상과 PPO clipped surrogate loss", styles),
            ]
        )
    )

    story += subsection("4.3 전역-로컬 정책 증류", styles)
    story.append(
        p(
            "Teacher 학습이 끝나면 teacher parameter theta_T*는 고정된다. 이후 student pi_S는 노드 u의 local observation o_u만으로 teacher의 next-hop 판단을 모방하도록 학습된다. 이때 hard label만 복사하지 않고, teacher가 각 후보에 부여한 soft probability를 KL divergence로 맞춘다. 이를 통해 student는 하나의 정답 후보뿐 아니라 후보 간 선호도 차이까지 학습한다.",
            styles["body"],
        )
    )
    story.append(
        KeepTogether(
            [
                formula(
                    [
                        "y_T(a) = softmax( z_T(a) / T ),     y_S(a) = softmax( z_S(a) / T )",
                        "L_KD = λ_KL T² KL(y_T || y_S) + λ_CE CE(a_T, π_S) + λ_O L_oracle",
                        "a_N = argmax_a π_S(a | o_u; θ_S)",
                    ],
                    styles,
                ),
                caption("식 4. Teacher-student 지식 증류와 nominal next-hop 선택", styles),
            ]
        )
    )
    story.append(
        p(
            "여기서 T는 distillation temperature, KL은 Kullback-Leibler divergence, CE는 cross entropy이다. L_oracle은 기하학적으로 명백히 잘못된 우회 또는 loop를 억제하기 위한 보조 손실이다. 배포 시 UAV는 teacher를 탑재하지 않으며, 학습된 student parameter만 사용한다.",
            styles["body"],
        )
    )

    story += subsection("4.4 예측 위험 전환 라우팅", styles)
    story.append(
        p(
            "Student만 사용할 경우 일반 topology에서는 효율적이지만, 곧 끊길 링크 또는 onward path가 부족한 후보를 선택할 수 있다. 이를 보완하기 위해 각 후보 i에 대해 링크 마진 m_i, 예측 링크 수명 l_i, 큐 headroom q_i, onward stability o_i, top-k onward link 평균, redundancy, survival probability, 에너지 효율 eta_i를 계산한다.",
            styles["body"],
        )
    )
    story.append(
        KeepTogether(
            [
                formula(
                    [
                        "x_i^risk = [ m_i, l_i, q_i, o_i ]",
                        "x_i^+ = [ o_bar_{i,topk}, ρ_i, p_{i,keep}, η_i ]",
                        "D_i = [g_m-m_i]_+ + [g_l-l_i]_+ + [g_o-o_i]_+ + [g_k-o_bar_{i,topk}]_+",
                        "      + 0.5[g_ρ-ρ_i]_+ + [g_p-p_{i,keep}]_+",
                        "Q_i = m_i + l_i + o_i + o_bar_{i,topk} + 0.5ρ_i + p_{i,keep}",
                    ],
                    styles,
                ),
                caption("식 5. 후보 next-hop의 위험도 D_i와 안전 효용 Q_i", styles),
            ]
        )
    )
    story.append(
        p(
            "위험도 D_i는 안전 기준보다 부족한 정도를 누적한 값이다. 반대로 Q_i는 후보가 얼마나 안정적인지 나타내는 안전 효용이다. risk-switch는 nominal action a_N이 충분히 안전하면 그대로 두고, drop이 선택되었거나 위험도가 임계값을 넘거나 predictive 후보 a_P가 명확히 더 안전할 때만 전환한다.",
            styles["body"],
        )
    )
    story.append(
        KeepTogether(
            [
                formula(
                    [
                        "G(a_P,a_N) = Q_{a_P} - Q_{a_N}",
                        "S = 1[ a_N=DROP or D_{a_N}>τ or (a_P≠a_N and G(a_P,a_N)>δ and D_{a_P}<D_{a_N}) ]",
                        "a* = a_P  if S=1",
                        "a* = a_N  if S=0",
                    ],
                    styles,
                ),
                caption("식 6. Risk-switch 조건과 최종 routing action", styles),
            ]
        )
    )
    story.append(
        p(
            "이 설계의 핵심은 보수적인 안전 분기를 항상 사용하지 않는다는 점이다. 일반 상황에서는 student의 빠른 결정을 유지하고, 예측 위험이 커지는 상황에서만 predictive branch로 전환한다. 따라서 전역 학습 기반 성능, local 실행 가능성, 링크 단절 회피를 동시에 겨냥한다.",
            styles["body"],
        )
    )

    story += section("5. 실험 환경 및 성능 평가", styles)
    story.append(
        p(
            "성능 평가는 Phase 12 full evaluation 결과를 사용하였다. 전체 실험은 5개 random seed(42, 77, 123, 314, 2718), 14개 scenario, scenario당 200 episode로 구성되며 총 84,000 episode row를 포함한다. 비교 기법은 GPSR, Predictive Geographic, Evo-QGeo, IQMR Q(lambda), DRAMA, Phase 8 Geo-Residual KD, Lite-GLOBE-P ablation을 포함한다.",
            styles["body"],
        )
    )
    perf_data = [
        ["Method", "PDR", "Deadline", "Delay p95", "Energy", "Input bytes"],
        ["GPSR", "0.683", "0.637", "4.163", "1.166", "2,284"],
        ["Predictive Geographic", "0.892", "0.823", "4.500", "1.809", "5,531"],
        ["Evo-QGeo", "0.887", "0.815", "4.557", "1.812", "6,452"],
        ["IQMR Q(lambda)", "0.516", "0.385", "6.289", "1.946", "7,953"],
        ["DRAMA", "0.891", "0.822", "4.504", "1.724", "6,240"],
        ["Phase 8 Geo-Residual KD", "0.803", "0.740", "4.110", "1.424", "3,956"],
        ["Lite-GLOBE-P prior only", "0.905", "0.838", "4.334", "1.776", "5,424"],
        ["Lite-GLOBE-P no-switch", "0.890", "0.822", "4.300", "1.726", "5,775"],
        ["Risk-Switch Lite-GLOBE-P", "0.905", "0.838", "4.264", "1.779", "4,821"],
    ]
    story.append(KeepTogether([make_table(perf_data, [50, 20, 23, 25, 22, 26]), caption("표 1. 14개 시나리오 전체 평균 성능 비교. PDR/Deadline은 높을수록, 나머지는 낮을수록 좋음", styles)]))
    story.append(
        p(
            "표 1에서 제안기법은 평균 PDR 0.905와 deadline delivery 0.838을 달성하여 전체 비교군 중 최고 수준을 보였다. delay p95는 Phase 8보다 높지만, Phase 8은 predictive-break 조건에서 실패율이 높기 때문에 단순 지연만으로 우수하다고 판단하기 어렵다. 제안기법은 Predictive Geographic, Evo-QGeo, DRAMA보다 낮은 delay p95를 보이며, 입력 정보량 역시 이들 baseline보다 작다.",
            styles["body"],
        )
    )

    improve_data = [
        ["Baseline", "PDR", "Deadline", "Delay p95", "Energy", "Input bytes"],
        ["GPSR", "+32.5%", "+31.5%", "2.4% worse", "52.6% worse", "111.1% more"],
        ["Predictive Geo.", "+1.5%", "+1.8%", "5.2% better", "1.6% better", "12.8% lower"],
        ["Evo-QGeo", "+2.1%", "+2.7%", "6.4% better", "1.8% better", "25.3% lower"],
        ["IQMR", "+75.4%", "+117.4%", "32.2% better", "8.6% better", "39.4% lower"],
        ["DRAMA", "+1.6%", "+1.9%", "5.3% better", "3.2% worse", "22.7% lower"],
    ]
    story.append(KeepTogether([make_table(improve_data, [40, 24, 25, 30, 27, 30], font_size=7.5), caption("표 2. Risk-Switch Lite-GLOBE-P의 baseline 대비 개선율", styles)]))

    for fname, cap in [
        ("phase12_pdr.png", "그림 2. 시나리오별 PDR 비교"),
        ("phase12_delay_p95.png", "그림 3. 시나리오별 p95 지연 비교"),
        ("phase12_input_bytes.png", "그림 4. 시나리오별 입력 정보량 비교"),
    ]:
        path = ROOT / "paper" / "figures" / fname
        if path.exists():
            story.append(Spacer(1, 2 * mm))
            story.append(KeepTogether([Image(str(path), width=156 * mm, height=83 * mm), caption(cap, styles)]))

    story += section("6. 논의", styles)
    story.append(
        p(
            "제안기법의 가장 큰 장점은 신뢰성이다. GPSR 대비 PDR과 deadline delivery가 크게 향상되었으며, predictive 및 message-passing 성격의 baseline과 비교해도 전체 평균에서 우수한 성능을 보였다. 특히 DRAMA 대비 PDR을 1.6% 개선하면서 입력 정보량을 22.7% 줄인 점은, 복잡한 online communication 없이도 경쟁력 있는 라우팅 성능을 얻을 수 있음을 시사한다.",
            styles["body"],
        )
    )
    story.append(
        p(
            "다만 모든 지표에서 무조건 우월한 것은 아니다. GPSR은 구조가 매우 단순하기 때문에 energy proxy와 입력 정보량 측면에서 여전히 유리하다. 또한 DRAMA 대비 energy proxy가 3.2% 나쁘게 나타났다. 이는 제안기법이 위험 링크를 피하기 위해 더 안정적인 우회 경로를 선택하면서 일부 홉 또는 거리 비용이 증가하기 때문이다. 따라서 실제 투고 시에는 '모든 지표 절대 우월'이 아니라 '신뢰성 및 deadline 성능을 가장 크게 개선하면서 predictive/RL baseline 대비 낮은 입력 footprint를 달성'했다는 식으로 주장하는 것이 타당하다.",
            styles["body"],
        )
    )
    story.append(
        p(
            "향후 개선 방향은 세 가지다. 첫째, risk-switch activation을 energy-aware하게 조정하여 안전성과 에너지 간 균형을 더 세밀하게 맞춰야 한다. 둘째, predictive-break with link-loss 조건에서 Evo-QGeo보다 낮은 일부 case를 개선하기 위해 link survival probability와 onward redundancy의 calibration을 강화해야 한다. 셋째, 실제 무선 채널 모델과 제어 패킷 overhead를 포함한 평가로 input bytes proxy의 한계를 보완해야 한다.",
            styles["body"],
        )
    )

    story += section("7. 결론", styles)
    story.append(
        p(
            "본 논문은 UAV 군집 네트워크에서 전역 학습과 로컬 실행을 결합한 Risk-Switch Lite-GLOBE-P 라우팅 기법을 제안하였다. 제안기법은 PPO 기반 global teacher를 offline에서 학습하고, teacher의 전역 경로 판단을 local student에 증류하며, online 실행 시 predictive risk-switch를 통해 곧 끊길 링크와 막다른 경로를 회피한다. Phase 12 full evaluation에서 제안기법은 전체 평균 PDR 0.905, deadline delivery 0.838을 달성하여 구현된 비교 기법 중 최고 수준의 신뢰성을 보였다. 이러한 결과는 제안기법이 실제 UAV 분산 라우팅 환경에서 높은 전달 신뢰성과 실행 가능성을 동시에 제공할 수 있음을 보여준다.",
            styles["body"],
        )
    )

    story += section("참고문헌", styles)
    references = [
        "[1] B. Karp and H. T. Kung, \"GPSR: Greedy Perimeter Stateless Routing for Wireless Networks,\" Proc. ACM MobiCom, pp. 243-254, 2000.",
        "[2] M. Xu, Y. Xia, W. Liu, and D. Huang, \"Reinforcement-Learning-Based Geographic Routing Considering Future Evolution of Link States for UAV Networks,\" IEEE Transactions on Vehicular Technology, 2026.",
        "[3] W. Zhang, C. Liu, J. Jiang, et al., \"DRAMA: A Dynamic Packet Routing Algorithm using Multi-Agent Reinforcement Learning with Emergent Communication,\" Proc. IJCNN, pp. 1-8, 2025.",
        "[4] X. Zeng, X. Wang, et al., \"Improved Q-learning based Multi-hop Routing for UAV-Assisted Communication,\" IEEE Transactions on Network and Service Management, 2024.",
        "[5] A. I. Ahmed, et al., \"GLo-MAPPO: Multi-Agent Deep Reinforcement Learning for Energy-Efficient UAV-Assisted LoRa Networks,\" arXiv:2509.17676, 2025.",
        "[6] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, \"Proximal Policy Optimization Algorithms,\" arXiv:1707.06347, 2017.",
        "[7] G. Hinton, O. Vinyals, and J. Dean, \"Distilling the Knowledge in a Neural Network,\" arXiv:1503.02531, 2015.",
        "[8] A. A. Rusu, et al., \"Policy Distillation,\" Proc. ICLR, 2016.",
        "[9] GLOBE Routing Project, \"Risk-Switch Lite-GLOBE-P Phase 12 Full Evaluation Artifacts,\" Local experiment artifacts, 2026.",
    ]
    for ref in references:
        story.append(p(ref, styles["reference"]))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    SOURCE_MD.write_text(build_source_markdown(), encoding="utf-8")


if __name__ == "__main__":
    build_pdf()
    print(OUTPUT_PDF)
    print(SOURCE_MD)
