import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const REPO = "/Users/alex/Documents/GLOBE_ROUTING";
const OUT_DIR = path.join(REPO, "ResearchAIWorkspace", "outputs", "presentations");
const VAULT_DIR = path.join(REPO, "ResearchAIWorkspace", "vault", "06_Experiments", "Lite-GLOBE");
const RENDER_DIR = path.join(OUT_DIR, "globe_professor_report_rendered");
const PPTX_PATH = path.join(OUT_DIR, "globe_lite_globe_professor_report_2026-06-28.pptx");
const NOTES_PATH = path.join(OUT_DIR, "globe_lite_globe_professor_report_2026-06-28_notes.md");
const OBSIDIAN_PATH = path.join(VAULT_DIR, "Professor_Presentation_2026-06-28.md");

const ARTIFACT_TOOL_ENTRYPOINT =
  process.env.ARTIFACT_TOOL_ENTRYPOINT ||
  "/private/tmp/globe_professor_deck_tool/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const { Presentation, PresentationFile } = await import(pathToFileURL(ARTIFACT_TOOL_ENTRYPOINT).href);

const W = 1280;
const H = 720;
const page = { left: 54, top: 42, width: 1172, height: 624 };
const c = {
  bg: "slate-50",
  ink: "slate-950",
  sub: "slate-600",
  muted: "slate-500",
  line: "slate-200",
  card: "white",
  blue: "blue-600",
  blue2: "sky-500",
  navy: "indigo-900",
  orange: "orange-500",
  green: "emerald-500",
  red: "rose-500",
  purple: "violet-500",
  yellow: "amber-400",
};

const phase12Overall = [
  ["GPSR", 0.683, 0.637, 4.163, 1.166, 2284],
  ["Phase8", 0.803, 0.740, 4.110, 1.424, 3956],
  ["Predictive Geo", 0.892, 0.823, 4.500, 1.809, 5531],
  ["Evo-QGeo", 0.887, 0.815, 4.557, 1.812, 6452],
  ["DRAMA", 0.891, 0.822, 4.504, 1.724, 6240],
  ["Risk-Switch", 0.905, 0.838, 4.264, 1.779, 4821],
];

const scenarioSummary = [
  ["일반", "0.946", "일반 성능 유지", "Phase8 0.948과 거의 동일"],
  ["Routing-hole", "0.849", "GPSR 실패 보완", "GPSR 0.040 대비 큰 개선"],
  ["Predictive-break", "0.756", "Phase8 약점 보완", "Phase8 0.032 -> 0.756"],
  ["노드 확장", "0.963", "확장성 유지", "DRAMA 0.948보다 우세"],
];

const phaseTimeline = [
  ["1-2", "환경/로컬 학생", "Gymnasium FANET, 1-hop action, GPSR baseline"],
  ["3-4", "전역 교사/KD", "전체 그래프 교사 정책을 지역 학생에게 증류"],
  ["7", "일반화 검증", "KD-only PDR 0.951, Teacher 대비 약 96%"],
  ["8", "Geo-Residual", "GPSR prior + residual correction, hole에서 강함"],
  ["9", "Risk-aware", "위험 feature 추가, 일부 불안정성 확인"],
  ["10", "외부 RL baseline", "Evo-QGeo, IQMR, DRAMA 적응 비교"],
  ["11-12", "Lite-GLOBE-P", "Risk-Switch로 예측 단절 약점 보완, Phase12 full 결과 확보"],
  ["13", "P+ 보완", "안정성, 중복성, keep probability, energy tie-break 추가"],
];

const slides = [];
const pendingSlideBuilds = [];

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function addText(slide, text, x, y, w, h, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: opts.size ?? 22,
    bold: opts.bold ?? false,
    color: opts.color ?? c.ink,
    italic: opts.italic ?? false,
  };
  return shape;
}

function addCard(slide, x, y, w, h, opts = {}) {
  return slide.shapes.add({
    geometry: "roundRect",
    position: { left: x, top: y, width: w, height: h },
    fill: opts.fill ?? c.card,
    line: { style: "solid", fill: opts.line ?? c.line, width: opts.lineWidth ?? 1 },
    borderRadius: opts.radius ?? "rounded-2xl",
    shadow: opts.shadow ?? "shadow-sm",
  });
}

function addHeader(slide, eyebrow, title, subtitle = "") {
  slide.background.fill = c.bg;
  const sectionLabel = eyebrow.includes("|") ? eyebrow.split("|").slice(1).join("|").trim() : eyebrow;
  const autoEyebrow = `${String(slides.length).padStart(2, "0")} | ${sectionLabel}`;
  addText(slide, autoEyebrow, page.left, 28, 420, 24, { size: 12, bold: true, color: c.muted });
  addText(slide, title, page.left, 56, 900, 58, { size: 31, bold: true, color: c.ink });
  if (subtitle) addText(slide, subtitle, page.left, 106, 980, 38, { size: 15, color: c.sub });
  const line = slide.shapes.add({
    geometry: "rect",
    position: { left: page.left, top: 148, width: page.width, height: 1.5 },
    fill: c.line,
    line: { style: "solid", fill: "none", width: 0 },
  });
  return line;
}

function addFooter(slide, n) {
  addText(slide, `GLOBE Routing 연구 보고 | ${n}`, page.left, 684, 220, 18, {
    size: 10,
    color: "slate-400",
  });
}

function addBullets(slide, items, x, y, w, h, opts = {}) {
  const text = items.map((item) => `• ${item}`).join("\n");
  return addText(slide, text, x, y, w, h, {
    size: opts.size ?? 18,
    color: opts.color ?? c.ink,
    bold: opts.bold ?? false,
  });
}

function addTag(slide, text, x, y, color = c.blue, width = 160) {
  addCard(slide, x, y, width, 32, { fill: color, line: color, radius: "rounded-full", shadow: "none" });
  addText(slide, text, x + 14, y + 7, width - 28, 18, { size: 12, bold: true, color: "white" });
}

function addMetricCard(slide, label, value, caption, x, y, w, color) {
  addCard(slide, x, y, w, 118, { fill: "white" });
  addText(slide, label, x + 22, y + 18, w - 44, 24, { size: 14, bold: true, color: c.muted });
  addText(slide, value, x + 22, y + 44, w - 44, 42, { size: 31, bold: true, color });
  addText(slide, caption, x + 22, y + 88, w - 44, 24, { size: 12, color: c.sub });
}

function addMiniTable(slide, rows, x, y, w, rowH, colWs, opts = {}) {
  rows.forEach((row, r) => {
    let left = x;
    const fill = r === 0 ? opts.headerFill ?? c.navy : r % 2 ? "white" : "slate-50";
    row.forEach((cell, ci) => {
      addCard(slide, left, y + r * rowH, colWs[ci], rowH, {
        fill,
        line: "slate-200",
        radius: 4,
        shadow: "none",
      });
      addText(slide, String(cell), left + 8, y + r * rowH + 8, colWs[ci] - 16, rowH - 12, {
        size: r === 0 ? 11 : opts.size ?? 12,
        bold: r === 0,
        color: r === 0 ? "white" : c.ink,
      });
      left += colWs[ci];
    });
  });
}

function addArrow(slide, x1, y1, x2, y2, label = "", color = c.blue) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.sqrt(dx * dx + dy * dy);
  const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
  const line = slide.shapes.add({
    geometry: "rect",
    position: { left: x1, top: y1, width: len, height: 3 },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
    rotation: angle,
  });
  const head = slide.shapes.add({
    geometry: "triangle",
    position: { left: x2 - 10, top: y2 - 8, width: 18, height: 16 },
    fill: color,
    line: { style: "solid", fill: color, width: 0 },
    rotation: angle + 90,
  });
  if (label) addText(slide, label, (x1 + x2) / 2 - 60, (y1 + y2) / 2 - 28, 140, 22, { size: 11, color });
  return [line, head];
}

function addNotes(slide, notes) {
  slide.speakerNotes.textFrame.setText(notes);
  slide.speakerNotes.setVisible(true);
}

function addSlide(presentation, title, builder, notes) {
  const slide = presentation.slides.add();
  const maybePromise = builder(slide);
  if (maybePromise && typeof maybePromise.then === "function") pendingSlideBuilds.push(maybePromise);
  addFooter(slide, slides.length + 1);
  addNotes(slide, notes);
  slides.push({ title, notes });
  return slide;
}

async function addImageIfExists(slide, relPath, x, y, w, h, alt) {
  const abs = path.join(REPO, relPath);
  try {
    const bytes = await fs.readFile(abs);
    slide.images.add({
      blob: bytes,
      contentType: "image/png",
      alt,
      fit: "contain",
      position: { left: x, top: y, width: w, height: h },
      geometry: "roundRect",
      borderRadius: "rounded-xl",
    });
    return true;
  } catch {
    return false;
  }
}

const presentation = Presentation.create({ slideSize: { width: W, height: H } });

addSlide(
  presentation,
  "Title",
  (slide) => {
    slide.background.fill = "slate-950";
    addText(slide, "교수님 보고용 연구 발표자료", 64, 48, 360, 26, { size: 13, bold: true, color: "slate-300" });
    addText(slide, "Risk-Switch Lite-GLOBE-P+\nUAV 군집 네트워크 라우팅 연구", 64, 128, 780, 150, {
      size: 39,
      bold: true,
      color: "white",
    });
    addText(
      slide,
      "전역 지식을 학습에만 사용하고, 실제 라우팅은 각 UAV가 1-hop 지역 정보만으로 수행하는 경량·위험회피형 라우팅 알고리즘",
      68,
      304,
      720,
      72,
      { size: 19, color: "slate-200" },
    );
    addTag(slide, "FANET Routing", 70, 420, c.blue, 150);
    addTag(slide, "Knowledge Distillation", 232, 420, c.purple, 210);
    addTag(slide, "Risk-Switch", 456, 420, c.orange, 150);
    addCard(slide, 850, 120, 330, 380, { fill: "slate-900", line: "slate-700" });
    addText(slide, "핵심 메시지", 884, 156, 220, 30, { size: 19, bold: true, color: "white" });
    addBullets(
      slide,
      [
        "GPSR의 routing-hole·링크 단절 약점을 보완",
        "DRAMA류 실행시 통신 부담과 차별화",
        "Phase12 full 결과에서 PDR·deadline 우수",
        "Phase13은 에너지·안정성 보완 방향",
      ],
      884,
      202,
      260,
      200,
      { size: 16, color: "slate-200" },
    );
    addText(slide, "2026-06-28", 70, 638, 180, 22, { size: 12, color: "slate-400" });
  },
  "첫 장에서는 연구를 한 문장으로 정리합니다. 이 연구는 UAV 군집망에서 전역 정보가 없어도 지역 노드가 안정적으로 다음 홉을 선택하도록 만드는 경량 라우팅 알고리즘입니다.",
);

addSlide(
  presentation,
  "Executive Summary",
  (slide) => {
    addHeader(slide, "01 | EXECUTIVE SUMMARY", "현재 연구의 결론을 먼저 말씀드리면", "핵심은 '전역 학습 + 지역 실행 + 위험 기반 스위칭'입니다.");
    addMetricCard(slide, "최종 실험 기준", "Phase12", "5 seeds, 14 scenarios, 84k rows", 70, 188, 260, c.blue);
    addMetricCard(slide, "전체 평균 PDR", "0.905", "GPSR 대비 +32.5%", 360, 188, 260, c.green);
    addMetricCard(slide, "Deadline 성공", "0.838", "GPSR 대비 +31.5%", 650, 188, 260, c.purple);
    addMetricCard(slide, "제어 바이트", "-22.7%", "DRAMA 대비 감소", 940, 188, 260, c.orange);
    addCard(slide, 76, 350, 520, 210);
    addText(slide, "논문에서 주장할 수 있는 방향", 104, 376, 440, 26, { size: 19, bold: true });
    addBullets(slide, [
      "단순 GPSR 개선이 아니라, 교사-학생 구조로 전역 라우팅 지식을 지역 정책에 이식",
      "실행 시에는 전체 그래프·중앙 명령·온라인 메시지 교환 없이 next-hop 결정",
      "링크가 곧 끊길 위험과 다음 홉 이후 경로 가능성을 함께 판단",
    ], 104, 420, 450, 108, { size: 16 });
    addCard(slide, 650, 350, 520, 210);
    addText(slide, "주의해서 말해야 할 부분", 678, 376, 420, 26, { size: 19, bold: true });
    addBullets(slide, [
      "Phase13은 smoke-test 완료, full Colab 검증은 아직 필요",
      "외부 baseline은 동일 환경 적응 구현이므로 원 논문 재현이라고 과장하면 안 됨",
      "에너지 지표는 DRAMA보다 약간 불리하여 보완 포인트로 제시",
    ], 678, 420, 440, 108, { size: 16 });
  },
  "교수님께는 결론부터 제시하는 것이 좋습니다. Phase12 기준으로 제안기법은 GPSR보다 확실히 개선되었고, 외부 baseline과 비교해도 평균 PDR과 deadline에서 경쟁력이 있습니다. 다만 Phase13은 아직 전체 검증 전이라는 점을 투명하게 말합니다.",
);

addSlide(
  presentation,
  "Problem",
  (slide) => {
    addHeader(slide, "02 | RESEARCH PROBLEM", "왜 UAV 군집 라우팅이 어려운가?", "노드가 빠르게 움직이고 링크가 자주 끊기기 때문에 단순 거리 기반 next-hop 선택이 흔들립니다.");
    const xs = [110, 280, 450, 620, 790, 960];
    xs.forEach((x, i) => {
      slide.shapes.add({
        geometry: "ellipse",
        position: { left: x, top: 260 + (i % 2) * 60, width: 52, height: 52 },
        fill: i === 5 ? c.green : c.blue2,
        line: { style: "solid", fill: "white", width: 2 },
      });
      addText(slide, i === 0 ? "S" : i === 5 ? "D" : `U${i}`, x + 14, 274 + (i % 2) * 60, 30, 22, { size: 14, bold: true, color: "white" });
    });
    for (let i = 0; i < xs.length - 1; i++) addArrow(slide, xs[i] + 55, 287 + (i % 2) * 60, xs[i + 1] - 4, 287 + ((i + 1) % 2) * 60, "", "slate-300");
    addCard(slide, 90, 450, 330, 120);
    addText(slide, "1. 부분 관측", 116, 472, 240, 24, { size: 18, bold: true, color: c.blue });
    addText(slide, "각 UAV는 주로 1-hop 이웃만 볼 수 있어 전체 경로 안정성을 직접 알기 어렵습니다.", 116, 506, 265, 44, { size: 14, color: c.sub });
    addCard(slide, 475, 450, 330, 120);
    addText(slide, "2. 빠른 링크 변화", 501, 472, 240, 24, { size: 18, bold: true, color: c.orange });
    addText(slide, "현재는 좋아 보이는 링크도 목적지까지 전달되기 전에 끊길 수 있습니다.", 501, 506, 265, 44, { size: 14, color: c.sub });
    addCard(slide, 860, 450, 330, 120);
    addText(slide, "3. Routing-hole", 886, 472, 240, 24, { size: 18, bold: true, color: c.red });
    addText(slide, "목적지에 가까운 이웃만 고르면 막다른 구간에 빠져 우회가 늦어집니다.", 886, 506, 265, 44, { size: 14, color: c.sub });
  },
  "이 슬라이드는 문제 정의입니다. UAV는 움직임 때문에 네트워크 토폴로지가 계속 바뀌고, 각 노드는 전체 네트워크가 아니라 주변 이웃만 압니다. 그래서 가장 가까운 노드로만 보내는 정책은 보기에는 단순하지만 routing-hole과 링크 단절에 취약합니다.",
);

addSlide(
  presentation,
  "Prior Art",
  (slide) => {
    addHeader(slide, "03 | PRIOR ART POSITIONING", "기존 기법과의 차별점", "우리 연구는 기존 기법을 부정하기보다, 각 기법의 강점과 빈틈 사이에 위치시킵니다.");
    const rows = [
      ["기법", "핵심 아이디어", "강점", "한계 / 우리 차별점"],
      ["GPSR", "목적지에 더 가까운 이웃 선택", "단순·저비용", "링크 수명·우회 가능성 부족"],
      ["Evo-QGeo", "미래 링크 상태와 Q-learning", "예측 단절에 강함", "우리도 예측 위험을 쓰지만 local policy로 경량화"],
      ["IQMR", "다중 목적 Q(lambda)", "큐·링크 등 목적 반영", "환경에 따라 안정성 약함"],
      ["DRAMA", "MARL + emergent communication", "협력 라우팅 강함", "실행시 메시지/제어 오버헤드가 큼"],
      ["우리 방법", "전역 교사 지식 -> 지역 학생 + 위험 스위치", "성능·오버헤드 균형", "Phase13 full 검증 필요"],
    ];
    addMiniTable(slide, rows, 72, 178, 1136, 58, [142, 290, 210, 494], { size: 13 });
    addText(slide, "논문 주장 포인트: '전역 정보를 계속 공유한다'가 아니라 '전역 정보를 학습 때만 사용하고, 실행 때는 지역 정보만으로 결정한다'입니다.", 98, 592, 1020, 36, { size: 18, bold: true, color: c.navy });
  },
  "기존 연구와 겹치지 않는 언어가 중요합니다. GPSR은 baseline, Evo-QGeo는 예측 링크 baseline, DRAMA는 실행 중 통신 baseline으로 두고, 우리 차별점은 학습 시 전역 지식과 실행 시 지역 정책의 분리라고 설명합니다.",
);

addSlide(
  presentation,
  "Novelty",
  (slide) => {
    addHeader(slide, "04 | DEFENSIBLE NOVELTY", "가장 방어 가능한 연구 기여", "문헌 비교 기준으로, 넓은 'global context' 주장은 약하고 아래 세 가지가 강합니다.");
    const cards = [
      ["1", "Training-time Global Teacher", "전체 그래프를 보는 교사가 경로 구조와 우회 가능성을 학습 데이터로 제공합니다.", c.blue],
      ["2", "Execution-time Local Student", "실제 UAV는 1-hop 이웃 feature만으로 next-hop을 선택합니다. 중앙 명령이 필요 없습니다.", c.green],
      ["3", "Risk-Switch Lite-GLOBE-P+", "현재 link만 보지 않고, 다음 홉 이후 경로 생존성·중복성·에너지까지 고려합니다.", c.orange],
    ];
    cards.forEach(([num, title, body, color], i) => {
      const x = 78 + i * 382;
      addCard(slide, x, 190, 330, 268);
      slide.shapes.add({ geometry: "ellipse", position: { left: x + 24, top: 220, width: 52, height: 52 }, fill: color, line: { style: "solid", fill: color, width: 0 } });
      addText(slide, num, x + 42, 234, 20, 20, { size: 18, bold: true, color: "white" });
      addText(slide, title, x + 24, 292, 260, 58, { size: 20, bold: true, color });
      addText(slide, body, x + 24, 360, 270, 72, { size: 15, color: c.sub });
    });
    addCard(slide, 122, 506, 1036, 72, { fill: "indigo-50", line: "indigo-100" });
    addText(slide, "교수님께 강조할 문장", 154, 524, 180, 22, { size: 14, bold: true, color: c.navy });
    addText(slide, "우리 기법은 '전역 정보를 매번 공유하는 RL 라우팅'이 아니라, 전역 경로 판단을 지역 실행 정책으로 증류한 통신 절약형 라우팅입니다.", 340, 522, 740, 30, { size: 18, bold: true, color: c.navy });
  },
  "이 장은 novelty 문장입니다. 기존 MARL이나 CTDE와 겹치지 않게, 전역 정보는 학습에만 쓰고 실제 라우팅은 지역 feature만 쓴다는 점을 분명히 해야 합니다.",
);

addSlide(
  presentation,
  "System Model",
  (slide) => {
    addHeader(slide, "05 | SYSTEM MODEL", "시뮬레이션 환경과 라우팅 문제 정의", "각 episode에서 source에서 destination까지 패킷을 hop-by-hop으로 전달합니다.");
    addCard(slide, 80, 182, 330, 330);
    addText(slide, "네트워크 모델", 110, 212, 230, 28, { size: 20, bold: true, color: c.blue });
    addBullets(slide, [
      "2D FANET / Random Waypoint",
      "통신 반경 기반 링크",
      "노드 이동으로 링크 상태 변화",
      "source-destination 단일 packet routing",
    ], 110, 264, 250, 150, { size: 16 });
    addCard(slide, 476, 182, 330, 330);
    addText(slide, "행동 공간", 506, 212, 230, 28, { size: 20, bold: true, color: c.green });
    addBullets(slide, [
      "1-hop 이웃 중 다음 홉 선택",
      "막힌 경우 DROP 선택 가능",
      "visited node 재방문 억제",
      "지역 실행 정책으로 deploy 가능",
    ], 506, 264, 250, 150, { size: 16 });
    addCard(slide, 872, 182, 330, 330);
    addText(slide, "성능 지표", 902, 212, 230, 28, { size: 20, bold: true, color: c.orange });
    addBullets(slide, [
      "PDR: packet delivery ratio",
      "Deadline success",
      "Delay p95",
      "Energy proxy",
      "Input/control bytes",
    ], 902, 264, 250, 170, { size: 16 });
    addText(slide, "목표: 높은 전달률과 deadline 성공률을 유지하면서, 실행시 정보 요구량과 제어 오버헤드를 낮추는 것", 106, 566, 1000, 30, { size: 18, bold: true, color: c.navy });
  },
  "환경 설명은 짧고 명확하게 갑니다. 이 연구는 통신 반경 기반 UAV 네트워크에서 next-hop을 고르는 문제이며, 성능은 전달률뿐 아니라 deadline, delay, energy, 입력 바이트까지 같이 봅니다.",
);

addSlide(
  presentation,
  "Method Evolution",
  (slide) => {
    addHeader(slide, "06 | METHOD EVOLUTION", "Phase 1부터 Phase 13까지의 연구 진화", "복잡해진 것이 아니라, 약점을 발견할 때마다 필요한 보완만 추가해 온 흐름입니다.");
    phaseTimeline.forEach(([phase, title, body], i) => {
      const x = 88 + (i % 4) * 292;
      const y = 178 + Math.floor(i / 4) * 180;
      addCard(slide, x, y, 248, 126, { fill: i >= 6 ? "emerald-50" : "white", line: i >= 6 ? "emerald-200" : c.line });
      addText(slide, `Phase ${phase}`, x + 18, y + 16, 110, 20, { size: 13, bold: true, color: i >= 6 ? "emerald-700" : c.blue });
      addText(slide, title, x + 18, y + 42, 200, 28, { size: 17, bold: true });
      addText(slide, body, x + 18, y + 76, 202, 34, { size: 12, color: c.sub });
      if (i % 4 !== 3) addArrow(slide, x + 252, y + 62, x + 282, y + 62, "", "slate-300");
    });
    addText(slide, "현재 논문 초점: Phase12 결과를 주 근거로 삼고, Phase13 P+는 최종 보완 모델 후보로 검증 예정", 118, 582, 960, 30, { size: 17, bold: true, color: c.navy });
  },
  "진화 과정을 한눈에 보여줍니다. Phase8은 구조적으로 강했고, Phase9는 위험 feature의 필요성을 보여줬지만 불안정했습니다. Phase11/12에서 risk-switch 구조로 정리했고, Phase13은 그 약점을 더 보완하는 단계입니다.",
);

addSlide(
  presentation,
  "Core Algorithm",
  (slide) => {
    addHeader(slide, "07 | CORE ALGORITHM", "제안기법의 전체 구조", "학습 단계와 실행 단계가 분리되어 있다는 점이 핵심입니다.");
    addCard(slide, 80, 190, 310, 250, { fill: "blue-50", line: "blue-100" });
    addText(slide, "전역 교사", 112, 220, 220, 32, { size: 23, bold: true, color: c.blue });
    addText(slide, "전체 UAV 그래프\n목적지 위치\n경로 안정성\n우회 가능성", 112, 276, 220, 120, { size: 18, color: c.navy });
    addCard(slide, 485, 190, 310, 250, { fill: "violet-50", line: "violet-100" });
    addText(slide, "지식 증류", 518, 220, 220, 32, { size: 23, bold: true, color: c.purple });
    addText(slide, "교사의 next-hop 선호도와\nrouting utility를\n학생 정책에 전달", 518, 276, 230, 100, { size: 18, color: c.navy });
    addCard(slide, 890, 190, 310, 250, { fill: "emerald-50", line: "emerald-100" });
    addText(slide, "지역 학생", 922, 220, 220, 32, { size: 23, bold: true, color: "emerald-700" });
    addText(slide, "1-hop 이웃 feature만 사용\n실행 시 추가 전역 통신 없음\n위험 기반 next-hop 선택", 922, 276, 235, 100, { size: 18, color: c.navy });
    addArrow(slide, 396, 315, 480, 315, "teacher logits", c.purple);
    addArrow(slide, 800, 315, 884, 315, "local policy", c.green);
    addCard(slide, 150, 506, 980, 70, { fill: "white" });
    addText(slide, "핵심 수식", 184, 522, 100, 22, { size: 15, bold: true, color: c.muted });
    addText(slide, "π_student(a | local obs, destination) ≈ π_teacher(a | global graph, destination)", 310, 518, 650, 30, { size: 24, bold: true, color: c.navy });
  },
  "이 장에서는 교사-학생 구조를 설명합니다. 교사는 전체 그래프를 보고 좋은 next-hop 분포를 만들고, 학생은 그 판단을 학습하지만 실제 deployment에서는 주변 이웃 정보만 사용합니다.",
);

addSlide(
  presentation,
  "Formal Problem",
  (slide) => {
    addHeader(slide, "08 | FORMAL PROBLEM", "제안기법의 문제 정의", "FANET 라우팅을 시간에 따라 변하는 그래프 위의 next-hop 선택 문제로 정의합니다.");
    addCard(slide, 76, 178, 530, 350);
    addText(slide, "동적 그래프 모델", 108, 208, 220, 26, { size: 20, bold: true, color: c.blue });
    addText(slide, "G_t = (V, E_t),    e_{uv}(t)=1[d(u,v,t) ≤ R_c]", 108, 258, 420, 34, { size: 24, bold: true, color: c.navy });
    addBullets(slide, [
      "V: UAV 노드 집합",
      "E_t: 시간 t에서 통신 가능한 링크",
      "R_c: 통신 반경",
      "d(u,v,t): 현재 노드 u와 후보 v 사이 거리",
    ], 108, 324, 410, 108, { size: 16 });
    addCard(slide, 674, 178, 530, 350);
    addText(slide, "라우팅 의사결정", 706, 208, 220, 26, { size: 20, bold: true, color: c.green });
    addText(slide, "a_t ∈ A(u_t) = N_t(u_t) ∪ {DROP}", 706, 258, 410, 34, { size: 24, bold: true, color: c.navy });
    addBullets(slide, [
      "u_t: 현재 패킷을 가진 UAV",
      "N_t(u_t): 현재 1-hop 이웃 후보",
      "a_t: 다음 홉 또는 DROP",
      "목표: 목적지 도달 확률을 높이고 deadline, delay, overhead를 동시에 관리",
    ], 706, 324, 410, 122, { size: 16 });
    addCard(slide, 134, 570, 1000, 46, { fill: "indigo-50", line: "indigo-100" });
    addText(slide, "핵심 제약: 실행 시 각 노드는 전체 G_t를 보지 못하고, 자신 주변의 local observation만 사용합니다.", 166, 584, 880, 22, { size: 17, bold: true, color: c.navy });
  },
  "문제 정의 슬라이드입니다. 그래프 G_t는 시간에 따라 링크가 바뀌고, 현재 패킷을 가진 노드 u_t는 1-hop 이웃 또는 DROP 중 하나를 선택합니다. 핵심 제약은 실행 시 전체 그래프를 보지 않는다는 점입니다.",
);

addSlide(
  presentation,
  "Learning Objective",
  (slide) => {
    addHeader(slide, "09 | LEARNING OBJECTIVE", "전역 교사 지식을 어떻게 지역 학생에게 옮기는가?", "학습 때는 전역 교사의 판단을 사용하지만, 배포 때는 학생 정책만 남깁니다.");
    addCard(slide, 74, 182, 540, 342);
    addText(slide, "교사-학생 증류 목적함수", 106, 212, 290, 26, { size: 20, bold: true, color: c.purple });
    addText(slide, "L_KD = T² · KL( softmax(zᵀ/T) || softmax(zˢ/T) )", 106, 262, 455, 36, { size: 21, bold: true, color: c.navy });
    addText(slide, "L = L_KD + α·L_CE + β·L_reg", 106, 328, 360, 32, { size: 23, bold: true, color: c.navy });
    addBullets(slide, [
      "zᵀ: 전역 교사의 candidate별 logit",
      "zˢ: 지역 학생의 candidate별 logit",
      "T: soft target을 부드럽게 만드는 temperature",
      "L_CE: 성공 경로/teacher action에 대한 보조 지도",
      "L_reg: 과도한 residual 또는 불안정한 정책 억제",
    ], 106, 390, 440, 106, { size: 14 });
    addCard(slide, 682, 182, 520, 342);
    addText(slide, "왜 이렇게 하는가?", 714, 212, 240, 26, { size: 20, bold: true, color: c.blue });
    addBullets(slide, [
      "전역 교사는 전체 경로 구조를 보므로 우회 가능성과 목적지 방향을 더 잘 판단",
      "학생은 teacher logit의 상대적 선호도를 배워 local observation만으로 유사한 결정을 수행",
      "단일 정답 action만 학습하는 것보다 후보 간 우열 정보를 더 많이 전달",
      "결과적으로 실행 시 centralized controller 없이 decentralized routing 가능",
    ], 714, 266, 405, 180, { size: 16 });
    addText(slide, "발표 포인트: 이 손실은 '전역 정보를 계속 쓰는 모델'이 아니라 '전역 판단을 지역 정책에 압축하는 학습 절차'입니다.", 138, 578, 980, 28, { size: 17, bold: true, color: c.navy });
  },
  "이 슬라이드는 학습 목적함수입니다. KL divergence로 교사의 후보별 선호 분포를 학생에게 전달합니다. 이것은 단순 imitation보다 풍부합니다. 후보 A가 가장 좋고 B가 두 번째라는 상대적 정보를 함께 배웁니다.",
);

addSlide(
  presentation,
  "Candidate Score",
  (slide) => {
    addHeader(slide, "10 | CANDIDATE SCORING", "다음 홉 후보 점수는 어떻게 계산하는가?", "Lite-GLOBE-P는 GPSR식 진행도, 예측 안정성, 학습 residual을 결합합니다.");
    addCard(slide, 72, 178, 548, 360);
    addText(slide, "Predictive Geographic Prior", 104, 208, 330, 26, { size: 20, bold: true, color: c.blue });
    addText(slide, "S_PG(j) = w_pP_j + w_fF_j + w_mM_j\n          + w_lL_j + w_qQ_j + w_oO_j", 104, 252, 450, 70, { size: 21, bold: true, color: c.navy });
    addMiniTable(slide, [
      ["항", "의미"],
      ["P_j", "목적지 방향 진행도"],
      ["F_j", "forwarding 가능성 / onward degree"],
      ["M_j", "현재 링크 margin"],
      ["L_j", "현재 링크 predicted lifetime"],
      ["Q_j", "queue headroom"],
      ["O_j", "best onward link lifetime"],
    ], 104, 346, 450, 25, [76, 374], { size: 11 });
    addCard(slide, 676, 178, 532, 360);
    addText(slide, "최종 candidate logit", 708, 208, 300, 26, { size: 20, bold: true, color: c.orange });
    addText(slide, "z_j = S_PG(j) + λ_r B·tanh(g_θ(o,j)) - λ_pG(j)", 708, 252, 440, 36, { size: 21, bold: true, color: c.navy });
    addText(slide, "G(j) = [τ_m-M_j]₊ + [τ_l-L_j]₊ + [τ_o-O_j]₊", 708, 314, 430, 30, { size: 19, bold: true, color: c.navy });
    addBullets(slide, [
      "g_θ(o,j): KD로 배운 residual correction",
      "B·tanh(·): residual 영향력 제한",
      "G(j): 링크 margin, 링크 수명, onward 수명이 부족할 때 부과되는 위험 penalty",
      "λ_r, λ_p: residual과 위험 penalty의 영향 조절",
    ], 708, 372, 410, 112, { size: 15 });
    addText(slide, "해석: 목적지에 가까운 후보라도 링크가 곧 끊기거나 다음 홉 이후 길이 약하면 점수가 낮아집니다.", 126, 582, 980, 26, { size: 18, bold: true, color: c.navy });
  },
  "후보 점수식 설명입니다. S_PG는 GPSR의 장점인 목적지 방향성을 살리면서, forwarding 가능성, 링크 margin, lifetime, queue, onward lifetime을 합친 prior입니다. 그 위에 학습 residual을 더하되 tanh로 영향력을 제한하고, 위험 gate penalty로 곧 끊길 링크를 강하게 낮춥니다.",
);

addSlide(
  presentation,
  "Risk Switch Details",
  (slide) => {
    addHeader(slide, "11 | RISK-SWITCH DETAILS", "Risk-Switch가 단순 heuristic이 아닌 이유", "위험 점수 D_i와 안전 이득 G(a_P,a_N)을 함께 사용해 불필요한 전환을 막습니다.");
    addCard(slide, 78, 178, 550, 360);
    addText(slide, "위험 점수", 110, 208, 160, 26, { size: 20, bold: true, color: c.red });
    addText(slide, "D_i = [g_m-m_i]₊ + [g_l-ℓ_i]₊ + [g_o-o_i]₊\n    + [g_k-ō_i,topk]₊ + 0.5[g_ρ-ρ_i]₊ + [g_p-p_keep]₊", 110, 252, 455, 70, { size: 19, bold: true, color: c.navy });
    addText(slide, "Q_i = m_i + ℓ_i + o_i + ō_i,topk + 0.5ρ_i + p_keep", 110, 350, 455, 30, { size: 19, bold: true, color: c.navy });
    addText(slide, "G(a_P,a_N) = Q(a_P) - Q(a_N)", 110, 404, 350, 28, { size: 20, bold: true, color: c.navy });
    addText(slide, "D_i는 위험 위반량, Q_i는 안전성 합산 점수입니다. 두 값을 같이 써서 '정말 더 안전할 때만' branch를 바꿉니다.", 110, 464, 440, 42, { size: 14, color: c.sub });
    addCard(slide, 690, 178, 500, 360);
    addText(slide, "Switch 조건", 722, 208, 180, 26, { size: 20, bold: true, color: c.orange });
    addText(slide, "S = 1[ a_N=DROP  or  D(a_N)>τ\n    or (a_P≠a_N ∧ G(a_P,a_N)>δ ∧ D(a_P)<D(a_N)) ]", 722, 252, 405, 72, { size: 18, bold: true, color: c.navy });
    addBullets(slide, [
      "a_N: normal branch가 고른 next-hop",
      "a_P: predictive branch가 고른 next-hop",
      "τ: normal 후보가 위험하다고 판단하는 기준",
      "δ: predictive 후보로 바꿀 만큼 충분한 안전 이득",
      "S=1이면 a_P를 사용, 아니면 a_N 유지",
    ], 722, 356, 385, 120, { size: 15 });
  },
  "Risk-Switch의 상세 슬라이드입니다. 핵심은 predictive branch가 다르다고 무조건 바꾸는 것이 아니라, normal 후보가 위험하거나 predictive 후보가 충분히 더 안전할 때만 바꾸는 것입니다.",
);

addSlide(
  presentation,
  "Algorithm Flow",
  (slide) => {
    addHeader(slide, "12 | ALGORITHM FLOW", "제안 알고리즘의 실행 흐름", "실제 라우팅 시에는 아래 5단계를 매 hop 반복합니다.");
    const steps = [
      ["1", "후보 생성", "현재 노드 u의 1-hop 이웃 N(u)와 DROP을 action 후보로 둡니다.", c.blue],
      ["2", "지역 feature 계산", "각 후보 i에 대해 진행도, 링크 여유, 링크 수명, 큐 여유, onward 안정성을 계산합니다.", c.purple],
      ["3", "Normal 후보 선택", "Geo-Residual / local student branch가 기본 next-hop a_N을 선택합니다.", c.green],
      ["4", "Predictive 후보 선택", "위험 penalty, keep 확률, energy tie-break를 반영해 a_P를 선택합니다.", c.orange],
      ["5", "Risk-Switch 결정", "a_N이 위험하거나 DROP이면, 충분히 안전 이득이 있는 경우에만 a_P로 전환합니다.", c.red],
    ];
    steps.forEach(([num, title, body, color], i) => {
      const x = 84 + (i % 3) * 382;
      const y = i < 3 ? 186 : 400;
      const w = i < 3 ? 330 : 505;
      const xx = i < 3 ? x : 180 + (i - 3) * 560;
      addCard(slide, xx, y, w, 148, { fill: "white", line: "slate-200" });
      slide.shapes.add({ geometry: "ellipse", position: { left: xx + 22, top: y + 22, width: 42, height: 42 }, fill: color, line: { style: "solid", fill: color, width: 0 } });
      addText(slide, num, xx + 38, y + 32, 16, 18, { size: 14, bold: true, color: "white" });
      addText(slide, title, xx + 78, y + 22, w - 110, 26, { size: 20, bold: true, color });
      addText(slide, body, xx + 78, y + 62, w - 118, 48, { size: 14, color: c.sub });
    });
    addText(slide, "핵심: 매 hop에서 후보를 전부 다시 평가하므로, UAV 이동으로 네트워크가 바뀌어도 local observation 기반으로 빠르게 대응할 수 있습니다.", 128, 594, 1010, 30, { size: 18, bold: true, color: c.navy });
  },
  "이 슬라이드는 pseudo-code보다 더 발표 친화적인 알고리즘 흐름입니다. 후보 생성, feature 계산, normal 후보, predictive 후보, switch 결정의 5단계로 설명하면 교수님이 전체 구조를 빠르게 이해할 수 있습니다.",
);

addSlide(
  presentation,
  "Executable Algorithm",
  (slide) => {
    addHeader(slide, "12 | EXECUTABLE ALGORITHM", "실제 실행 알고리즘", "각 hop마다 모든 후보 이웃을 평가하고, 최종 next-hop 하나를 선택합니다.");
    addCard(slide, 72, 176, 650, 388);
    addText(slide, "Algorithm: Risk-Switch Lite-GLOBE-P+ Routing", 104, 204, 440, 24, { size: 18, bold: true, color: c.blue });
    addText(slide, [
      "Input: current node u, destination d, 1-hop neighbors N(u)",
      "Output: next-hop action a*",
      "",
      "1. For each candidate i in N(u):",
      "   compute P_i, F_i, M_i, L_i, Q_i, O_i",
      "   compute plus features: ō_i,topk, ρ_i, p_keep, η_i",
      "   compute S_PG(i), G(i), D_i, Q_i^safe",
      "",
      "2. Normal branch:",
      "   a_N = argmax_i z_i^N",
      "",
      "3. Predictive branch:",
      "   z_i^{P+} = z_i^P + λ_Eη_i",
      "   suppress DROP if safe forward candidate exists",
      "   a_P = argmax_i z_i^{P+}",
      "",
      "4. Switch decision:",
      "   if a_N=DROP or D(a_N)>τ or safe_gain(a_P,a_N)>δ:",
      "      a* = a_P",
      "   else:",
      "      a* = a_N",
    ].join("\n"), 104, 244, 560, 290, { size: 11, color: c.navy });
    addCard(slide, 780, 176, 420, 388);
    addText(slide, "계산 복잡도와 배포성", 812, 204, 260, 24, { size: 20, bold: true, color: c.green });
    addBullets(slide, [
      "후보 수를 k=|N(u)|라 하면 후보별 feature 계산은 O(k)",
      "top-k onward 요약은 local neighbor summary로 계산",
      "실행 시 global graph, centralized controller, online message passing이 필요 없음",
      "DRAMA류와 달리 후보 간 hidden communication을 유지하지 않음",
      "따라서 UAV onboard routing rule로 설명 가능",
    ], 812, 260, 320, 154, { size: 15 });
    addCard(slide, 808, 466, 330, 54, { fill: "emerald-50", line: "emerald-100" });
    addText(slide, "요약: 성능 향상을 위해 복잡한 MARL 통신을 추가한 것이 아니라, 지역 후보 평가식을 더 똑똑하게 만든 방식입니다.", 828, 478, 290, 28, { size: 13, bold: true, color: "emerald-800" });
  },
  "실제 실행 알고리즘입니다. 각 hop마다 후보 이웃의 진행도, 링크 안정성, 큐, onward 안정성, 에너지 효율을 계산하고 normal branch와 predictive branch 중 안전한 선택을 고릅니다. 복잡도는 후보 수에 선형이며 실행 시 전역 그래프가 필요 없습니다.",
);

addSlide(
  presentation,
  "Variable Dictionary",
  (slide) => {
    addHeader(slide, "13 | VARIABLE DICTIONARY", "수식 변수 한 장 요약", "교수님 질문에 바로 답할 수 있도록 주요 변수를 의미 중심으로 정리합니다.");
    const rows = [
      ["변수", "의미", "직관"],
      ["P_j", "목적지 방향 진행도", "클수록 GPSR처럼 목적지에 가까워짐"],
      ["F_j", "forwardability / onward degree", "다음 홉 이후에도 보낼 후보가 많음"],
      ["M_j, m_i", "현재 링크 margin", "통신 반경 안쪽에 여유가 큼"],
      ["L_j, ℓ_i", "현재 링크 predicted lifetime", "현재 링크가 오래 유지됨"],
      ["Q_j, q_i", "queue headroom", "후보 노드가 덜 혼잡함"],
      ["O_j, o_i", "best onward lifetime", "다음 홉 이후 가장 좋은 링크가 오래 감"],
      ["ō_i,topk", "상위 k개 onward lifetime 평균", "경로가 하나에만 의존하지 않음"],
      ["ρ_i", "onward 후보 중복성", "우회 선택지가 많음"],
      ["p_keep", "링크 유지 확률", "stochastic loss에서도 살아남을 가능성"],
      ["η_i", "에너지 효율 proxy", "짧은 링크일수록 송신 부담이 작음"],
    ];
    addMiniTable(slide, rows, 70, 170, 1140, 43, [120, 350, 670], { size: 12 });
    addText(slide, "중요한 해석: 이 변수들은 대부분 현재 노드와 1-hop 후보, 그리고 후보가 제공하는 onward 요약에서 계산되므로 local execution 조건을 유지합니다.", 116, 650, 1000, 28, { size: 16, bold: true, color: c.navy });
  },
  "이 장은 질문 대비용입니다. 수식에 등장하는 변수의 의미를 한 번에 보여줍니다. 특히 각 값이 높을수록 어떤 의미인지 직관적으로 설명하면 교수님이 알고리즘을 빠르게 이해할 수 있습니다.",
);

addSlide(
  presentation,
  "Mathematical Features",
  (slide) => {
    addHeader(slide, "08 | RISK FEATURES", "위험을 수식으로 어떻게 판단하는가?", "단순히 가까운 노드를 고르지 않고, 링크가 버틸지와 다음 홉 이후 길이 있는지를 수치화합니다.");
    addCard(slide, 82, 188, 500, 320);
    addText(slide, "기본 위험 feature", 112, 218, 250, 26, { size: 20, bold: true, color: c.blue });
    addText(slide, "xᵢʳⁱˢᵏ = [mᵢ, ℓᵢ, qᵢ, oᵢ]", 112, 266, 380, 36, { size: 27, bold: true, color: c.navy });
    addBullets(slide, [
      "mᵢ: 통신 반경 대비 링크 여유도",
      "ℓᵢ: 예측된 현재 링크 생존 시간",
      "qᵢ: 큐 여유도",
      "oᵢ: 다음 홉 이후 가장 좋은 onward link 수명",
    ], 112, 330, 410, 118, { size: 16 });
    addCard(slide, 650, 188, 500, 320);
    addText(slide, "P+ 추가 feature", 680, 218, 250, 26, { size: 20, bold: true, color: c.orange });
    addText(slide, "xᵢᵖˡᵘˢ = [ōᵢ,topk, ρᵢ, p_keep, ηᵢ]", 680, 266, 420, 36, { size: 27, bold: true, color: c.navy });
    addBullets(slide, [
      "ōᵢ,topk: 상위 k개 onward link 평균 안정성",
      "ρᵢ: 다음 단계에서 선택 가능한 후보 중복성",
      "p_keep: 현재 링크가 유지될 확률",
      "ηᵢ = 1 - (dᵢ/Rc)²: 에너지 효율 proxy",
    ], 680, 330, 410, 118, { size: 16 });
    addText(slide, "쉽게 말하면: '이웃이 목적지에 가까운가?'뿐 아니라 '그 이웃을 거쳐도 길이 계속 살아 있는가?'를 묻습니다.", 130, 574, 980, 30, { size: 19, bold: true, color: c.navy });
  },
  "수식은 어렵게 보이지 않게 변수 의미를 같이 풀어줍니다. m은 링크 여유, l은 링크 수명, o는 다음 홉 이후의 길, rho는 후보 경로의 중복성입니다.",
);

addSlide(
  presentation,
  "Switch Rule",
  (slide) => {
    addHeader(slide, "09 | RISK-SWITCH RULE", "언제 기존 선택을 바꾸는가?", "좋아 보이는 노드라도 위험하면 predictive candidate로 바꿉니다. 단, 무조건 자주 바꾸지는 않습니다.");
    addCard(slide, 80, 190, 525, 320);
    addText(slide, "위험 점수", 112, 220, 160, 24, { size: 20, bold: true, color: c.red });
    addText(slide, "Dᵢ = [gₘ-mᵢ]₊ + [gℓ-ℓᵢ]₊ + [gₒ-oᵢ]₊\n     + [gₜ-ōᵢ,topk]₊ + 0.5[gρ-ρᵢ]₊ + [gp-p_keep]₊", 112, 266, 440, 74, { size: 20, bold: true, color: c.navy });
    addText(slide, "Dᵢ가 클수록 위험합니다. 즉 링크 여유, 링크 수명, onward 안정성, 후보 중복성, keep 확률이 기준보다 낮으면 벌점을 받습니다.", 112, 374, 424, 72, { size: 15, color: c.sub });
    addCard(slide, 675, 190, 525, 320);
    addText(slide, "스위치 조건", 707, 220, 180, 24, { size: 20, bold: true, color: c.orange });
    addText(slide, "S = 1[ aₙ = DROP  or  D(aₙ)>τ\n        or  (aₚ≠aₙ and G(aₚ,aₙ)>δ) ]", 707, 266, 430, 74, { size: 20, bold: true, color: c.navy });
    addBullets(slide, [
      "aₙ: 현재 네트워크 정책의 후보",
      "aₚ: predictive/risk-aware 후보",
      "G: 후보 간 안전성 이득",
      "τ, δ: 너무 잦은 전환을 막는 기준값",
    ], 707, 370, 410, 108, { size: 16 });
    addText(slide, "교수님께 설명할 요지: '위험할 때만 바꾸는 구조'라서 성능 개선과 안정성을 동시에 노립니다.", 126, 580, 980, 30, { size: 18, bold: true, color: c.navy });
  },
  "Risk-switch는 핵심입니다. 현재 후보가 DROP이거나 위험 점수가 높거나, 예측 후보가 충분히 더 안전할 때만 바꿉니다. 이렇게 해야 무조건 복잡한 정책보다 안정적입니다.",
);

addSlide(
  presentation,
  "P+ Algorithm",
  (slide) => {
    addHeader(slide, "10 | PHASE13 P+ DESIGN", "현재 보완 중인 최종 알고리즘 후보", "Phase12의 약점인 energy와 predictive-break+loss를 줄이기 위한 보완입니다.");
    addCard(slide, 80, 188, 520, 340);
    addText(slide, "P+에서 추가된 장치", 112, 218, 250, 26, { size: 20, bold: true, color: c.green });
    addBullets(slide, [
      "Top-k onward stability: 다음 홉 이후 경로가 하나뿐인지 여러 개인지 확인",
      "Onward redundancy: 우회 후보가 많을수록 안전한 후보로 판단",
      "Link keep probability: 곧 끊길 링크를 사전에 회피",
      "Energy tie-break: 비슷한 후보라면 더 짧고 효율적인 링크 선택",
      "Drop suppression: 안전 후보가 있으면 DROP을 쉽게 선택하지 않음",
    ], 112, 266, 420, 180, { size: 15 });
    addCard(slide, 680, 188, 500, 340);
    addText(slide, "간단한 의사코드", 712, 218, 250, 26, { size: 20, bold: true, color: c.blue });
    addText(slide, [
      "for each candidate neighbor i:",
      "  compute progress, margin, lifetime",
      "  compute onward stability and redundancy",
      "  compute danger D_i and energy η_i",
      "",
      "choose normal action a_n",
      "choose predictive action a_p",
      "if a_n is risky and a_p is safer:",
      "  route through a_p",
      "else:",
      "  keep a_n",
    ].join("\n"), 712, 264, 390, 216, { size: 15, color: c.navy });
  },
  "Phase13은 교수님께 '최종 후보'라고 말해야 합니다. 아직 full run 전이지만, 설계 방향은 명확합니다. 예측 단절을 피하면서도 불필요한 switch와 energy 낭비를 줄이는 것입니다.",
);

addSlide(
  presentation,
  "Phase12 Results Chart",
  (slide) => {
    addHeader(slide, "11 | PHASE12 FULL RESULT", "전체 평균에서 제안기법은 어느 정도 좋은가?", "Phase12 full 결과 기준, Risk-Switch는 평균 PDR과 deadline에서 가장 높은 그룹입니다.");
    addCard(slide, 66, 178, 550, 360);
    slide.charts.add("bar", {
      position: { left: 100, top: 218, width: 490, height: 270 },
      categories: phase12Overall.map((r) => r[0]),
      series: [
        { name: "PDR (%)", values: phase12Overall.map((r) => Number((r[1] * 100).toFixed(1))), fill: c.blue },
        { name: "Deadline (%)", values: phase12Overall.map((r) => Number((r[2] * 100).toFixed(1))), fill: c.green },
      ],
      hasLegend: true,
      legend: { position: "bottom" },
      barOptions: { direction: "column", grouping: "clustered", gapWidth: 55 },
      yAxis: { minimumScale: 0, maximumScale: 100, numberFormatCode: "0", majorGridlines: { style: "solid", fill: "slate-200", width: 1 } },
      dataLabels: { showValue: true, position: "outEnd" },
    });
    addCard(slide, 665, 178, 550, 360);
    slide.charts.add("bar", {
      position: { left: 700, top: 218, width: 480, height: 270 },
      categories: ["vs GPSR", "vs Predictive", "vs Evo-QGeo", "vs DRAMA"],
      series: [
        { name: "PDR 개선", values: [32.5, 1.5, 2.1, 1.6], fill: c.orange },
      ],
      hasLegend: false,
      barOptions: { direction: "column", grouping: "clustered", gapWidth: 65 },
      yAxis: { minimumScale: 0, maximumScale: 35, majorGridlines: { style: "solid", fill: "slate-200", width: 1 } },
      dataLabels: { showValue: true, position: "outEnd" },
    });
    addText(slide, "Risk-Switch: PDR 0.905, Deadline 0.838 | GPSR 대비 PDR +32.5%, Deadline +31.5%", 120, 578, 960, 30, { size: 18, bold: true, color: c.navy });
  },
  "수치 설명은 간결하게 합니다. Phase12에서 Risk-Switch는 GPSR보다 확실히 좋고, Predictive Geographic/Evo-QGeo/DRAMA와 비교해도 평균 PDR이 근소하게 더 높습니다.",
);

addSlide(
  presentation,
  "Metric Table",
  (slide) => {
    addHeader(slide, "12 | METRIC-LEVEL INTERPRETATION", "성능지표별 특징", "PDR만 보지 않고 delay, energy, overhead까지 같이 봅니다.");
    const rows = [
      ["Method", "PDR", "Deadline", "Delay p95", "Energy", "Bytes"],
      ...phase12Overall.map(([m, p, d, delay, e, b]) => [m, p.toFixed(3), d.toFixed(3), delay.toFixed(3), e.toFixed(3), String(b)]),
    ];
    addMiniTable(slide, rows, 80, 180, 1120, 54, [190, 160, 160, 190, 190, 230], { size: 15 });
    addCard(slide, 116, 572, 1010, 66, { fill: "amber-50", line: "amber-100" });
    addText(slide, "해석: Risk-Switch는 reliability(PDR/deadline)와 overhead(bytes)의 균형이 좋지만,\nenergy proxy는 DRAMA보다 약간 높아 Phase13에서 보완 중입니다.", 144, 588, 930, 38, { size: 15, bold: true, color: "amber-900" });
  },
  "표에서는 강점과 약점을 같이 말합니다. Risk-Switch는 PDR과 deadline이 좋고 DRAMA보다 제어 바이트가 낮지만, energy는 DRAMA보다 약간 좋지 않으므로 P+에서 개선 중이라고 연결합니다.",
);

addSlide(
  presentation,
  "Scenario Findings",
  (slide) => {
    addHeader(slide, "13 | SCENARIO FINDINGS", "어떤 상황에서 강하고, 어디가 아직 어려운가?", "평균뿐 아니라 상황별 결과를 봐야 논문 방어력이 생깁니다.");
    const rows = [["Scenario", "Risk-Switch PDR", "의미", "비교 포인트"], ...scenarioSummary];
    addMiniTable(slide, rows, 74, 180, 1132, 66, [190, 180, 310, 452], { size: 15 });
    addCard(slide, 126, 540, 460, 78, { fill: "emerald-50", line: "emerald-100" });
    addText(slide, "강점", 154, 558, 80, 22, { size: 15, bold: true, color: "emerald-700" });
    addText(slide, "routing-hole과 일반/확장 환경에서는 안정적으로 강함", 238, 556, 300, 24, { size: 16, bold: true, color: c.navy });
    addCard(slide, 680, 540, 460, 78, { fill: "rose-50", line: "rose-100" });
    addText(slide, "주의", 708, 558, 80, 22, { size: 15, bold: true, color: "rose-700" });
    addText(slide, "predictive-break 일부 조건에서는 Evo-QGeo가 아직 더 강함", 792, 556, 310, 24, { size: 16, bold: true, color: c.navy });
  },
  "교수님이 물을 가능성이 높은 부분입니다. 전체 평균만 좋다고 하지 말고, routing-hole에서는 강하고 predictive-break 일부 조건에서는 아직 Evo-QGeo가 강한 케이스가 있다고 말해야 합니다. 이 약점이 Phase13의 동기입니다.",
);

addSlide(
  presentation,
  "External Baselines",
  (slide) => {
    addHeader(slide, "14 | BASELINE ROLE", "외부 baseline 3개를 왜 넣었는가?", "논문 설득력을 위해 GPSR만이 아니라 최신 RL 계열과도 비교했습니다.");
    const items = [
      ["Evo-QGeo", "미래 링크 상태 + Q-learning", "예측 단절에 강한 기준점"],
      ["IQMR Q(lambda)", "링크·큐·경로 품질 다중 목적", "전통 RL 라우팅과 비교"],
      ["DRAMA", "MARL + emergent communication", "협력/통신 기반 라우팅과 비교"],
    ];
    items.forEach(([name, idea, why], i) => {
      const x = 110 + i * 360;
      addCard(slide, x, 190, 300, 252);
      addText(slide, name, x + 24, 222, 210, 30, { size: 22, bold: true, color: [c.orange, c.purple, c.blue][i] });
      addText(slide, idea, x + 24, 284, 230, 56, { size: 17, bold: true, color: c.navy });
      addText(slide, why, x + 24, 360, 230, 48, { size: 15, color: c.sub });
    });
    addCard(slide, 128, 508, 1010, 70, { fill: "slate-100", line: "slate-200" });
    addText(slide, "단, 이 baseline들은 동일 시뮬레이션 환경에 맞춘 적응 구현입니다. 원 논문 전체 재현이라고 표현하면 안 되고, fair comparison을 위해 공통 환경에 이식한 비교 기법이라고 설명해야 합니다.", 160, 524, 930, 32, { size: 16, color: c.ink });
  },
  "외부 baseline의 역할을 설명합니다. GPSR만 이기면 약하므로 Evo-QGeo, IQMR, DRAMA를 넣었습니다. 다만 원 논문을 완전 재현했다고 과장하지 않는 것이 중요합니다.",
);

addSlide(
  presentation,
  "Evidence Figures",
  async (slide) => {
    addHeader(slide, "15 | VISUAL EVIDENCE", "논문 그림으로 이어질 수 있는 결과", "기존 생성 그래프를 발표자료에 넣어 연구 진행 상황을 시각적으로 보여줍니다.");
    await addImageIfExists(slide, "ResearchAIWorkspace/vault/06_Experiments/Lite-GLOBE/Figures/Phase12/risk_switch_pdr.png", 70, 184, 700, 356, "Phase12 PDR chart");
    await addImageIfExists(slide, "ResearchAIWorkspace/vault/06_Experiments/Lite-GLOBE/Figures/Phase12/risk_switch_delay_p95.png", 824, 184, 350, 166, "Phase12 delay p95 chart");
    await addImageIfExists(slide, "ResearchAIWorkspace/vault/06_Experiments/Lite-GLOBE/Figures/Phase12/risk_switch_input_bytes.png", 824, 374, 350, 166, "Phase12 input bytes chart");
    addText(slide, "발표에서는 PDR 그림을 중심으로 설명하고, 논문 본문에서는 delay / overhead / scenario별 ablation을 별도 Figure로 분리하는 구성이 좋습니다.", 118, 574, 980, 34, { size: 17, bold: true, color: c.navy });
  },
  "이 슬라이드는 기존 결과 그래프를 그대로 보여주는 장입니다. 말로만 설명하는 것보다 Phase12 결과가 이미 시각화되어 있다는 점을 보여줄 수 있습니다.",
);

addSlide(
  presentation,
  "Claim Boundaries",
  (slide) => {
    addHeader(slide, "16 | CLAIM BOUNDARIES", "논문에서 어디까지 주장할 수 있는가?", "좋은 논문은 강점뿐 아니라 주장 범위를 정확히 제한합니다.");
    addCard(slide, 92, 190, 500, 310, { fill: "emerald-50", line: "emerald-100" });
    addText(slide, "강하게 주장 가능", 124, 220, 260, 28, { size: 21, bold: true, color: "emerald-700" });
    addBullets(slide, [
      "GPSR 대비 routing-hole와 link-break 상황에서 큰 개선",
      "DRAMA 대비 낮은 control/input byte로 유사 또는 더 높은 reliability",
      "전역 교사 지식을 지역 실행 정책으로 증류하는 구조",
      "일반 환경 성능을 크게 희생하지 않고 위험 상황 보완",
    ], 124, 274, 390, 142, { size: 16 });
    addCard(slide, 688, 190, 500, 310, { fill: "rose-50", line: "rose-100" });
    addText(slide, "조심해야 할 주장", 720, 220, 260, 28, { size: 21, bold: true, color: "rose-700" });
    addBullets(slide, [
      "모든 조건에서 항상 최고라고 말하면 위험",
      "외부 baseline 원 논문 재현이라고 과장 금지",
      "실제 UAV 장비·MAC contention·beacon overhead 검증은 아직 부족",
      "Phase13 P+는 full run 후 최종 claim 확정",
    ], 720, 274, 390, 142, { size: 16 });
  },
  "이 장은 교수님께 신뢰를 주는 장입니다. 강한 주장은 명확히 하되, 실제 장비나 MAC 계층까지 검증한 것은 아니라고 선을 긋습니다.",
);

addSlide(
  presentation,
  "Next Plan",
  (slide) => {
    addHeader(slide, "17 | NEXT VALIDATION PLAN", "교수님께 제안드릴 다음 실험 계획", "Phase13 P+가 최종 제안기법이 되려면 아래 검증이 필요합니다.");
    const steps = [
      ["1", "Phase13 full run", "5 seeds 이상, Phase12와 동일 scenario로 비교"],
      ["2", "Ablation study", "no top-k, no energy tie, no drop suppression 등 제거 실험"],
      ["3", "Overhead analysis", "DRAMA 대비 control/input byte per delivered packet 명확화"],
      ["4", "Stress test", "link-loss, high mobility, 16/24 UAV, communication range 변화"],
      ["5", "Manuscript alignment", "수식·알고리즘·그림·baseline 설명을 논문 본문으로 정리"],
    ];
    steps.forEach(([n, title, body], i) => {
      const y = 180 + i * 74;
      slide.shapes.add({ geometry: "ellipse", position: { left: 102, top: y, width: 42, height: 42 }, fill: c.blue, line: { style: "solid", fill: c.blue, width: 0 } });
      addText(slide, n, 118, y + 10, 16, 18, { size: 14, bold: true, color: "white" });
      addText(slide, title, 172, y + 2, 260, 22, { size: 19, bold: true, color: c.navy });
      addText(slide, body, 172, y + 31, 820, 22, { size: 15, color: c.sub });
    });
    addCard(slide, 830, 186, 330, 230, { fill: "indigo-50", line: "indigo-100" });
    addText(slide, "최종 판단 기준", 860, 216, 190, 24, { size: 19, bold: true, color: c.navy });
    addBullets(slide, [
      "PDR/deadline: 최고 또는 동률권",
      "Delay: DRAMA보다 낮거나 유사",
      "Bytes: DRAMA/Evo보다 낮음",
      "Energy: Phase12보다 개선",
    ], 860, 264, 250, 108, { size: 15 });
  },
  "다음 계획은 실험의 우선순위를 제시합니다. Phase13 full run, ablation, overhead, stress test, manuscript 반영 순서로 가면 교수님께도 연구가 체계적으로 진행 중이라는 인상을 줄 수 있습니다.",
);

addSlide(
  presentation,
  "Closing",
  (slide) => {
    slide.background.fill = "slate-950";
    addText(slide, "최종 요약", 74, 60, 200, 28, { size: 14, bold: true, color: "slate-300" });
    addText(slide, "Lite-GLOBE-P+의 논문 메시지", 74, 112, 720, 50, { size: 34, bold: true, color: "white" });
    addCard(slide, 90, 220, 1080, 270, { fill: "slate-900", line: "slate-700" });
    addText(slide, "UAV 라우팅에서 중요한 것은 단순히 목적지에 가까운 노드를 고르는 것이 아니라,\n'그 링크가 얼마나 버틸지'와 '그 다음에도 경로가 살아 있을지'를 지역 정보만으로 판단하는 것입니다.\n\n우리 연구는 전역 교사의 판단을 지역 학생에게 증류하고, 위험 상황에서만 predictive candidate로 전환함으로써\n전달률·deadline·제어 오버헤드의 균형을 개선하는 방향으로 정리됩니다.", 130, 252, 1000, 175, { size: 22, bold: true, color: "white" });
    addText(slide, "다음 단계: Phase13 full 검증 후 최종 제안기법과 논문 그림/표 확정", 130, 550, 880, 30, { size: 20, bold: true, color: "slate-200" });
  },
  "마지막은 한 문장 메시지로 마무리합니다. 교수님께는 이 연구가 단순 성능 튜닝이 아니라, 지역 실행 가능한 위험회피 라우팅 알고리즘으로 정리되고 있다고 보고하면 됩니다.",
);

await fs.mkdir(OUT_DIR, { recursive: true });
await fs.mkdir(RENDER_DIR, { recursive: true });
await fs.mkdir(VAULT_DIR, { recursive: true });

await Promise.all(pendingSlideBuilds);

for (const [index, slide] of presentation.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  await writeBlob(path.join(RENDER_DIR, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
  await fs.writeFile(path.join(RENDER_DIR, `${stem}.layout.json`), await (await slide.export({ format: "layout" })).text());
}

await writeBlob(path.join(RENDER_DIR, "deck-montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 1 }));
const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(PPTX_PATH);

const notesMd = [
  "# Risk-Switch Lite-GLOBE-P+ 교수님 보고 발표자료",
  "",
  `- PPTX: ${PPTX_PATH}`,
  `- Render preview: ${path.join(RENDER_DIR, "deck-montage.webp")}`,
  `- 생성일: 2026-06-28`,
  "",
  "## 발표 핵심 메시지",
  "",
  "이 연구는 UAV/FANET 라우팅에서 전역 그래프를 계속 공유하는 방식이 아니라, 학습 단계에서만 전역 교사를 사용하고 실제 실행 단계에서는 각 UAV가 1-hop 지역 정보만으로 next-hop을 선택하도록 만드는 경량 위험회피 라우팅 기법이다.",
  "",
  "## 슬라이드별 발표 노트",
  "",
  ...slides.flatMap((s, i) => [`### ${i + 1}. ${s.title}`, "", s.notes, ""]),
].join("\n");

await fs.writeFile(NOTES_PATH, notesMd);
await fs.writeFile(OBSIDIAN_PATH, notesMd);

console.log(JSON.stringify({ pptx: PPTX_PATH, notes: NOTES_PATH, obsidian: OBSIDIAN_PATH, render: RENDER_DIR }, null, 2));
