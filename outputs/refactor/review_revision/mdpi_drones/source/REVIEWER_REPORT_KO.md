# DiSwitch 예상 심사 의견 및 수정 대응 기록

검토일: 2026-09-07. 대상: outputs/refactor의 IEEE Access와 MDPI Drones 원고. 이 의견은 실제 심사 결과가 아닌, 원고·기존 감사 기록·일부 구현을 검토한 예상 의견이다. 이번 작업은 원고 수정이며 새로운 학습, Colab 실행, 통계 재계산을 수행하지 않았다. 따라서 문장으로 해결한 사항과 추가 증거가 필요한 사항을 구분한다.

## 이번 원고의 표시와 사용 방법

- 검정: 이번에 수정한 본문 및 유지한 내용.
- 빨강: 향후 삭제하거나 보충자료로 이동할 후보. D01~D05 식별자를 PDF와 TeX에 같이 사용한다.
- 빨간 테두리: 그림 자체의 삭제/이동 후보이며 그림 내부의 데이터 값은 바꾸지 않았다.
- 파랑: 이메일·기여·연구비·데이터 공개·학습 설정처럼 확인·보완할 항목. 삭제를 권하는 표시가 아니다.
- Overleaf에서는 각 ZIP의 `main.tex`를 Main document, pdfLaTeX를 compiler로 선택한다. `\reviewmarksfalse`는 후보를 숨기지만, 최종 제출 전 주변 설명도 편집해야 한다. 색만 없애는 것으로 미완성 정보가 해결되지는 않는다.
- IEEE와 MDPI의 과학적 본문은 같은 DiSwitch 개정 내용으로 동기화했다. MDPI에는 IEEE 최신본의 가독성 개선 그림도 반영했다.

## 예상 Major comments

### R01. 신규성이 단순 중복 호출 제거인가?

예상 의견: “자체 wrapper의 중복 inference를 없앤 것을 새로운 routing algorithm contribution으로 보기 어렵다. 비교 기준이 의도적으로 느린 구현 아닌가?”

반영: Introduction 기여에서 실행 재사용을 구현 최적화로 낮추고, 초록과 결론에서 Legacy 대비 41.66%를 핵심 성과로 내세우지 않도록 다시 썼다. 최종 정책의 reliability와 측정한 절대 latency를 제시했다. Legacy 비교 그림을 D01로 표시했다.

남은 일: 같은 simulator와 측정 경계에서 외부 baseline의 decision latency를 측정해야 cross-method 효율성 주장이 가능하다. 현재 5.836 ms는 최종 모델의 측정값이며 외부 논문 대비 우수성을 입증하지 않는다.

### R02. Strict-local observation이 실제로 UAV에서 획득 가능한가?

예상 의견: “destination state, neighbor queue, best onward lifetime은 어디서 오는가? one-hop neighbor가 two-hop 정보를 제공한다면 메시지 비용은?”

반영: System Model에 simulator 입력의 local 범위와 실제 측정 가능성의 차이, 정보 획득·신선도·무선 overhead 미측정을 명시했다.

남은 일: feature별 source, beacon payload, 갱신 주기, timestamp, noise, missing-data 처리 표를 작성하고 stale/noisy observation ablation을 실행한다. 로컬 입력이라는 사실만으로 무선 제어비용이 낮다고 주장하면 안 된다.

### R03. 목적함수와 실제 학습이 일치하는가?

예상 의견: “PDR·delay·energy를 서로 다른 단위로 합한 목적식을 실제로 최적화했는가? lambda와 latency budget은?”

반영: 실제로 푼 적 없는 통합 constrained optimization 식을 제거하고 reliability 우선, delay/latency 별도 보고로 문제 정의를 바꿨다. PPO와 KD 손실은 유지했다.

남은 일: 실제 reward, value/entropy loss 계수, gamma, GAE, optimizer, epoch, batch, early stopping을 checkpoint config에서 추출해 표로 제공해야 한다.

### R04. Distillation의 기여가 독립적으로 검증되었는가?

예상 의견: “Predictive branch는 residual을 0으로 설정했다는데 두 branch 모두 learned policy인가? KD가 없어도 같은 성능인가?”

반영: 초록에서 local student + predictive prior로 구체화하고, ablation을 역사적 checkpoint 비교로부터 강한 인과적 기여를 단정할 수 없도록 고쳤다. 활성 loss weights와 checkpoint-selection rule 누락을 파란색으로 표시했다.

남은 일: 동일 구조·training budget·split을 쓰는 no-KD student, geographic-only, normal-only, predictive-only, learned/fixed switch 비교가 필요하다. teacher upper bound도 동일 observation contract 차이를 명시하고 제시할 수 있다.

### R05. Risk switch 수식에서 DROP의 위험도는 무엇인가?

예상 의견: “DROP에는 link margin이 없는데 D(a)와 G(aP,aN)를 어떻게 평가하는가? no-neighbor와 missing-risk에서 정의되는가?”

반영: 실제 `implementations/lite_globe/models/student_policy.py`의 clamp 규칙을 확인하여 candidate capacity K, DROP index K, iota(a)=min(a,K−1)를 수식 앞에 명시했다. no valid neighbor와 missing-risk에서 normal branch로 전환하지 않는(S=0) 동작도 명시했다. 이를 물리적 DROP 위험도가 아닌 구현 인덱싱 규칙으로 설명했다.

남은 일: predictive action=DROP, 마지막 슬롯 padding, no-neighbor, missing-risk의 exhaustive regression test가 필요하다. 이 동작을 개선하면 정책이 달라질 수 있으므로 기존 수치에 새 semantics를 소급 적용하지 말고 재평가한다. “안전 보장”이라는 표현은 현재 근거로 불가하다.

### R06. Acceptance gate가 preregistered인가? Non-inferiority 해석이 올바른가?

예상 의견: “0.005는 어떤 근거로 정했으며 95% LCB는 one-sided인가? gate 실패가 0.5pp 이상 열화의 증명인가?”

반영: preregistration을 입증하지 못하므로 predeclared를 study-defined로 바꿨다. LB가 보관 분석의 양측 95% t interval 하단이라는 점을 설명했다. 0.005=0.5 percentage points를 명시하고 gate 실패와 허용치 초과 열화의 증명을 구분했다. delay·energy에는 정량 margin이 없음을 밝혀 숨겨진 판정 조건을 줄였다.

남은 일: 응용 요구 기반 margin justification과 prospective 분석 계획을 별도 동결한다. Fast PDR CI는 [-0.03606,-0.00011]로 -0.005를 가로지르므로 “열화가 확실히 0.5pp보다 크다”는 표현은 부적절하다.

### R07. 다섯 시드와 다중 비교로 통계적 주장이 과도하지 않은가?

예상 의견: “scenario와 episode를 독립 복제로 세지 않았는가? best baseline을 결과 확인 후 골랐는가?”

반영: seed 수준 추론은 유지하고, 다중비교 미보정·평가 집합 내 strongest comparator 선택·deterministic baseline의 0 variance 한계를 명시했다. significance를 평가하는 비격식 표현을 제거했다.

남은 일: 표본수/검정력 계획에 따른 독립 seed 확장, 사전 지정 primary contrast, 시나리오 계층 bootstrap이나 이에 맞는 분석이 필요하다. “10개면 충분”이라고 보장할 수는 없다.

### R08. 성공한 패킷만의 delay가 생존 편향을 만들지 않는가?

예상 의견: “많이 버리는 알고리즘이 빨라 보일 수 있다. simulator step이 실제 E2E ms인가?”

반영: 성공 패킷 집합이 method마다 다름을 설명했다. AODV/OLSR의 step-delay 차이를 neural inference가 없기 때문이라고 연결했던 잘못된 인과 설명을 제거했다. physical E2E로 환산하려면 MAC·queue·time-step 모델이 필요함을 명시했다.

남은 일: unconditional deadline-delivery curve, failure 포함 latency distribution 또는 time-to-delivery with censoring 분석을 추가한다. 현재 step 수에 임의 ms 계수를 곱하지 않는다.

### R09. Adapted baseline이 공정한가?

예상 의견: “AODV/OLSR route discovery와 control traffic이 실제 프로토콜 수준인가? deep RL baseline의 학습 budget과 hyperparameter search는 동등한가?”

반영: 초록의 우수성 주장을 evaluated implementations로 한정했다. 기존 adapted 설명과 simulator limitation을 유지했다.

남은 일: 원 논문 대 구현 기능 차이 표, 동일 simulator budget, 동일 traffic seeds, tuning budget, observation access를 공개한다. 실제 프로토콜 재현 전 “원 논문을 능가”한다고 쓰면 안 된다.

### R10. Ablation 결과가 causal evidence인가?

예상 의견: “전 단계 체크포인트와 최종 모델은 dataset과 학습 budget이 다른데 component contribution이라 부를 수 있는가?”

반영: historical checkpoints 비교의 confounding을 Results에 추가했다. supports를 is consistent with로 조정했다. 반복 개발 실패를 장황하게 나열한 단락은 D04로 표시했다.

남은 일: paired initialization, 동일 train/validation split, matched budget의 one-factor-at-a-time ablation을 진행한다.

### R11. CPU가 CUDA보다 빠른 원인을 증명했는가?

예상 의견: “동기화와 launch 비용을 직접 분리한 실험 없이 원인을 단정한다. host CPU가 항공기 컴퓨터도 아니다.”

반영: 62.16% faster를 정확한 p95 latency reduction 표현으로 고쳤다. 원인 해석을 profiler와 일치하는 설명으로 완화하고 causal attribution이 아님을 명시했다.

남은 일: target-device timing, randomized blocks, raw timings, CPU model, clocks, power, thermal state를 보완한다. A100 CUDA와 Colab host CPU 결과를 모든 CPU/GPU로 일반화하지 않는다.

### R12. p95와 CI가 정확히 무엇을 요약하는가?

예상 의견: “mean of seed-specific p95를 pooled p95로 오인할 수 있다. raw A100 trace가 없는데 tail을 재검증할 수 있는가?”

반영: 초록과 본문에서 mean seed-specific p95라고 명시하고, quantile 평균과 pooled quantile의 차이를 설명했다. raw trace 미보존이라는 기존 제한은 유지했다.

남은 일: 각 호출 runtime, block order, seed, device, synchronization scope를 남겨 새로 측정한다. 현재 요약 통계로 raw CDF를 재구성하지 않는다.

### R13. Energy와 input bytes가 운영 지표인가?

예상 의견: “negative energy CI, Joule 단위 부재, input bytes를 signaling overhead처럼 보이게 하는 표는 혼동을 준다.”

반영: proxy와 physical quantity를 분리한 설명을 유지했고, 음의 CI 하한이 음의 에너지가 아니라 unconstrained small-sample t interval의 결과임을 설명했다. 표의 bold는 모든 열 최우수가 아닌 proposed row 표시라고 명시했다.

남은 일: energy proxy 산식·failure penalty·분모를 공개한다. support-respecting CI는 원자료로 계산해야 하며 하한을 임의로 0에 잘라 바꾸면 안 된다. physical energy나 radio overhead의 증거로 사용하지 않는다.

### R14. Literature coverage 그림은 체계적 문헌조사인가?

예상 의견: “45개가 convenience sample인데 latency 지표가 드물다는 분야 전체 결론을 내리는가?”

반영: convenience corpus로 명시하고 해당 그림을 D02로 표시했다.

남은 일: 그림을 유지하려면 검색 DB, query, 검색일, 포함·제외 기준, coding rule 및 누락 논문을 공개한다. 현재는 보충자료 이동을 권한다.

### R15. 재현 가능한 공개 아티팩트인가?

예상 의견: “dirty worktree에서 만든 결과라면 commit만으로 복원되는가? 공개 URL과 DOI가 없다.”

반영: 기존 provenance 제한은 삭제하지 않았다. 데이터 공개 및 training config 누락은 파란색으로 남겼다.

남은 일: 정확한 source bundle hash, checkpoint hash, clean release tag, environment lock, raw results를 연결하고 재현 smoke test를 기록한다. 이번 Overleaf ZIP은 논문 컴파일용이지 전체 학습 재현 번들이 아니다.

## 삭제·이동 후보 목록

|ID|대상|권장 처리|이유|삭제 시 함께 확인할 것|
|---|---|---|---|---|
|D01|Legacy repeated execution vs DiSwitch 그림 (`fig:fusion`)|본문에서 빼고 구현 부록으로 이동|내부 wrapper 개선은 라우팅 신규성의 핵심이 아님|execution reuse 설명, legacy latency 행, 내부 equivalence 근거 위치|
|D02|45-paper metric coverage (`fig:litcoverage`)|삭제 또는 literature audit 부록|편의표본 수치가 체계적 조사처럼 읽힐 위험|corpus-count 본문 설명|
|D03|candidate decision map (`fig:decision`)|본문 삭제, 필요 시 supplement 유지|Pareto와 gate 결과와 반복|중복 캡션 및 내부 실험 관리 표현|
|D04|32→48, density augmentation 등 개발 시행착오 단락|근거 테이블과 함께 supplement 이동|비통제 intervention 나열이 causal evidence처럼 읽힘|실패 사례 자체를 숨기지 말고 검증된 핵심 한계는 유지|
|D05|Colab 세션 회수로 미완료된 추가 실행 단락|실험 운영 로그로 이동|논문 핵심 결과와 연결이 약함|raw trace 및 미측정 항목의 한계는 계속 공개|

부정 결과·한계·AI 고지·conflicts·funding은 삭제 후보가 아니다. 붉은 표시는 불리한 증거를 숨기기 위한 것이 아니라 본문/보충자료의 역할을 정리하기 위한 것이다.

## 저널별 예상 의견

### IEEE Access

1. 학습 구조와 구현 최적화 각각의 기여를 명확히 구분할 것 — 반영.
2. 재현 가능한 코드/데이터 및 환경 식별자 제공 — 일부 문서화, 공개 release는 미완료.
3. 두 열 그림 가독성 및 수식 길이 점검 — 분리된 training/deployment 그림 유지, 긴 bound 표기 축약.
4. metadata, funding, biography, AI 고지 확인 — 파란 저자 확인 항목 및 고지 유지. 제공받지 않은 학력·경력은 단정하지 않도록 수정.

### MDPI Drones

1. UAV 운영에서 지리·이동·무선 설정의 실용성과 적용 범위 설명 — 본문의 observability/hardware 한계 강화.
2. 기존 MDPI 초안과 IEEE 최신본의 모델명/그림/수식 불일치 — DiSwitch와 최신 그림으로 동기화.
3. Author Contributions, Funding, Data Availability, IRB/Consent, Conflicts — 필요한 back matter 포함, 미확정 항목 파랑 표시.
4. figure 중복과 개발 연혁 중심 설명을 줄일 것 — D01~D05 표시.

## 투고 전에 필요한 증거의 우선순위

1. 현 코드·체크포인트와 논문의 수식/설정 완전 매핑 및 split manifest 공개.
2. matched no-KD / no-switch / normal-only / predictive-only ablation.
3. adapted baseline contract 및 tuning budget audit.
4. target-device와 무선/MAC/queue를 포함한 E2E 검증.
5. 표본수 계획에 따른 independent seed 확장, raw timing, 다중비교 계획.
6. DOI·저자 승인·공식 소속·이메일·funding·release 확정.

현재 판단은 '서술과 논리 구조는 개선했으나 추가 증거가 필요한 major revision 단계'이다. 문법 교정만으로 위 실험 공백을 해결하거나 게재를 보장할 수 없다.

## 참조한 공식 제출 안내

- [IEEE Access Preparing Your Article](https://ieeeaccess.ieee.org/authors/preparing-your-article/): 공식 템플릿 및 AI 생성 내용 고지 위치 확인.
- [IEEE Access Submission Guidelines](https://ieeeaccess.ieee.org/authors/submission-guidelines/): 제출 형식 점검.
- [MDPI Drones Instructions for Authors](https://www.mdpi.com/journal/drones/instructions): back matter 및 데이터 공개 항목 확인.

위 링크는 형식·정책 확인에 사용했다. 개별 UAV 논문의 모든 DOI/서지사항/주장에 대한 새 원문 전수 검증을 수행한 것은 아니다.
