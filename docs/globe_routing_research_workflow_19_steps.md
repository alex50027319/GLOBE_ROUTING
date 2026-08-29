# GLOBE-ROUTING 연구 하네스 19단계 실행 매뉴얼

> Claude Code 논문 챌린지의 연구 하네스 원칙을 Lite-GLOBE / GLOBE++ / Lite-GLOBE-P+ 코드·실험·논문 흐름에 맞게 변환한 프로젝트 전용 매뉴얼

## 0. 문서 목적과 적용 범위

이 문서는 다음 원문을 바탕으로 작성했다.

- 참고 원문: `클로드코드_논문_챌린지_강의_상세_분석.md`
- 원문의 핵심: 연구 맥락을 파일에 보존하고, 작은 실행·검증 단위로 작업하며, 중요한 판단은 연구자가 승인한다.
- 프로젝트 적용 대상: `GLOBE_ROUTING` 저장소의 Phase 1~13 구현, 로컬 테스트, Colab 실험, 결과 감사, 논문 작성과 투고 준비
- 기준 브랜치: `feature/phase13`의 현재 구조
- 작성 기준일: 2026-08-29

원문에는 명시적인 연구 단계가 14개 있다. 이 매뉴얼은 GLOBE-ROUTING에서 빠지면 안 되는 연구 질문 확정, 체크포인트 계보, smoke/full 분리, seed 단위 복구, 결과 감사를 독립 단계로 세분하여 **19개 실행 단계**로 재구성했다.

이 문서는 실험 결과 자체를 주장하지 않는다. 현재 저장소에 Phase 13 bundle과 seed 42의 chunk 로그가 보이더라도, 그것만으로 Phase 13 full run이 완료되었다고 판단하지 않는다.

---

## 1. 이 프로젝트에서 지켜야 할 세 가지 원칙

### 1.1 정본 우선순위

서로 다른 파일의 설명이 충돌하면 다음 순서로 판단한다.

1. 실행 코드와 phase YAML
2. 원시 episode CSV, checkpoint metadata, manifest
3. 자동 생성된 통계·표·그림
4. README, `.claude/context`, 연구 노트
5. 논문과 발표자료

문서가 코드와 충돌하면 문서의 설명을 사실로 확정하지 말고 `needs-verification`으로 기록한다.

### 1.2 결과 완료 판정

다음은 완료의 충분조건이 아니다.

- 프로세스가 종료됨
- Colab 로그의 마지막 줄이 정상처럼 보임
- 결과 ZIP이 하나 존재함
- 일부 seed의 표가 생성됨
- smoke가 통과함

full 완료는 최소한 `mode=full`, 요청 seed 집합, scenario-method 조합, episode 수, 중복·누락 여부, raw CSV와 manifest를 함께 확인해 판정한다.

### 1.3 사람 승인 없이 넘지 않는 게이트

- 연구 질문·가설과 주장 범위
- config, seed, scenario, method, episode 수
- upstream checkpoint 선택
- 유료 Colab/GPU full run 시작
- 메인 표·그림과 핵심 성능 주장
- 목표 저널과 원고의 최종 문장
- 추가 분석, 주장 축소, 윤리·AI 공개, 최종 제출

---

## 2. GLOBE-ROUTING의 연구 객체와 정본 경로

일반 데이터 연구의 `raw data → processed data → model → results`는 이 프로젝트에서 다음처럼 대응한다.

| 연구 객체 | GLOBE-ROUTING 정본 | 설명 |
| --- | --- | --- |
| 연구 문제 | `docs/paper_method_summary.md` | Dec-POMDP, global-to-local distillation, PRS/P+ 정식화 |
| 환경·관측·행동 | `implementations/lite_globe/env/` | FANET 동역학, observation, action mask, reward |
| 방법 | `models/`, `algorithms/`, `baselines/` | Teacher, Student, PPO/KD, GPSR, external RL, risk-aware 정책 |
| 시나리오 | `scenarios/`, `experiments/` | 일반/OOD, structural hole, predictive break |
| 실험 계획 | `config/phaseN.yaml`, `run_phaseN.py` | seed, episode, calibration grid, CLI 계약 |
| 코드 검증 | `tests/lite_globe/` | checkpoint 없이 가능한 결정적 검증 |
| 학습·평가 실행 | `run_phaseN.py`, `scripts/` | 로컬/Colab 실행, 패키징, 재개, 병합 |
| 학습 상태 | `artifacts/lite_globe/phaseN/checkpoints/` | seed별 모델과 training metadata |
| 원시 결과 | `artifacts/lite_globe/phaseN/raw/` | episode와 seed summary CSV |
| 통계 결과 | `artifacts/lite_globe/phaseN/summaries/` | 집계와 paired effect |
| 논문용 산출물 | `artifacts/lite_globe/phaseN/tables/`, `figures/` | 자동 생성된 표와 그림 |
| 실행 증명 | `artifacts/lite_globe/phaseN/manifest.json` | phase, mode, config, 환경, row 수 |
| 논문 | `paper/`, `kci_paper/` | 브랜치 정책을 확인한 뒤 편집 |

`artifacts/`는 생성 결과다. 수동으로 CSV·표·그림을 고치지 않고 생성 코드에서 수정한 뒤 새 출력 경로에 재생성한다.

### Phase 계보

| Phase | 연구상 역할 | 주요 의존성 |
| ---: | --- | --- |
| 1 | FANET 환경과 Random/GPSR 기준선 | 없음 |
| 2 | 1-hop Local Student 구조 | Phase 1 관측 계약 |
| 3 | privileged Global Teacher와 PPO | Phase 1 환경 |
| 4 | Teacher → Student offline distillation | Phase 3 Teacher |
| 5 | Local PPO fine-tuning | Phase 4 Student, 선택적 dataset |
| 6 | multi-seed 평가와 보고 | Phase 3~5 방법 |
| 7 | 동적 topology·OOD·타당성 강화 | 앞선 Teacher/Student 계보 |
| 8 | Geo-Residual 최적화 | Phase 7 checkpoints |
| 9 | risk-aware 학습·평가 | Phase 7·8 checkpoints |
| 10 | IQMR/DRAMA/Evo-QGeo 등 외부 RL 비교 | Phase 8 checkpoint |
| 11 | predictive branch인 Lite-GLOBE-P | Phase 7·8 checkpoints |
| 12 | normal/predictive Risk-Switch 보정 | Phase 8·11 checkpoints |
| 13 | redundancy·loss keep·energy tie·drop suppression의 P+ | Phase 8·11·12 checkpoints |

> 주의: Global Teacher는 구현 계보상 Phase 3에서 도입된다. Phase 7은 Teacher/Student 구조의 시작점이 아니라 일반화와 타당성 강화 단계다.

---

## 3. 원문과 19단계의 대응표

| 이 매뉴얼 | 원문의 대응 내용 |
| ---: | --- |
| 1 | 프로젝트 폴더로 컨텍스트 엔지니어링 |
| 2 | 인간 승인 지점, 연구 질문 고정 |
| 3 | EDA를 시뮬레이터 입력·계약 감사로 변환 |
| 4 | EDA와 탐색적 가설 생성 |
| 5 | 최신 문헌으로 연구 갭 검증 |
| 6 | 탐색적/확인적 가설 분리와 승인 |
| 7 | 분석 계획 파일화 |
| 8 | 분석 실행 전 코드·smoke 검증 |
| 9 | 재현 환경과 체크포인트 계보 고정 |
| 10 | 가설별 분석과 multi-seed full 실행 |
| 11 | 병렬화, 중단 복구, seed 결과 병합 |
| 12 | 결과 검증 체크리스트와 엄밀성 장치 |
| 13 | 원고 전 그림·표 설계 |
| 14 | 초록 우선 작성 |
| 15 | 섹션별 문헌 확장과 DOI/RIS 검증 |
| 16 | scientific writing guide와 순차 원고 작성 |
| 17 | 목표 저널 탐색 |
| 18 | 네 관점의 모의 심사와 수정 |
| 19 | author guideline, 최종 감사, 제출 파일 |

---

# 19단계 실행 절차

## Step 1. 프로젝트 하네스와 현재 상태를 고정한다

### 목적

에이전트와 사람이 같은 저장소 지도, 정본 우선순위, 브랜치 상태를 기준으로 작업하게 한다.

### 적용 절차

1. 저장소 루트에서 브랜치, commit, 변경 파일을 기록한다.
2. `CLAUDE.md`, `README.md`, `docs/paper_method_summary.md`, `implementations/lite_globe/README.md`, `assumptions.md`를 먼저 읽는다.
3. 현재 목표 phase와 upstream dependency를 적는다.
4. 기존 사용자 변경, checkpoint, artifact, paper를 덮어쓰지 않는다고 명시한다.
5. 세션의 한 가지 목표, 입력 파일, 산출물, 성공 기준을 파일에 남긴다.

```bash
pwd
git branch --show-current
git rev-parse HEAD
git status --short
python3 --version
```

`pyproject.toml`의 요구사항은 Python 3.11 이상이다. 시스템 `python3`가 3.11 미만이면 그 인터프리터로 실험을 시작하지 않는다.

### 권장 관리 파일

현재 브랜치에서 연구 제어 문서를 추가할 경우 다음처럼 코드와 분리한다.

```text
docs/research/
├── research_question.md
├── hypotheses.md
├── analysis_plan.md
├── decision_log.md
├── ai_usage_log.md
├── literature/
├── claims/
└── submission/
```

논문 원본은 브랜치 정책을 따른다. `feature/phase13`에서 없던 `paper/`를 자동 복원하지 않는다.

### 완료 기준

- 다른 사람이 현재 branch/commit, 목표 phase, 정본, 다음 작업을 설명할 수 있다.
- 작업 전 dirty file과 보호 대상 artifact를 알고 있다.

### 인간 승인

연구 범위와 이번 세션의 산출물 경로를 승인한다.

---

## Step 2. 연구 질문과 주장 경계를 한 문장으로 고정한다

### 목적

Phase를 실행하는 것과 연구 질문에 답하는 것을 구분한다.

### GLOBE-ROUTING 적용 예시

상위 질문은 다음 틀로 작성한다.

> 전역 그래프 정보를 이용해 학습한 Teacher의 정책을 1-hop 관측만 사용하는 Student에 증류하고, 예측 위험에 따라 안전 분기로 전환하면, 동적 FANET의 일반·OOD·구조적 위험 조건에서 비용을 과도하게 늘리지 않으면서 전달 신뢰성을 개선할 수 있는가?

하위 질문은 분리한다.

- RQ1: Global-to-Local distillation이 로컬 정책의 성능과 경량성에 주는 효과는 무엇인가?
- RQ2: Geo-Residual이 일반/OOD와 structural hole에서 GPSR 계열의 local minimum을 줄이는가?
- RQ3: predictive branch와 Risk-Switch가 `predictive_break`에서 normal branch의 실패를 줄이는가?
- RQ4: P+ 구성요소가 PDR뿐 아니라 deadline, p95 delay, energy, input bytes, drop에 미치는 trade-off는 무엇인가?

### 주장 수준 라벨

모든 claim에는 다음 중 하나를 붙인다.

- `methodological`: 구조·알고리즘 자체에 대한 주장
- `experimental`: 지정된 scenario/seed/metric에서 확인한 결과
- `exploratory`: 결과를 본 뒤 형성한 후보
- `limitation`: 단일 packet, 2D mobility, proxy metric 등 적용 한계
- `needs-verification`: 코드·raw result·문헌 대조 전 상태

### 산출물

`docs/research/research_question.md` 또는 기존 연구 문서에 질문, 포함 범위, 제외 범위, 금지 표현을 기록한다.

### 완료 기준

- 각 질문이 최소 하나의 scenario, method 비교, metric으로 검증 가능하다.
- “실제 FANET 전체에서 우월하다”처럼 시뮬레이션 범위를 넘는 문장이 없다.

### 인간 승인

연구 질문, primary outcome, 주장 범위를 연구자가 승인한다.

---

## Step 3. 시뮬레이터·설정·지표를 데이터처럼 EDA한다

### 목적

일반 연구의 raw-data EDA를 이 프로젝트의 **실험 생성 계약 감사**로 바꾼다.

### 입력

- `implementations/lite_globe/env/`
- `implementations/lite_globe/scenarios/`
- `implementations/lite_globe/experiments/`
- `implementations/lite_globe/config/phaseN.yaml`
- `implementations/lite_globe/evaluation/`
- `implementations/lite_globe/assumptions.md`

### 점검 항목

1. 상태·관측·행동 공간과 action mask
2. source/destination의 무작위 또는 고정 여부
3. mobility, link loss, node 수, 통신 반경, TTL
4. seed가 환경·정책·평가에 전달되는 방식
5. PDR 분모, delay가 성공 episode만 사용하는지 여부
6. energy, ETX, input bytes, link lifetime이 실측값인지 proxy인지 여부
7. 일반/OOD, structural hole, predictive break의 목적 차이
8. simulation unit과 meter의 변환 가정 존재 여부

### 산출물

- scenario × parameter × endpoint 규칙 표
- metric 정의와 분모 표
- 구현과 문서의 불일치 목록
- 미검증 단위와 proxy 목록

### 금지 사항

- 코드에 물리 단위 변환이 없는데 `area_size=10.0`을 meter로 확정하지 않는다.
- queue feature가 존재한다는 이유만으로 다중 packet queueing을 구현했다고 쓰지 않는다.
- toy routing-hole 결과를 일반 FANET 성능으로 해석하지 않는다.

### 완료 기준

각 논문 metric이 어느 함수와 raw column에서 만들어지는지 추적할 수 있다.

---

## Step 4. 기준선 행동을 탐색하고 탐색적 가설을 만든다

### 목적

모델을 추가하기 전에 Random, GPSR, Teacher, 기존 Student가 어떤 조건에서 실패하는지 파악한다.

### 적용 절차

1. checkpoint가 필요 없는 unit test와 Phase 1/2의 작은 실행부터 확인한다.
2. 시나리오별 실패 원인을 `disconnect`, `loop`, `TTL`, `agent drop`, `deadline` 등으로 분해한다.
3. PDR만 보지 않고 delay, path stretch, energy, input bytes, drop을 함께 본다.
4. structural hole과 predictive break를 서로 다른 문제로 취급한다.
5. 관찰을 exploratory claim으로 저장한다.

```bash
python -m pytest tests/lite_globe/test_environment.py \
  tests/lite_globe/test_baselines.py
python -m implementations.lite_globe.run_phase1 --episodes 20 --seed 42
python -m implementations.lite_globe.run_phase2 --episodes 20 --seed 42
```

### 탐색적 가설 예시

- E1: geographic progress만 사용하는 GPSR은 structural hole에서 local minimum에 취약하다.
- E2: 현재 link margin만 좋은 후보는 relative mobility 때문에 곧 끊길 수 있다.
- E3: 안전 분기 전환이 과하면 PDR이 좋아져도 delay·input bytes·switch 횟수가 악화될 수 있다.
- E4: Student의 성능은 Teacher 모방 오차뿐 아니라 global-to-local information gap의 영향을 받는다.

### 완료 기준

각 후보에 예상 방향, 반대 결과, 교란 요인, 검증 scenario, 비교 method, metric이 연결되어 있다.

### 인간 승인

이 단계에서는 가설을 확정하지 않는다. 다음 문헌 검증으로 넘길 후보만 선택한다.

---

## Step 5. 최신 문헌으로 연구 갭과 비교 기준을 검증한다

### 목적

Lite-GLOBE-P+의 새로움을 구성요소 이름이 아니라 문제·정보·학습·실행 관점에서 검증한다.

### 검색 축

- FANET/UAV ad hoc routing
- geographic routing과 local minimum/perimeter recovery
- GNN 기반 privileged/global routing policy
- centralized training and decentralized execution
- policy distillation / knowledge distillation for routing
- predictive link lifetime, mobility-aware routing
- risk-aware switching, safe RL, hybrid routing
- IQMR, DRAMA, Evo-QGeo 및 실제 구현한 Phase 10 baseline

### 비교 표의 필수 열

| 열 | 질문 |
| --- | --- |
| Observation | global graph, k-hop, 1-hop beacon 중 무엇을 보는가? |
| Training | supervised, RL, distillation, hybrid 중 무엇인가? |
| Deployment | 중앙집중형인가 분산형인가? |
| Prediction | 현재 링크만 쓰는가 미래 lifetime을 쓰는가? |
| Switching | 단일 정책인가 normal/safety 분기인가? |
| Evaluation | node 수, mobility, loss, seed, episode가 무엇인가? |
| Cost | model parameter, input bytes, delay, energy를 보고하는가? |
| Evidence | 원문/코드/공식 DOI를 확인했는가? |

### 산출물

- `docs/research/literature/search_protocol.md`
- `docs/research/literature/gap_validation.md`
- `docs/research/literature/evidence_table.csv`
- 반대 결과와 중복 위험 목록

### 완료 기준

- 핵심 선행연구는 원문까지 확인했다.
- DOI, 제목, 저자, 연도, 저널이 공식 출처와 일치한다.
- Phase 10의 재구현 결과와 원 논문의 보고 결과를 혼합하지 않는다.

### 인간 승인

연구 갭과 기여 문장은 연구자가 원문을 읽은 뒤 승인한다.

---

## Step 6. 탐색적 후보를 검증 가설과 ablation 질문으로 변환한다

### 목적

결과를 본 뒤 만든 설명을 사전에 정한 확인 가설처럼 쓰는 일을 막는다.

### 권장 가설 구조

- H1 — distillation: Student가 로컬 관측만 사용하면서 지정된 조건에서 Teacher의 유용한 행동 구조를 학습하는가?
- H2 — generalization: Geo-Residual이 held-out/OOD 및 structural hole에서 지정 기준선보다 일관된 개선을 보이는가?
- H3 — prediction: Lite-GLOBE-P가 predictive break에서 Phase 8 normal branch의 실패를 줄이는가?
- H4 — switching: Phase 12 Risk-Switch가 normal 조건의 성능 보존 허용치 안에서 predictive 조건을 개선하는가?
- H5 — P+ ablation: Phase 13의 redundancy/loss keep/energy tie/drop suppression 효과가 seed 전반에서 어떤 trade-off를 보이는가?

### 각 가설의 필수 필드

```text
ID / 상태(exploratory 또는 confirmatory)
비교 method
대상 scenario
primary metric과 방향
secondary metric과 허용 가능한 손실
seed와 episode
통계 비교 단위
성공·실패·중단 기준
반대 결과의 해석
결과 확인 전 작성 시각과 승인자
```

### 완료 기준

가설별로 코드 실행 없이도 성공 판정 규칙을 설명할 수 있다.

### 인간 승인

confirmatory 라벨과 primary metric을 고정한다. 이후 변경은 `post_hoc`로 기록한다.

---

## Step 7. 분석·실험 계획을 파일로 동결한다

### 목적

채팅 속 계획이 아니라 재현 가능한 실행 계약을 만든다.

### 프로젝트 적용

`.claude/templates/experiment-plan.md`를 시작점으로 사용할 수 있다. 각 Phase 실행마다 별도 plan을 만들고 다음 항목을 채운다.

- phase, branch, commit, config checksum
- training/evaluation seed
- scenario, method, episode per cell
- 기대 raw row 수 산식
- upstream checkpoint 경로와 checksum
- smoke/full/merged 구분
- primary/secondary metric
- paired comparison과 confidence interval 방식
- calibration grid와 선택 tie-break
- output directory
- resume/recovery 정책
- 성공·실패·중단 기준
- GPU 비용 승인 여부

### Phase 13 계획 예시

```text
Phase: 13
Seeds: 42, 77, 123, 314, 2718
Evaluation episodes: scenario-method-seed cell당 200
Dependencies: Phase 8 geo_residual_kd, Phase 11 lite_globe_p,
              Phase 12 risk_switch_lite_globe_p
Primary question: P+가 Phase 12 대비 predictive risk 조건을 개선하는가?
Guardrails: normal/OOD 성능 보존, delay/energy/input-bytes/drop 동시 확인
Output: 기존 결과와 겹치지 않는 명시적 directory
```

### 완료 기준

- 계획만 읽고 실행 명령, 기대 산출물, 중단 후 복구 방법을 알 수 있다.
- 아직 full run을 실행하지 않았다.

### 인간 승인

config·seed·metric·비용·출력 경로를 승인한 뒤 Step 8로 간다.

---

## Step 8. unit test와 smoke를 실행 전 게이트로 사용한다

### 목적

코드 계약 오류를 비싼 full run 전에 찾는다.

### 검증 순서

```bash
# 0) Python 3.11+ 가상환경 준비
python3 -c 'import sys; assert sys.version_info >= (3, 11), sys.version'
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lite-globe.txt
python -m pip install -e .

# 1) 전체 결정적 코드 검증
python -m pytest tests/lite_globe

# 2) 대상 phase의 집중 검증
python -m pytest tests/lite_globe/test_phase13_risk_switch_plus.py

# 3) upstream checkpoint를 준비한 뒤 축소 실제 파이프라인
python -m implementations.lite_globe.run_phase13 \
  --smoke \
  --device cpu \
  --resume \
  --phase8-checkpoint-dir /validated/checkpoints/phase8 \
  --phase11-checkpoint-dir /validated/checkpoints/phase11 \
  --phase12-checkpoint-dir /validated/checkpoints/phase12 \
  --output-dir artifacts/lite_globe/phase13_smoke_YYYYMMDD
```

작성 시점의 기본 셸은 Python 3.9.6이며 `pytest`가 설치되어 있지 않아 전체 suite를 실행할 수 없었다. 따라서 실제 실행 전 위 환경 준비를 완료하고, 그 검증 결과를 experiment plan에 기록해야 한다.

### 해석 규칙

- `pytest` 통과는 checkpoint 없이 코드 레벨 계약이 맞다는 뜻이다.
- `--smoke` 통과는 축소된 실제 campaign이 끝났다는 뜻이다.
- smoke 수치는 논문 full 결과에 합치지 않는다.
- Phase 8 이후 smoke는 upstream checkpoint가 필요하다.
- 기존 smoke 디렉터리를 무심코 덮어쓰지 않는다.

### 완료 기준

test와 smoke가 모두 통과하고, smoke manifest의 phase/mode/config/output이 계획과 일치한다.

---

## Step 9. 체크포인트·환경·bundle 계보를 검증한다

### 목적

잘못된 upstream 모델 또는 불완전 bundle로 실행한 결과를 차단한다.

### Phase 13 필수 checkpoint

각 seed `42, 77, 123, 314, 2718`에 대해 다음 파일이 필요하다.

```text
phase8/checkpoints/seed_<seed>/geo_residual_kd.pt
phase8/checkpoints/seed_<seed>/training_metrics.json
phase11/checkpoints/seed_<seed>/lite_globe_p.pt
phase11/checkpoints/seed_<seed>/training_metrics.json
phase12/checkpoints/seed_<seed>/risk_switch_lite_globe_p.pt
phase12/checkpoints/seed_<seed>/training_metrics.json
```

### 검증 절차

1. `scripts/package_phase13_colab.py`의 `required_paths()`와 실제 bundle 목록을 비교한다.
2. 모든 seed의 모델·metadata가 있는지 확인한다.
3. checkpoint가 같은 코드 계보와 hidden dimension을 사용하는지 확인한다.
4. Python, Torch, OS, device를 manifest에 남긴다.
5. bundle의 hash와 생성 시각을 experiment plan에 기록한다.

```bash
unzip -l artifacts/lite_globe/phase13_colab_bundle.zip
shasum -a 256 artifacts/lite_globe/phase13_colab_bundle.zip
python -m pip freeze
```

### 현재 상태 해석 예시

- bundle 안에 checkpoint가 있다는 사실과 로컬 기본 checkpoint 경로에 파일이 있다는 사실은 다르다.
- seed chunk 로그에 bootstrap 시작이 보인다는 사실과 해당 seed 결과 ZIP이 완성되었다는 사실은 다르다.

### 완료 기준

요청 seed 전체의 dependency와 metadata가 검증되고 plan에 checksum이 기록되어 있다.

### 인간 승인

어느 checkpoint 세트를 정본으로 사용할지 승인한다.

---

## Step 10. 가설별 full campaign을 독립 실행한다

### 목적

승인된 계획의 비교를 동일한 조건에서 실행하고 원시 결과를 보존한다.

### 병렬 실행 가능 범위

- seed가 서로 독립이고 출력 경로가 분리된 실행
- 서로 다른 문헌 검색 축
- 결과를 공유하지 않는 독립 reviewer

### 순차 실행 범위

- Phase 8 → 11 → 12 → 13 checkpoint 계보
- calibration → fixed-policy evaluation
- raw result → aggregation → figure/table
- Results → Discussion

### Phase 13 권장 full 실행

유료 원격 실행은 사용자 승인 후에만 시작한다.

```bash
python scripts/run_phase13_seed_queue.py \
  --seeds 42,77,123,314,2718 \
  --gpu A100 \
  --calibration-candidates-per-chunk 32 \
  --evaluation-units-per-chunk 20 \
  --exec-timeout 7200 \
  --detach
```

### 실행 중 기록

- session name에 phase와 seed
- chunk 번호와 시작/종료 시각
- local process, remote session, log 수정 시각
- 입력 bundle hash
- 새로 처리한 calibration/evaluation unit
- result ZIP 경로와 hash
- 실패 원인과 재개 지점

### 완료 기준

각 seed의 최종 결과 ZIP이 존재하고 내부 raw CSV와 실행 metadata가 유효하다.

---

## Step 11. 중단을 복구하고 seed 결과를 검증·병합한다

### 목적

Colab 세션 단절을 연구 결과 손실이나 중복 실행으로 바꾸지 않는다.

### 복구 원칙

1. 오래된 로그 한 줄만 보고 실행 중이라고 판단하지 않는다.
2. 로컬 프로세스, 원격 session, 결과 ZIP을 각각 확인한다.
3. 마지막 유효 chunk의 progress signature를 확인한다.
4. 같은 queue 명령으로 미완료 unit만 재개한다.
5. 완료 seed를 `--force` 없이 다시 실행하지 않는다.

### 병합 예시

```bash
python scripts/merge_phase13_artifacts.py \
  --inputs \
    artifacts/lite_globe/phase13_seeds_42_results.zip \
    artifacts/lite_globe/phase13_seeds_77_results.zip \
    artifacts/lite_globe/phase13_seeds_123_results.zip \
    artifacts/lite_globe/phase13_seeds_314_results.zip \
    artifacts/lite_globe/phase13_seeds_2718_results.zip \
  --output-dir artifacts/lite_globe/phase13_merged_YYYYMMDD
```

병합 스크립트가 검사하는 seed/scenario/method/episode grid를 우회해 CSV를 수동 결합하지 않는다.

### 완료 기준

- 병합 manifest의 seed 집합이 plan과 같다.
- raw episode 수와 seed summary의 declared episode 수가 같다.
- 중복 row와 예상 밖 조합이 없다.
- `partial_manifest.json`만 있는 결과를 full로 쓰지 않는다.

---

## Step 12. 결과를 통계·재현성·주장 관점에서 감사한다

### 목적

“코드가 실행되었다”를 “논문 근거로 사용할 수 있다”와 구분한다.

### 감사 입력

- `manifest.json`
- `raw/episodes.csv`
- `raw/seed_summaries.csv`
- `raw/training_metrics.csv`
- `summaries/statistics.csv`
- `summaries/paired_effects.csv`
- `tables/`
- `figures/`
- phase config와 reporting code

### 필수 검사

1. phase와 mode가 맞는가?
2. seed·scenario·method grid가 완전한가?
3. 중복 row가 없는가?
4. metric 분모와 성공 episode 필터가 맞는가?
5. seed 단위 paired comparison인가?
6. 평균, 표본 표준편차, 95% CI가 올바른가?
7. raw → summary → table → figure 숫자가 일치하는가?
8. 실패 원인과 불리한 metric이 보존되었는가?
9. calibration에 사용한 평가를 최종 검증처럼 재사용하지 않았는가?
10. Phase 10 외부 baseline과 Phase 12/13 내부 ablation을 같은 campaign처럼 합치지 않았는가?

### Phase 13 해석 묶음

최소한 다음을 함께 본다.

- PDR와 deadline
- mean/p95 delay
- energy
- input bytes
- agent drop rate
- switch steps 또는 switch rate
- seed 변동과 paired effect

### 산출물

`.claude/templates/result-audit.md` 형식의 감사 보고서를 만들고 verdict를 다음 중 하나로 쓴다.

- `complete`
- `incomplete`
- `not comparable`
- `needs rerun`

### 인간 승인

감사 verdict와 논문에 사용할 수치·주장 범위를 승인한다.

---

## Step 13. 논문을 쓰기 전에 주장–표–그림을 설계한다

### 목적

가장 예쁜 그림이 아니라 연구 질문에 답하는 최소 증거 묶음을 선택한다.

### 권장 claim-evidence map

| Claim | Evidence | 확인할 반대 지표 |
| --- | --- | --- |
| 일반/OOD 성능 보존 | Phase 8/12/13 동일 조건 paired result | delay, energy, input bytes |
| predictive break 개선 | Phase 11/12/13 predictive scenarios | normal scenario 손실, switch 과다 |
| P+ 구성요소 기여 | Phase 13 ablation | seed 불일치, 단일 metric 개선 |
| 외부 RL 대비 위치 | Phase 10 결과와 조건 정합 비교 | 다른 campaign/config 위험 |
| 경량 분산 실행 | parameter/input bytes/inference cost | Teacher는 배포 대상 아님 |

### Phase 13 자동 생성 주요 산출물

```text
tables/risk_switch_plus_results.md
tables/risk_switch_plus_paired_effects.md
figures/risk_switch_plus_pdr.svg
figures/risk_switch_plus_delay_p95.svg
figures/risk_switch_plus_input_bytes.svg
figures/risk_switch_plus_switch_steps.svg
```

### 그림·표 규칙

- smoke와 full을 시각적으로도 혼합하지 않는다.
- scenario, method, seed 수, episode 수, 단위, CI를 캡션에 쓴다.
- PDR와 delay/energy를 같은 척도처럼 표현하지 않는다.
- 색각 이상과 흑백 인쇄를 고려한다.
- figure의 값은 SVG를 눈으로 읽어 전사하지 않고 summary CSV에서 생성한다.

### 완료 기준

각 메인 claim이 raw result까지 역추적되고, 각 figure/table의 위험한 해석이 기록되어 있다.

### 인간 승인

메인 Figure/Table과 캡션의 핵심 메시지를 승인한다.

---

## Step 14. 검증된 수치만 사용해 초록을 먼저 쓴다

### 목적

연구 질문–방법–결과–기여의 논리적 틈을 짧은 형식에서 먼저 찾는다.

### 작성 순서

1. FANET의 동적 topology와 로컬 관측 제약
2. global Teacher와 local Student 사이의 문제
3. distillation + Geo-Residual + predictive Risk-Switch/P+
4. scenario, seed, 주요 비교 방법
5. primary 결과와 uncertainty
6. delay/energy/input overhead trade-off
7. 시뮬레이션 범위로 제한된 기여

### 수치 추적 형식

초록의 모든 숫자 옆에 내부 검토용 근거를 연결한다.

```text
[숫자] → phase / scenario / method / seeds / metric
       → summaries/paired_effects.csv row
       → raw/seed_summaries.csv
```

### 버전

- `abstract_draft_ai.md`
- `abstract_reviewed.md`
- `abstract_submission.md`

### 금지 사항

- full result가 없으면 숫자를 채워 넣지 않는다.
- PDR 하나만으로 overall superiority를 선언하지 않는다.
- 2D·single-packet·proxy 기반 시뮬레이션을 실세계 검증으로 표현하지 않는다.

### 인간 승인

제목, 핵심 수치, novelty 문장, 한계 문장을 승인한다.

---

## Step 15. 초록을 기준으로 문헌을 확장하고 DOI/RIS를 검증한다

### 목적

원고의 각 기능 문장에 실제로 맞는 근거를 연결한다.

### 섹션별 검색

- Introduction: FANET routing 문제, partial observation, 기존 접근의 한계
- Methods: GNN Teacher, PPO, policy distillation, CTDE, link lifetime, switching
- Results: 동일 metric·scenario를 보고한 비교 연구
- Discussion: 정보 격차, 일반화, 안전–비용 trade-off, simulation-to-real gap

### 문헌 레코드

```text
Citation key
지지하는 원고 문장
직접/간접 근거
연구 설계·환경·표본
핵심 결과와 반대 해석
DOI와 공식 URL
초록/원문 확인 상태
철회·정정 확인 상태
예상 인용 위치
```

### RIS 처리

- DOI는 doi.org, Crossref, 출판사 공식 페이지 등으로 대조한다.
- 제목·저자·연도·저널이 모두 맞을 때만 verified로 표시한다.
- 자동 접근 실패나 메타데이터 불일치는 `manual_todo.md`로 보낸다.
- 확인되지 않은 저자·페이지·DOI를 추정 생성하지 않는다.

### 완료 기준

모든 핵심 인용 문장을 원문이 실제로 지지하고, 반대 결과와 핵심 원전이 포함되어 있다.

### 인간 승인

최종 인용 전 연구자가 원문 문맥을 확인한다.

---

## Step 16. 프로젝트 writing guide를 만들고 원고를 순차 작성한다

### 목적

섹션 간 용어·수치·주장의 불일치를 줄이고 모든 문장을 근거로 추적한다.

### writing guide 필수 항목

- GLOBE, Lite-GLOBE, Lite-GLOBE-P, Risk-Switch, P+ 용어 정의
- Teacher는 privileged reference policy이며 배포 대상/optimal upper bound가 아님
- Student는 1-hop local observation만 사용
- structural hole과 predictive break의 구분
- PDR, delay, deadline, energy, input bytes, drop의 표기
- simulation unit과 proxy 공개 방식
- 탐색적/확인적 결과 라벨
- 통계 수치, CI, 반올림 규칙
- AI 사용 기록과 공개 문안

### 작성 순서

```text
Methods
→ Results
→ Discussion
→ Introduction
→ Conclusion
→ 전체 통합 검토
```

### 섹션별 정본

- Methods: code, config, scenario, analysis plan
- Results: raw/summaries/table/figure
- Discussion: Results + verified literature + limitations
- Introduction: research question + gap validation + final contribution
- Conclusion: 본문에 이미 제시된 근거만 사용

### 문단 작성 루프

```text
섹션 목적
→ 소제목
→ 문단 기능
→ 핵심 문장
→ 인간 승인
→ 한 문단 작성
→ claim-evidence 대조
→ 섹션 통합
```

### 필수 Methods 공개

- Python/Torch와 hardware/device
- phase config, seeds, episodes
- source/destination 규칙
- checkpoint와 training 계보
- calibration과 evaluation의 분리
- metric 정의와 통계 방식
- resume/merge 절차
- 코드·데이터·artifact 가용성

### 완료 기준

모든 수치·인용·방법 문장을 코드나 evidence로 추적할 수 있고 섹션 간 용어가 일치한다.

### 인간 승인

AI가 만든 모든 최종 문장을 저자가 이해·수정·승인한다.

---

## Step 17. 목표 저널은 적합성과 탈락 위험을 함께 탐색한다

### 목적

AI의 추천을 게재 가능성 예측이 아니라 후보 탐색 도구로 제한한다.

### 비교 항목

- aims & scope와 최근 2~3년 실제 논문
- FANET, networking, distributed AI/RL 독자층 적합성
- simulation-only 연구 허용 범위
- reproducibility, code/data policy
- article type과 분량
- APC, indexing, review model
- 생성형 AI 공개 정책
- desk rejection 위험
- 약탈적 학술지 위험

### GLOBE-ROUTING 특화 반증 질문

- 외부 hardware/testbed 검증이 없다는 점이 치명적인가?
- 단일 packet, 2D mobility, simplified link model이 scope 기대에 미달하는가?
- Phase 10 external baseline 비교가 충분히 동등한 조건인가?
- novelty가 기존 mobility-aware/risk-aware routing과 실제로 구별되는가?

### 산출물

후보별 공식 URL, 확인 날짜, 적합 근거, 부적합 근거, 필요한 추가 실험을 `docs/research/submission/target_journal.md`에 기록한다.

### 인간 승인

저널 선택은 연구자가 결정한다.

---

## Step 18. 네 관점의 독립 모의 심사와 수정 결정을 수행한다

### 목적

같은 원고를 다른 실패 관점에서 공격해 투고 전 약점을 찾는다.

### 리뷰 역할

| 리뷰어 | GLOBE-ROUTING 관점 |
| --- | --- |
| A | FANET·라우팅 이론, 분산 실행 가능성, scenario 현실성 |
| B | RL/KD 학습, seed, calibration leakage, 통계·paired comparison |
| C | novelty, Phase 10 baseline 공정성, 기존 연구 대비 기여 |
| D | 코드·config·checkpoint·raw CSV·manifest 재현성 |

### 리뷰 출력

- 요약
- Major concerns: 원고 위치와 근거
- Minor concerns
- 필요한 추가 분석 또는 주장 축소
- 현재 정보로 판단 불가능한 사항
- `P0`~`P3` 심각도

### 우선순위

- P0: 데이터/분석 오류, leakage, 윤리 문제 — 투고 중단
- P1: 핵심 가설을 위협 — 추가 분석 또는 주장 축소
- P2: 재현성·보고·인용 누락 — 원고/부록/코드 보완
- P3: 표현·형식 — 편집 단계 수정

### 주의

같은 모델의 네 persona가 같은 의견을 냈다고 네 명의 독립 전문가가 동의한 것은 아니다. raw artifact, 공식 문헌, 지도교수/동료 검토로 다시 확인한다.

### 완료 기준

모든 P0/P1에 해결, 추가 분석, 주장 축소, 수용 중 하나의 결정과 근거가 있다.

### 인간 승인

추가 GPU 실행, 새 ablation, 주장 축소를 승인한다.

---

## Step 19. author guideline과 최종 제출 패키지를 일대일 감사한다

### 목적

내용이 맞는 원고를 실제 제출 가능한 파일 묶음으로 변환한다.

### 제출 요건

- article type, word/figure/table/reference 한도
- title page, author, affiliation, corresponding author, ORCID
- abstract, keyword, section 순서
- 익명 심사 파일 분리
- figure 해상도·색상·형식, table 형식
- reference style
- data/code availability
- ethics, consent, COI, funding, author contribution
- AI/generative tool disclosure
- cover letter, highlights, graphical abstract, supplement
- 공식 규정 URL, 확인 날짜, 수동 확인 항목

### GLOBE-ROUTING 최종 연구 감사

- [ ] 질문–가설–방법–결과–해석이 일치한다.
- [ ] exploratory와 confirmatory가 구분되어 있다.
- [ ] Phase 10/12/13의 실험 목적과 조건을 혼합하지 않았다.
- [ ] 모든 숫자가 full raw CSV와 manifest로 추적된다.
- [ ] seed 누락과 중복 row가 없다.
- [ ] PDR 외 delay/deadline/energy/input/drop을 함께 보고한다.
- [ ] Teacher가 배포 정책이나 optimal upper bound로 표현되지 않았다.
- [ ] simulation unit, proxy, 2D·single-packet·link-model 한계를 공개했다.
- [ ] 그림·표·본문의 값과 호출 순서가 일치한다.
- [ ] DOI와 인용 문맥을 사람이 확인했다.
- [ ] code/data availability 문구가 실제 공개 상태와 같다.
- [ ] AI 사용 공개가 실제 로그와 저널 정책을 반영한다.
- [ ] Word/PDF 변환 후 수식, 표, 그림, 인용, 특수문자를 시각 검수했다.

### 완료 기준

`submission_checklist.md`의 모든 항목이 `충족`, `미충족`, `수동 확인` 중 하나로 표시되고, 미충족 항목이 해결되기 전에는 제출하지 않는다.

### 인간 승인

저자·기여·순서, COI, 윤리, 공개 범위, 실제 업로드 파일을 최종 승인한다.

---

## 4. 단계별 산출물 요약

| Step | 핵심 산출물 | 다음 단계 게이트 |
| ---: | --- | --- |
| 1 | project map, session state | 범위 승인 |
| 2 | research question, claim boundary | 질문 승인 |
| 3 | simulator/scenario/metric audit | 계약 정합성 |
| 4 | exploratory hypotheses | 문헌 검증 후보 선택 |
| 5 | search protocol, evidence table | gap 승인 |
| 6 | confirmed hypotheses | 가설·metric 동결 |
| 7 | experiment plan | config·비용 승인 |
| 8 | test/smoke report | full 실행 허용 |
| 9 | checkpoint/bundle provenance | upstream 승인 |
| 10 | seed별 full artifact | seed 완료 |
| 11 | merged full artifact | grid 완전성 |
| 12 | result audit | 논문 사용 수치 승인 |
| 13 | claim-evidence map, figures/tables | 메인 산출물 승인 |
| 14 | reviewed abstract | 제목·수치 승인 |
| 15 | verified references, RIS | 인용 원문 승인 |
| 16 | writing guide, manuscript | 모든 문장 승인 |
| 17 | target journal comparison | 저널 선택 |
| 18 | four reviews, revision log | P0/P1 해결 |
| 19 | submission package/checklist | 최종 제출 승인 |

---

## 5. 에이전트에게 요청할 때 쓰는 공통 프롬프트

```text
[연구 의도]
이번 작업의 목적은 {목적}이다.
내가 결정해야 할 것은 {결정}이다.

[정본 파일]
- @{code/config}: 실행 계약
- @{raw/manifest}: 결과 정본
- @{plan}: 승인된 계획
- @{paper/guideline}: 원고 또는 제출 규칙

[포함 범위]
{수행할 일}

[제외 범위]
- 확인되지 않은 수치·인용·완료 상태를 만들지 않는다.
- smoke와 full을 섞지 않는다.
- 다른 phase/campaign의 조건이 같다고 가정하지 않는다.
- 기존 checkpoint와 artifact를 덮어쓰지 않는다.
- GPU/Colab 실행은 승인 전 시작하지 않는다.

[검증]
- 모든 주장의 근거 경로를 표시한다.
- 실패·누락·불확실성을 보고한다.
- 코드, raw result, manifest, 문서가 충돌하면 충돌을 보고한다.

[산출물]
- {경로 1}
- {경로 2}
- 완료 기준: {검증 가능한 조건}

[인간 승인]
{중요 결정} 전에 선택지와 장단점을 보고하고 멈춘다.
```

---

## 6. 세션 시작·종료 체크리스트

### 시작

- [ ] branch, commit, dirty files를 확인했다.
- [ ] 이번 세션 목표를 하나로 적었다.
- [ ] 정본 파일과 초안 파일을 구분했다.
- [ ] phase, mode, seed, checkpoint, output path를 확인했다.
- [ ] GPU 비용과 외부 업로드 승인 여부를 확인했다.
- [ ] 기존 artifact와 사용자 변경을 보호한다.

### 종료

- [ ] 생성·수정 파일 목록을 확인했다.
- [ ] 실행 명령, 환경, seed, 결과 경로를 기록했다.
- [ ] test/smoke/full 상태를 정확히 표시했다.
- [ ] 실패와 미확인 항목을 남겼다.
- [ ] 변경된 결정과 이유를 decision log에 기록했다.
- [ ] 대화에만 남은 중요한 판단을 파일로 옮겼다.
- [ ] 다음 세션의 첫 작업을 적었다.

---

## 7. 현재 하네스 사용 전 정합성 점검 사항

현재 `.claude/` 하네스는 유용한 context, rules, agents, skills, templates를 포함하지만 다음을 먼저 점검해야 한다.

1. 일부 `.claude/rules/*.md`의 path glob이 `ResearchAIWorkspace/...`를 가리킨다. 현재 핵심 소스가 저장소 루트의 `implementations/`, `tests/`, `scripts/`에 있으므로 실제 경로에 rule이 적용되는지 확인해야 한다.
2. `.claude/context/method-lineage.md`와 일부 eval은 Global Teacher/Student distillation의 시작을 Phase 7로 설명하지만, 실행 구현은 Phase 3 Teacher와 Phase 4 distillation부터 존재한다. Phase 7은 일반화 hardening이다.
3. `.claude/context`보다 코드와 YAML을 우선한다. 불일치를 고치기 전에는 해당 문서를 단독 정본으로 사용하지 않는다.
4. 현재 seed 42 chunk 로그가 존재해도 Phase 13 완료를 뜻하지 않는다. seed별 결과 ZIP과 merged manifest를 확인해야 한다.

이 네 항목을 해결한 뒤 하네스를 지속적인 연구 운영 체계로 사용한다.

---

## 8. 최종 운영 공식

```text
GLOBE-ROUTING 연구 하네스
= 코드·config·scenario의 명시적 컨텍스트
+ seed·checkpoint·artifact의 재현 계보
+ test → smoke → full → merge → audit 검증 루프
+ claim → raw result → table/figure → manuscript 추적성
+ 비용·가설·주장·투고에 대한 인간 승인
```

가장 먼저 실행할 최소 작업은 다음 세 가지다.

1. Step 1의 현재 상태 기록
2. Step 2의 연구 질문·주장 경계 승인
3. Step 7의 Phase 13 experiment plan 동결

그 후에만 checkpoint 검증과 full campaign으로 넘어간다.
