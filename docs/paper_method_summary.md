# GLOBE++ / Lite-GLOBE-P+ — Problem Formulation & Method Summary

이 문서는 원래 `paper/sections/03_problem_formulation.tex`, `paper/sections/04_method.tex`에
있던 내용 중 구현(implementations/lite_globe)을 이해하고 유지보수하는 데 필요한 핵심만
Markdown으로 옮긴 것이다. 관련 연구 비교, 실험 설계, 결과 분석 등 구현과 직접 관련 없는
절은 제외했다. 전체 논문 원문(LaTeX 빌드, 그림, 참고문헌 등)은 이 브랜치에 포함하지 않는다.

## 1. Problem Formulation (Dec-POMDP)

FANET(Flying Ad Hoc Network)의 분산 라우팅 문제를 Dec-POMDP로 정식화한다. 학습 시점의
전역 정보와 배포 시점의 지역 행동을 분리하는 것이 핵심이다.

### 1.1 Network & Communication Model

- 시각 $t$의 FANET 토폴로지는 시변 방향 그래프 $\mathcal{G}_t = (\mathcal{V}, \mathcal{E}_t)$.
  $\mathcal{V} = \{1, \ldots, N\}$: UAV(라우터) 노드 집합.
- $(u, v) \in \mathcal{E}_t$: $v$가 $u$의 통신 반경 $R_c$ 내에 있고 채널 품질이 전송을
  허용할 때 존재하는 방향 링크.
- 각 UAV $u$는 위치 $p_u(t) \in \mathbb{R}^3$, 속도 $v_u(t) \in \mathbb{R}^3$, 큐 점유
  $q_u(t) \in [0, Q_{\max}]$, 잔여 에너지 $E_u(t)$로 특징지어진다.
- 목적지 $d$로 향하는 패킷이 노드 $u$에 있을 때, 행동 공간은
  $\mathcal{A}_u(t) = \mathcal{N}_u(t) \cup \{\text{DROP}\}$, 여기서
  $\mathcal{N}_u(t) = \{v \in \mathcal{V} : (u, v) \in \mathcal{E}_t\}$는 1-hop 이웃,
  `DROP`은 버퍼 오버플로우/데드라인 만료/라우팅 실패로 인한 명시적 폐기 행동.

### 1.2 Partial Observation

로컬 정책은 전역 그래프 $\mathcal{G}_t$나 원거리 노드의 미래 궤적에 접근할 수 없다.
노드 $u$의 로컬 관측:

$$
o_u(t) = \mathcal{O}(s_t, u) = \{x_u(t), \{x_i(t): i \in \mathcal{N}_u(t)\}, \{x_{u,i}(t): i \in \mathcal{N}_u(t)\}, x_d(t)\}
$$

- $x_u(t) = [p_u(t), v_u(t), q_u(t), E_u(t)]$: 자기 상태.
- $x_i(t) = [p_i(t), v_i(t), q_i(t)]$: 주기적 1-hop 비콘으로 전파되는 이웃 상태.
- $x_{u,i}(t) = [m_i(t), \ell_i(t)]$: 링크 엣지 특징. $m_i$ = RSSI 신호 마진, $\ell_i$ = 예측
  링크 수명.
- $x_d(t) = [p_d(t)]$: 목적지 좌표.

학습 중에는 특권을 가진 전역 teacher $\pi_T(a \mid s_t)$가 전체 상태 $s_t$를 관측하고,
배포되는 student $\pi_S(a \mid o_u(t))$는 제한된 로컬 관측 $o_u(t)$만 사용한다.

### 1.3 Routing Objective & Metrics

$$
\max_{\pi} \; \mathbb{E}_{\pi}\Big[\sum_{t=0}^{T_{\max}} \gamma^t (R_{\text{succ}} - \alpha D_t - \beta E_t - \gamma C_t - \xi H_t)\Big]
$$

$R_{\text{succ}}$: 전달 성공 보상, $D_t$: 지연 페널티, $E_t$: 에너지 비용, $C_t$: 제어
오버헤드, $H_t$: 홉당 페널티, $\gamma \in [0,1)$: 할인율.

평가 지표: PDR, Deadline 준수율, 평균/95백분위 지연($\bar D$, $\text{Delay}_{95}$), 처리율,
제어 오버헤드, 에너지, teacher-student 정책 정렬(KL divergence $D_{\text{KL}}(\pi_T \Vert \pi_S)$),
정책 성능 격차 $\Delta J = J(\pi_T) - J(\pi_S)$.

이 지표들은 `implementations/lite_globe/evaluation/`의 costs.py, statistics.py,
reporting.py 계열 모듈이 계산한다.

## 2. Proposed Method (Lite-GLOBE-P+)

세 단계로 구성:

1. **Offline Privileged Training**: 전역 그래프 정보를 활용해 teacher 정책 $\pi_T$를 강화학습으로
   학습 (`implementations/lite_globe/algorithms/teacher_trainer.py`, `models/teacher_gnn.py`).
2. **Global-to-Local Knowledge Distillation**: 학습된 teacher를 고정하고, 1-hop 후보 특징만
   사용하는 student $\pi_S$로 soft next-hop 확률을 증류 (`algorithms/distillation.py`,
   `models/student_policy.py`).
3. **Predictive Risk Switching (PRS) 실행**: 배포 시 실시간 링크/큐 위험 지표에 따라 nominal
   student 정책과 predictive safety branch 사이를 전환 (`baselines/risk_oracle.py`,
   Phase 12/13 관련 모듈).

### 2.1 Global Teacher (RL, PPO)

- Teacher $\pi_T(a \mid s_t; \theta_T)$는 전역 상태 $s_t = (\mathcal{G}_t, p_t, v_t, q_t, d)$를 관측.
- 보상: $r_t = \alpha_{\text{succ}} R_{\text{succ}} - \alpha_D D_t - \alpha_E E_t - \alpha_H H_t - \alpha_F F_t$.
- Clipped PPO 목적함수:
  $\mathcal{L}_{\text{PPO}}(\theta_T) = -\mathbb{E}_t[\min(\rho_t A_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t)]$,
  $\rho_t = \pi_T(a_t\mid s_t;\theta_T)/\pi_T(a_t\mid s_t;\theta_T^{\text{old}})$.
- 전체 손실: $\mathcal{L}_T = \mathcal{L}_{\text{PPO}} + c_V \mathcal{L}_V - c_H \mathcal{H}(\pi_T)$.

### 2.2 Global-to-Local Policy Distillation

- Teacher/student 로짓을 온도 $T$로 softmax하여 soft target $y_T, y_S$ 생성.
- 증류 손실:
  $\mathcal{L}_{\text{KD}} = \lambda_{\text{KL}} T^2 D_{\text{KL}}(y_T \Vert y_S) + \lambda_{\text{CE}}\,\text{CE}(a_T, \pi_S) + \lambda_O \mathcal{L}_{\text{oracle}}$,
  $a_T = \arg\max_a \pi_T(a\mid s_t;\theta_T^\star)$는 teacher hard action, $\mathcal{L}_{\text{oracle}}$은
  최단경로 이탈에 대한 페널티(unseen topology에서 실행 오류를 제한).

### 2.3 Predictive Candidate Features & P+ Extension (Phase 13)

기본 안전 분기의 위험 특징: $x_i^{\text{risk}} = [m_i, \ell_i, q_i, o_i]$
($o_i$ = 이웃 $i$의 outgoing 링크 중 최선 예측 수명).

P+ 확장 특징: $x_i^{+} = [\bar o_{i,\text{topk}}, \rho_i, p_{i,\text{keep}}, \eta_i]$

- $\bar o_{i,\text{topk}}$: 이웃 $i$의 상위 $k$개 outgoing 링크 평균 수명.
- $\rho_i$: 정규화된 outgoing 링크 중복도(대체 경로 밀도).
- $p_{i,\text{keep}}$: 확률적 채널 모델 기반 링크 생존 확률.
- $\eta_i = 1 - (d_i/R_c)^2$: 거리 기반 에너지 효율 proxy.

### 2.4 Danger Score, Safety Utility, Risk-Switch Decision

$$
D_i = [g_m - m_i]_+ + [g_\ell - \ell_i]_+ + [g_o - o_i]_+ + [g_k - \bar o_{i,\text{topk}}]_+ + 0.5[g_\rho - \rho_i]_+ + [g_p - p_{i,\text{keep}}]_+
$$

$Q_i = m_i + \ell_i + o_i + \bar o_{i,\text{topk}} + 0.5\rho_i + p_{i,\text{keep}}$
(Safety Utility), Safety Gain $G(a_P, a_N) = Q_{a_P} - Q_{a_N}$.

PRS 활성화 지시자:

$$
S = \mathbb{1}\big[a_N = \text{DROP} \;\lor\; D_{a_N} > \tau \;\lor\; (a_P \ne a_N \land G(a_P,a_N) > \delta \land D_{a_P} < D_{a_N})\big]
$$

최종 행동: $a^\star = a_P$ if $S=1$, else $a_N$. ($\tau$: 위험 허용 임계값, $\delta$: 최소
안전 이득)

### 2.5 Energy-Aware Tie Breaking & Drop Suppression

- Energy-aware tie break: $z_i^{P+} = z_i^{P} + \lambda_E \eta_i$ (동률일 때 더 짧고
  에너지 효율적인 링크 선호, $\lambda_E > 0$).
- Drop suppression: 안전 게이트를 통과하는 이웃이 하나라도 있으면
  $z_{\text{DROP}}^{P+} = z_{\text{DROP}} - \lambda_D$ ($\lambda_D > 0$, 조기 폐기 방지).

## 3. 구현 매핑 참고

| 논문 구성 요소 | 관련 구현 위치 |
| --- | --- |
| Teacher GNN / PPO | `implementations/lite_globe/models/teacher_gnn.py`, `algorithms/teacher_trainer.py` |
| Student policy / masking | `implementations/lite_globe/models/student_policy.py`, `models/student_actor_critic.py`, `models/masking.py` |
| Distillation | `implementations/lite_globe/algorithms/distillation.py`, `data/distillation_dataset.py` |
| Local fine-tuning | `implementations/lite_globe/algorithms/student_finetune.py` |
| Risk-switch / P+ (Phase 12-13) | `implementations/lite_globe/run_phase12.py`, `run_phase13.py`, `baselines/risk_oracle.py`, `evaluation/phase12_reporting.py`, `evaluation/phase13_reporting.py` |
| 평가 지표 (PDR, delay, KL 등) | `implementations/lite_globe/evaluation/costs.py`, `statistics.py`, `records.py` |

Phase별 구현 범위에 대한 좀 더 자세한 설명은 `implementations/lite_globe/README.md`와
`implementations/lite_globe/assumptions.md`를 참고한다.
