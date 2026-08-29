# SwitchGLOBE 방법론과 구현 대응

## 1. 문제 정의

시각 $t$의 FANET을 시변 그래프 $\mathcal{G}_t=(\mathcal{V},\mathcal{E}_t)$로
정의한다. 목적지 $d$의 패킷이 노드 $u$에 있을 때 행동 공간은 현재 1-hop 이웃과
명시적 DROP으로 제한된다.

$$
\mathcal{A}_u(t)=\mathcal{N}_u(t)\cup\{\mathrm{DROP}\}.
$$

학습 중 privileged Teacher $\pi_T(a\mid s_t)$는 전역 그래프를 관측한다. 배포되는
Student $\pi_S(a\mid o_u(t))$는 자기 상태, 1-hop 이웃과 링크, 패킷 상태, 제한된
forwardability/risk feature만 관측한다. Teacher도 현재 노드의 유효한 1-hop 행동만
선택할 수 있다.

## 2. PPO Global Teacher

Teacher actor-critic은 clipped PPO로 학습된다.

$$
\mathcal{L}_{\mathrm{PPO}}=-\mathbb{E}_t\left[
\min\left(\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)\right].
$$

관련 코드:

- `models/teacher_gnn.py`
- `algorithms/ppo.py`
- `algorithms/teacher_trainer.py`
- `experiments/phase7_campaign.py`

## 3. Global-to-Local Knowledge Distillation

Teacher와 Student logit에 동일한 action mask와 temperature를 적용한다. 기본 목적은
masked forward KL이며, 실제 Geo-Residual/Predictive 학습에서는 teacher action,
shortest-path oracle, risk-aware oracle cross-entropy를 설정에 따라 결합한다.

$$
\mathcal{L}=D_{\mathrm{KL}}(\pi_T^T\Vert\pi_S^T)
+\lambda_T\mathcal{L}_{T\text{-action}}
+\lambda_O\mathcal{L}_{\text{oracle}}
+\lambda_R\mathcal{L}_{\text{risk-oracle}}.
$$

Dataset은 개별 hop이 아니라 scenario와 episode seed group으로 분리한다.

관련 코드: `algorithms/distillation.py`, `data/generate_teacher_data.py`,
`data/distillation_dataset.py`.

## 4. Normal branch: Geo-Residual Student

후보 $i$의 목적지 방향 geographic progress $g_i$ 위에 shared local MLP가 출력하는
residual을 결합한다.

$$
z_i^N=\alpha g_i+r_\theta(o_u,i).
$$

forwardability feature와 구조적 visited-action mask는 greedy dead end와 loop를 줄인다.
관련 구현은 `GeographicResidualStudentPolicy`와 `phase8_campaign.py`다.

## 5. Predictive branch

Predictive Student는 후보별 네 가지 local risk feature를 사용한다.

$$
x_i^{risk}=[m_i,\ell_i,q_i,o_i],
$$

여기서 $m_i$는 link margin, $\ell_i$는 현재 링크의 예측 수명, $q_i$는 queue
headroom, $o_i$는 최선 onward link의 예측 수명이다. Geographic/forwardability prior,
predictive bonus, gate violation penalty와 bounded residual을 결합해 학습한다.

관련 구현은 `LiteGlobePStudentPolicy`, `observation.py`, `phase11_campaign.py`다.

## 6. SwitchGLOBE

최종 배포 정책은 normal action $a_N$과 residual을 끈 predictive-prior action $a_P$를
동시에 계산한다. 후보의 위험도는 다음과 같다.

$$
D_i=[g_m-m_i]_+ + [g_\ell-\ell_i]_+ + [g_o-o_i]_+.
$$

다음 중 하나를 만족하면 predictive branch를 선택한다.

1. normal branch가 DROP을 선택함
2. $D_{a_N}>\tau$
3. $a_P\ne a_N$이고 safety gain이 0.10보다 크며 $D_{a_P}<D_{a_N}$

그 외에는 normal branch를 유지한다. Calibration은 일반 및 structural-hole PDR이
normal 기준에서 0.005 이상 하락하지 않는 후보 중 predictive PDR이 가장 높은 gate를
선택한다.

관련 구현:

- `models/student_policy.py::SwitchGlobePolicy`
- `experiments/phase12_campaign.py`(역사적 구현명)
- `run_switchglobe.py`
- `evaluation/phase12_reporting.py`(역사적 구현명)

## 7. 주장 범위

SwitchGLOBE는 PPO로 직접 마지막 단계의 weight를 갱신하지 않는다. PPO Teacher로부터
증류된 두 Student를 사용하므로 RL-derived 알고리즘이며, 최종 단계의 새 최적화는
risk-switch calibration이다. Phase 13/P+의 redundancy, loss-keep, energy tie 및 DROP
suppression은 최종 SwitchGLOBE 정의에 포함하지 않는다.
