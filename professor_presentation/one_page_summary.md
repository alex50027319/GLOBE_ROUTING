# One Page Summary

## 연구 제목

**GLOBE++ / Risk-Switch Lite-GLOBE-P: Global-to-Local Policy Distillation for Decentralized FANET Routing**

## 한 줄 요약

FANET에서 전역 topology를 직접 사용할 수 없는 UAV가, offline global teacher의 라우팅 판단을 local student policy로 증류받고, 위험 상황에서는 predictive branch로 전환하여 안정적인 next-hop routing을 수행하도록 하는 연구다.

## 문제 배경

UAV 네트워크는 노드가 빠르게 이동하기 때문에 링크가 자주 바뀌고, GPSR 같은 단순 지리적 라우팅은 routing hole과 link break에 취약하다. 중앙집중형 라우팅은 전체 상태 수집 비용이 크고, 완전 분산 라우팅은 local observation만 사용하기 때문에 장기 경로 안정성을 놓치기 쉽다.

## 제안 구조

1. **Global Teacher**
   - 학습 단계에서만 전역 graph와 packet 상태를 본다.
   - actor-critic 기반 PPO로 next-hop policy를 학습한다.

2. **Local Student**
   - 실행 단계에서 각 UAV가 사용할 경량 policy다.
   - teacher의 행동분포를 forward-KL policy distillation으로 학습한다.

3. **Risk-Switch Lite-GLOBE-P**
   - normal student가 선택한 next-hop이 위험하면 predictive student branch로 전환한다.
   - 위험 기준은 link margin, predicted lifetime, onward stability 등이다.

4. **Lite-GLOBE-P+**
   - Phase 13에서 link-loss-aware danger, top-k onward stability, energy-aware tie-breaking, drop suppression을 추가했다.
   - 현재 audit 기준 full multi-seed 최종 결과는 추가 확인이 필요하다.

## 핵심 수식

Teacher policy:

$$
\pi_T(a \mid s_t)
$$

Student policy:

$$
\pi_S(a \mid o_{u,t})
$$

Policy distillation:

$$
\mathcal{L}_{KD}
=
D_{KL}
\left(
\pi_T(\cdot \mid s_t)
\Vert
\pi_S(\cdot \mid o_{u,t})
\right)
$$

Risk switch:

$$
a^* =
\begin{cases}
a_P, & D(a_N) > \tau \\
a_N, & D(a_N) \le \tau
\end{cases}
$$

## 현재 근거

- Phase 12 full multi-seed 결과 존재.
- Phase 10 external RL baseline 결과 존재.
- Phase 13 P+ code, Colab bundle, smoke result 존재.
- KCI 스타일 draft PDF 존재.

## 현재 주의점

- Phase 13 P+ full multi-seed 결과는 최종 확정 전이다.
- 현재 main student는 MLP 기반 후보 스코어러이며, local GNN student는 향후 확장 claim으로 다뤄야 한다.
- energy 값은 실제 Joule이 아니라 simulator-level proxy다.

