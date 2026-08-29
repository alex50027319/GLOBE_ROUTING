# Method Lineage

## Phase 7

Global Teacher와 Local Student의 정책 증류 구조를 명시적으로 도입했다.
Teacher는 전역 토폴로지를 보고, Student는 로컬 관측으로 Teacher의 행동 분포를
KL divergence 기반으로 모방한다.

## Phase 8

기존 Phase 7 Teacher를 활용해 geographic prior에 학습 residual을 결합하는
Geographic Residual Student를 최적화했다. 일반·노드 수 변화·구조적 홀에서
강하지만 미래 링크 단절을 명시적으로 예측하는 능력은 제한적이다.

## Phase 10

IQMR, DRAMA, Evo-QGeo 등 외부 baseline과의 비교를 수행한다. Phase 12의 직접
실험 구성으로 오해하지 않는다.

## Phase 11

Phase 8 기반 정책에 미래 링크 단절 위험을 고려하는 predictive branch를 추가한다.

## Phase 12

normal branch와 predictive branch를 위험도에 따라 선택하는 Risk-Switch를
보정하고 내부 ablation을 평가한다. 실제 Phase 12 figure에는 IQMR/DRAMA가 없다.

## Phase 13

Risk-Switch Lite-GLOBE-P+의 게이트, redundancy, loss keep, energy tie,
drop suppression 요소를 확장해 평가한다.
